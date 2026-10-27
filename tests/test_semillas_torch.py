"""Rama de torch de `fijar_semillas`: dos corridas cortas de MobileNetV3-Small (forward + backward + paso del
optimizador) con la misma semilla dan salidas idénticas en CPU. Lo que NO se puede garantizar (hilos, GPU) está
documentado en `docs/reproducibilidad.md`; acá solo se afirma lo que se puede comprobar sin GPU."""

import os

import pytest
import torch
import torch.nn.functional as F

from foliares.models.mobilenet import MobileNetV3SmallGAP
from foliares.utils.seeds import fijar_semillas


@pytest.fixture(autouse=True)
def restaurar_estado_global_de_torch():
    hilos = torch.get_num_threads()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    torch.set_num_threads(hilos)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


def _corrida(semilla: int, determinista: bool = False, hilos: int | None = None, pasos: int = 2):
    """Devuelve (logits por paso, pérdidas, gradiente del cabezal, pesos finales)."""
    if hilos is not None:
        torch.set_num_threads(hilos)
    fijar_semillas(semilla, determinista=determinista)
    m = MobileNetV3SmallGAP(n_clases=12).train()
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    g = torch.Generator().manual_seed(semilla)
    x, y = torch.randn(4, 3, 64, 64, generator=g), torch.randint(0, 12, (4,), generator=g)
    logits, perdidas = [], []
    for _ in range(pasos):
        opt.zero_grad()
        salida = m(x)
        perdida = F.cross_entropy(salida, y)
        perdida.backward()
        opt.step()
        logits.append(salida.detach().clone())
        perdidas.append(perdida.item())
    return logits, perdidas, m.fc.weight.grad.clone(), [p.detach().clone() for p in m.parameters()]


def _iguales(a, b) -> bool:
    return (all(torch.equal(x, y) for x, y in zip(a[0], b[0])) and a[1] == b[1] and torch.equal(a[2], b[2])
            and all(torch.equal(x, y) for x, y in zip(a[3], b[3])))


def test_misma_semilla_misma_salida_bit_a_bit_en_cpu():
    assert _iguales(_corrida(42), _corrida(42))


def test_semilla_distinta_cambia_la_salida():
    assert not _iguales(_corrida(42), _corrida(43))


def test_modo_determinista_no_cambia_el_resultado_en_cpu():
    assert _iguales(_corrida(42, determinista=False), _corrida(42, determinista=True))


def test_con_otra_cantidad_de_hilos_solo_cambia_el_redondeo():
    # En CPU el orden de las reducciones depende de los hilos: no hay igualdad bit a bit entre configuraciones de
    # hilos (por eso el JSON de cada corrida registra los hilos), pero la diferencia es de redondeo de float32.
    a, b = _corrida(42, hilos=1), _corrida(42, hilos=2)
    assert all(torch.allclose(x, y, atol=1e-4, rtol=1e-4) for x, y in zip(a[3], b[3]))


def test_fijar_semillas_informa_lo_fijado_y_prepara_cublas(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "x")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    r = fijar_semillas(7)
    assert r["torch"] and r["semilla"] == 7 and r["determinista"] is False and "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    r = fijar_semillas(7, determinista=True)
    assert r["determinista"] is True and os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()


def test_pasar_de_determinista_a_no_determinista_deja_los_dos_estados_en_false():
    # fijar_semillas fija SIEMPRE los dos estados: una llamada con determinista=False no puede heredar el True de una anterior.
    r = fijar_semillas(1, determinista=True)
    assert r["determinista"] is True and torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic
    r = fijar_semillas(1, determinista=False)
    assert r["determinista"] is False
    assert not torch.are_deterministic_algorithms_enabled() and not torch.backends.cudnn.deterministic
