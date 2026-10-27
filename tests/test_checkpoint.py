"""Checkpoint reanudable: escritura atómica, carga segura y reanudación idéntica a la corrida continua (CPU)."""

import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from foliares.models.mobilenet import MobileNetV3SmallGAP
from foliares.training.checkpoint import cargar_checkpoint, guardar_checkpoint
from foliares.utils.seeds import fijar_semillas


def _armar(semilla):
    fijar_semillas(semilla)
    m = MobileNetV3SmallGAP(n_clases=4).train()
    return m, torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)


def _datos():
    g = torch.Generator().manual_seed(0)
    return torch.randn(4, 3, 64, 64, generator=g), torch.randint(0, 4, (4,), generator=g)


def _pasos(m, opt, x, y, k):
    for _ in range(k):
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()


def _pesos(m):
    return [p.detach().clone() for p in m.state_dict().values()]


def test_reanudar_da_lo_mismo_que_seguir_sin_parar(tmp_path):
    x, y = _datos()
    m1, o1 = _armar(1)
    _pasos(m1, o1, x, y, 2)
    guardar_checkpoint(tmp_path / "ck.pt", m1, o1, paso=2)
    _pasos(m1, o1, x, y, 2)                                        # corrida continua
    m2, o2 = _armar(99)                                            # otro modelo, otra semilla: todo sale del checkpoint
    assert cargar_checkpoint(tmp_path / "ck.pt", m2, o2)["paso"] == 2
    _pasos(m2, o2, x, y, 2)
    assert all(torch.equal(a, b) for a, b in zip(_pesos(m1), _pesos(m2)))


def test_restaura_los_generadores(tmp_path):
    m, o = _armar(1)
    guardar_checkpoint(tmp_path / "ck.pt", m, o, paso=0)
    esperado = (random.random(), float(np.random.rand()), torch.rand(3))
    random.seed(5), np.random.seed(5), torch.manual_seed(5)        # se pierde el estado...
    cargar_checkpoint(tmp_path / "ck.pt", m, o)                    # ...y vuelve con la carga
    obtenido = (random.random(), float(np.random.rand()), torch.rand(3))
    assert esperado[0] == obtenido[0] and esperado[1] == obtenido[1] and torch.equal(esperado[2], obtenido[2])


def test_escritura_atomica_y_registro(tmp_path):
    m, o = _armar(1)
    r = guardar_checkpoint(tmp_path / "sub" / "ck.pt", m, o, paso=3, extra={"run_id": "r1"})
    assert sorted(p.name for p in (tmp_path / "sub").iterdir()) == ["ck.pt"]       # sin temporales
    assert r["bytes"] > 0 and len(r["sha256"]) == 64
    assert cargar_checkpoint(tmp_path / "sub" / "ck.pt", m, o)["extra"] == {"run_id": "r1"}


def test_un_fallo_al_guardar_no_deja_temporales_ni_pisa_el_anterior(tmp_path, monkeypatch):
    m, o = _armar(1)
    guardar_checkpoint(tmp_path / "ck.pt", m, o, paso=1)
    antes = (tmp_path / "ck.pt").read_bytes()

    def falla(*a, **k):
        raise OSError("sesión cortada")

    monkeypatch.setattr(torch, "save", falla)
    with pytest.raises(OSError):
        guardar_checkpoint(tmp_path / "ck.pt", m, o, paso=2)
    assert (tmp_path / "ck.pt").read_bytes() == antes and [p.name for p in tmp_path.iterdir()] == ["ck.pt"]


def test_formato_desconocido_se_rechaza(tmp_path):
    torch.save({"formato": 999}, tmp_path / "mal.pt")
    m, o = _armar(1)
    with pytest.raises(ValueError, match="Formato de checkpoint"):
        cargar_checkpoint(tmp_path / "mal.pt", m, o)


class _TensorEnCuda(torch.Tensor):
    """Simula un tensor en `cuda:0` sin GPU: `.numpy()` falla como en CUDA y `.cpu()` lo devuelve a CPU."""

    def numpy(self, *a, **k):
        raise TypeError("can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.")

    def cpu(self, *a, **k):
        return self.as_subclass(torch.Tensor)


def _como_si_estuviera_en_cuda(obj):
    """Lo que hace `torch.load(map_location=cuda)`: todos los tensores del checkpoint, también los estados de RNG."""
    if isinstance(obj, torch.Tensor):
        return obj.as_subclass(_TensorEnCuda)
    if isinstance(obj, dict):
        return {k: _como_si_estuviera_en_cuda(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_como_si_estuviera_en_cuda(v) for v in obj]
    return obj


def test_los_generadores_se_restauran_aunque_torch_load_deje_los_tensores_en_cuda(tmp_path, monkeypatch):
    # Regresión de la segunda corrida de Colab: cargar con map_location=cuda movía también el estado de RNG a la GPU y
    # `restaurar_rng` fallaba con "can't convert cuda:0 device type tensor to numpy".
    m, o = _armar(1)
    guardar_checkpoint(tmp_path / "ck.pt", m, o, paso=0)
    esperado = (random.random(), float(np.random.rand()), torch.rand(3))
    random.seed(5), np.random.seed(5), torch.manual_seed(5)
    carga_real = torch.load
    # el torch.load real no puede ir a "cuda" sin GPU: carga a CPU y el mock deja los tensores "en cuda"
    monkeypatch.setattr(torch, "load", lambda *a, **k: _como_si_estuviera_en_cuda(carga_real(*a, **{**k, "map_location": "cpu"})))
    cargar_checkpoint(tmp_path / "ck.pt", m, o, map_location="cuda")
    obtenido = (random.random(), float(np.random.rand()), torch.rand(3))
    assert esperado[0] == obtenido[0] and esperado[1] == obtenido[1] and torch.equal(esperado[2], obtenido[2])


def test_por_defecto_se_carga_siempre_a_cpu(tmp_path, monkeypatch):
    # El modelo y el optimizador reciben los tensores igual (`load_state_dict` copia al dispositivo de los parámetros):
    # no hace falta cargar el checkpoint entero en la GPU.
    m, o = _armar(1)
    guardar_checkpoint(tmp_path / "ck.pt", m, o, paso=0)
    vistos = []
    carga_real = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: vistos.append(k.get("map_location")) or carga_real(*a, **k))
    cargar_checkpoint(tmp_path / "ck.pt", m, o)
    assert vistos == ["cpu"]
