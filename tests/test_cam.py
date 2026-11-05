"""CAM: combinación lineal esperada, sin backward; muestra fija; guarda de particiones; figuras (humo)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from foliares.eval import figuras as FIG
from foliares.explain import cam as C
from foliares.explain.muestra import elegir_pv_val, exigir_particion_permitida, seleccionar_pd_dev
from foliares.models.mobilenet import MobileNetV3SmallGAP


def test_cam_es_la_combinacion_lineal_esperada_con_pesos_y_mapas_sinteticos():
    # 2 canales, mapa 2×2; clase 0 pesa (1, 0), clase 1 pesa (0, 2)
    A = torch.tensor([[[[1., 2.], [3., 4.]], [[10., 20.], [30., 40.]]]])          # (1, 2, 2, 2)
    W = torch.tensor([[1., 0.], [0., 2.]])
    assert torch.equal(C.cam(A, W, torch.tensor([0]))[0], torch.tensor([[1., 2.], [3., 4.]]))
    assert torch.equal(C.cam(A, W, torch.tensor([1]))[0], torch.tensor([[20., 40.], [60., 80.]]))
    W2 = torch.tensor([[0.5, 1.0]])
    esperado = 0.5 * A[0, 0] + 1.0 * A[0, 1]
    assert torch.equal(C.cam(A, W2, torch.tensor([0]))[0], esperado)


def test_cam_no_usa_gradientes_ni_llama_a_backward(monkeypatch):
    def prohibido(*a, **k):
        raise AssertionError("el CAM no puede llamar a backward")
    monkeypatch.setattr(torch.Tensor, "backward", prohibido)
    monkeypatch.setattr(torch.autograd, "backward", prohibido)
    A = torch.randn(2, 4, 3, 3, requires_grad=True)
    W = torch.randn(5, 4, requires_grad=True)
    out = C.cam(A, W, torch.tensor([1, 3]))
    assert not out.requires_grad and out.grad_fn is None
    m = MobileNetV3SmallGAP(5).eval()
    r = C.cam_de_modelo(m, torch.randn(2, 3, 64, 64))
    assert not r["cam"].requires_grad and r["cam"].grad_fn is None and r["probs"].grad_fn is None
    assert all(p.grad is None for p in m.parameters())


def test_cam_del_modelo_sale_del_mismo_forward_y_su_media_mas_el_sesgo_da_el_logit():
    m = MobileNetV3SmallGAP(4).eval()
    x = torch.randn(3, 3, 64, 64)
    r = C.cam_de_modelo(m, x)
    assert torch.equal(r["clase_cam"], r["pred"]) and torch.allclose(r["probs"].sum(1), torch.ones(3), atol=1e-5)
    with torch.no_grad():
        logit = r["cam"].mean(dim=(1, 2)) + m.fc.bias[r["pred"]]
        assert torch.allclose(logit, r["logits"].gather(1, r["pred"][:, None])[:, 0], atol=1e-4)
    otra = C.cam_de_modelo(m, x, clases=torch.tensor([0, 0, 0]))
    assert torch.equal(otra["pred"], r["pred"]) and torch.equal(otra["clase_cam"], torch.tensor([0, 0, 0]))


def test_reescalar_normaliza_por_imagen_y_una_constante_queda_en_ceros():
    c = torch.stack([torch.tensor([[0., 1.], [2., 4.]]), torch.full((2, 2), 3.0)])
    r = C.reescalar(c, 8)
    assert r.shape == (2, 8, 8) and r[0].min() == 0 and r[0].max() == pytest.approx(1.0) and r[1].abs().max() == 0


def test_superponer_forma_y_rango():
    img = np.random.default_rng(0).random((8, 8, 3))
    s = C.superponer(img, np.linspace(0, 1, 64).reshape(8, 8))
    assert s.shape == (8, 8, 3) and s.min() >= 0 and s.max() <= 1 and not np.allclose(s, img)
    assert np.allclose(C.superponer(img, np.zeros((8, 8)), alpha=0.0), img)


def _pred(n_por_clase):
    """n_por_clase[c] = (aciertos, errores) de la clase c."""
    filas = []
    for c, (a, e) in enumerate(n_por_clase):
        filas += [{"etiqueta": c, "pred": c}] * a + [{"etiqueta": c, "pred": (c + 1) % len(n_por_clase)}] * e
    return pd.DataFrame(filas)


def test_la_muestra_toma_un_acierto_y_un_error_cuando_hay():
    s = seleccionar_pd_dev(_pred([(5, 4), (3, 3), (4, 2)]), 3, 42)
    for c in range(3):
        sub = s[s.etiqueta == c]
        assert sorted(sub.tipo) == ["acierto", "error"] and len(sub) == 2


def test_la_muestra_cuando_falta_un_tipo():
    s = seleccionar_pd_dev(_pred([(5, 0), (0, 6), (1, 0)]), 3, 42)
    assert s[s.etiqueta == 0].tipo.tolist() == ["acierto", "acierto"]
    assert s[s.etiqueta == 1].tipo.tolist() == ["error", "error"]
    assert s[s.etiqueta == 2].tipo.tolist() == ["acierto"]                       # una sola disponible


def test_la_muestra_es_estable_con_la_semilla_y_cambia_con_otra():
    p = _pred([(20, 20)] * 3)
    a, b = seleccionar_pd_dev(p, 3, 42), seleccionar_pd_dev(p, 3, 42)
    assert a.fila.tolist() == b.fila.tolist() and a.fila.tolist() != seleccionar_pd_dev(p, 3, 7).fila.tolist()
    assert all(p.loc[f, "etiqueta"] == e for f, e in zip(a.fila, a.etiqueta))


def test_pv_val_una_por_clase():
    filas = pd.DataFrame({"etiqueta": [0, 0, 1, 1, 1, 2]})
    s = elegir_pv_val(filas, 3, 42)
    assert s.etiqueta.tolist() == [0, 1, 2] and all(filas.loc[f, "etiqueta"] == e for f, e in zip(s.fila, s.etiqueta))
    assert s.fila.tolist() == elegir_pv_val(filas, 3, 42).fila.tolist()


@pytest.mark.parametrize("dataset,particion", [("plantdoc", "test"), ("plantvillage", "test"), ("plantvillage", "train"), ("plantdoc", "dev")])
def test_la_grilla_rechaza_cualquier_test_y_el_train(dataset, particion):
    if (dataset, particion) == ("plantdoc", "dev"):
        exigir_particion_permitida(SimpleNamespace(dataset=dataset, particion=particion))
    else:
        with pytest.raises(ValueError, match="nunca imágenes de un test"):
            exigir_particion_permitida(SimpleNamespace(dataset=dataset, particion=particion))
    exigir_particion_permitida(SimpleNamespace(dataset="plantvillage", particion="val"))


def _tests_json():
    ic = lambda e: {"estimacion": e, "ic_bajo": e - 0.02, "ic_alto": e + 0.02}  # noqa: E731
    nombres = ["Pepper,_bell___healthy", "Tomato___Tomato_mosaic_virus", "Tomato___Late_blight"]
    m = np.array([[0.8, 0.1, 0.1], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])

    def res(e):
        return {"n": 30, "base": {"matriz_normalizada_por_fila": m.tolist(), "soporte_por_clase": dict(zip(nombres, [10, 10, 10]))},
                "base_macro_f1_ic": ic(e), "trivial_macro_f1_ic": ic(0.01)}
    r = {"PV-test": {**res(0.9), "base_macro_f1_sin_mosaic_ic": ic(0.92)}, "PD-test": res(0.13)}
    return {"resultados": r, "orden_clases": nombres, "brecha_macro_f1_pv_test_menos_pd_test": ic(0.77)}


def test_las_figuras_de_numeros_se_generan(tmp_path):
    t = _tests_json()
    for f in (FIG.figura_brecha(t, tmp_path / "b.png"), FIG.figura_matrices(t, tmp_path / "m.png", ic_json="results/x/ic_por_clase.json")):
        assert f.is_file() and f.stat().st_size > 5000


def test_la_leyenda_de_las_matrices_va_arriba_y_con_la_ruta_del_json(tmp_path, monkeypatch):
    capturadas = []
    cerrar = FIG.plt.close
    monkeypatch.setattr(FIG.plt, "close", lambda f=None: (capturadas.append(f), cerrar(f))[1])
    FIG.figura_matrices(_tests_json(), tmp_path / "m.png", ic_json="results/x/ic_por_clase.json")
    assert capturadas[0]._suptitle.get_text() == "Descriptiva, sin IC; IC por clase en results/x/ic_por_clase.json"
    assert capturadas[0]._suptitle.get_position()[1] > 0.9                              # arriba, no pegada a las etiquetas del eje x


def test_la_figura_de_matrices_exige_citar_el_json_de_los_ic():
    assert FIG.leyenda_matrices("results/03_brecha/r/ic_por_clase.json") ==         "Descriptiva, sin IC; IC por clase en results/03_brecha/r/ic_por_clase.json"
    with pytest.raises(TypeError):
        FIG.figura_matrices(_tests_json(), "x.png")                                # sin ic_json no se puede dibujar


def test_la_grilla_de_cam_se_genera(tmp_path):
    g = np.random.default_rng(0)
    celdas = [{"imagen": g.random((16, 16, 3)), "cam01": g.random((16, 16)), "real": c, "pred": (c + i) % 3, "prob": 0.7, "origen": "x",
               "columna": c, "fila_grilla": (i, f"fila {i}")} for c in range(3) for i in range(3)]
    f = FIG.grilla_cam(celdas, ["Pepper,_bell___healthy", "Tomato___Late_blight", "Tomato___healthy"], tmp_path / "g.png", "prueba")
    assert f.is_file() and f.stat().st_size > 5000


def test_construir_celdas_rechaza_un_dataset_de_test():
    falso = SimpleNamespace(dataset="plantdoc", particion="test")
    with pytest.raises(ValueError, match="nunca imágenes de un test"):
        FIG.construir_celdas(None, falso, pd.DataFrame({"fila": [0], "slot": [0]}), "x", {0: "a"})


def test_el_json_de_tests_real_tiene_lo_que_usan_las_figuras():
    ruta = next(Path(__file__).resolve().parents[1].glob("results/03_brecha/03_brecha_tests_*/tests.json"), None)
    if ruta is None:
        pytest.skip("todavía no hay tests.json")
    t = json.loads(ruta.read_text(encoding="utf-8"))
    assert {"resultados", "orden_clases", "brecha_macro_f1_pv_test_menos_pd_test", "geometria"} <= set(t)
