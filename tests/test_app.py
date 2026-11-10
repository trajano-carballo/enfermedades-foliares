"""Sesión 12, fase 2: la app (camino de una imagen, cartel, equivalencia con la evaluación, sin tests)."""

import importlib.util
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import torch
from PIL import Image

from foliares.data.preproceso import preprocesar
from foliares.eval.calibracion import softmax_t
from foliares.eval.evaluacion import predecir
from foliares.inferencia import predecir_imagen
from foliares.models.mobilenet import MobileNetV3SmallGAP

REPO = Path(__file__).resolve().parents[1]
NOMBRES = [f"Clase___{i}" for i in range(12)]
T = 14.973042339479479


def _app():
    spec = importlib.util.spec_from_file_location("app_gradio", REPO / "app" / "app.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def modelo():
    torch.manual_seed(0)
    return MobileNetV3SmallGAP(12, pesos_imagenet=False).eval()


@pytest.fixture
def imagen(tmp_path):
    ruta = tmp_path / "hoja.png"
    arr = np.random.default_rng(0).integers(0, 255, (180, 260, 3), dtype=np.uint8)
    Image.fromarray(arr).save(ruta)
    return ruta


def test_el_camino_de_una_imagen_con_y_sin_cartel_forzando_tau(modelo, imagen):
    sin = predecir_imagen(modelo, imagen, NOMBRES, T, 0.0, "aplastar")          # τ = 0: siempre responde
    con = predecir_imagen(modelo, imagen, NOMBRES, T, 1.01, "aplastar")         # τ > 1: nunca alcanza
    assert not sin["baja_confianza"] and con["baja_confianza"]
    assert sin["top"] == con["top"] and len(sin["top"]) == 3 and sin["prob_max"] == sin["top"][0][1]
    ps = [p for _, p in sin["top"]]
    assert ps == sorted(ps, reverse=True) and 0 < sum(ps) <= 1
    assert sin["cam"].shape == (224, 224, 3) and sin["cam"].dtype == np.uint8
    assert predecir_imagen(modelo, imagen, NOMBRES, T, 0.5, "aplastar", con_cam=False)["cam"] is None


def test_la_app_y_la_evaluacion_dan_la_misma_prediccion_y_probabilidad(modelo, imagen):
    class _DS:                                    # lo mínimo que usa `predecir` de la evaluación
        etiquetas = torch.zeros(1, dtype=torch.int64)
        def __len__(self): return 1
        def __getitem__(self, i): return preprocesar(imagen, "aplastar", 224), torch.tensor(0)

    ev = predecir(modelo, _DS(), 64, torch.device("cpu"))
    p_ev = softmax_t(ev["logits"], T)[0]
    for con_cam in (True, False):
        app = predecir_imagen(modelo, imagen, NOMBRES, T, 0.5, "aplastar", con_cam=con_cam)
        assert app["top"][0][0] == NOMBRES[int(ev["pred"][0])]
        np.testing.assert_allclose(app["prob_max"], p_ev.max(), atol=1e-6)
        np.testing.assert_allclose([p for _, p in app["top"]], np.sort(p_ev)[::-1][:3], atol=1e-6)


def test_la_funcion_de_la_app_muestra_el_cartel_solo_con_confianza_baja(modelo, imagen):
    a = _app()
    cam, top, cartel = a.crear_responder(modelo, NOMBRES, T, 1.01, "aplastar", 224)(str(imagen))
    assert a.CARTEL in cartel and len(top) == 3 and cam.shape == (224, 224, 3)
    assert a.crear_responder(modelo, NOMBRES, T, 0.0, "aplastar", 224)(str(imagen))[2] == ""
    with pytest.raises(gr.Error):
        a.crear_responder(modelo, NOMBRES, T, 0.5, "aplastar", 224)(None)


def test_la_entrada_acepta_galeria_y_camara_y_se_arma_la_interfaz(modelo):
    a = _app()
    demo = a.construir_demo(a.crear_responder(modelo, NOMBRES, T, 0.5, "aplastar", 224), NOMBRES, T, 0.5)
    imgs = [b for b in demo.blocks.values() if isinstance(b, gr.Image) and b.sources]
    assert imgs and list(imgs[0].sources) == ["upload", "webcam"]


def test_el_aviso_de_alcance_tiene_el_texto_acordado_y_cita_el_run():
    a = _app()
    for frase in ("no un diagnóstico", "consultá a un técnico", "tomate, papa y pimiento", "{n} clases",
                  "La confianza se calibró con fotos de campo, donde el modelo acierta poco. Por eso es baja incluso cuando la respuesta "
                  "es correcta, y el cartel de confianza baja aparece en todas las imágenes probadas. El orden del top-3 sigue siendo "
                  "informativo: con fotos de laboratorio, la primera opción acierta el 97 %."):
        assert frase in a.AVISO.replace("**", "")
    assert "06_app_pv_val_20261110_093958" in (REPO / "app" / "app.py").read_text(encoding="utf-8")


def test_la_app_no_lee_ningun_test_ni_las_particiones():
    for f in (REPO / "app" / "app.py", REPO / "src" / "foliares" / "inferencia.py"):
        fuente = f.read_text(encoding="utf-8")
        for prohibido in ("DatasetParticion", "filas_particion", "splits", "PD-test", "PV-test", '"test"', "'test'"):
            assert prohibido not in fuente, (f.name, prohibido)


def test_gradio_esta_fijada_a_la_version_instalada():
    reqs = (REPO / "requirements.txt").read_text(encoding="utf-8")
    assert f"gradio=={gr.__version__}" in reqs
