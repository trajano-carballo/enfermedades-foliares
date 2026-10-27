"""La prueba de humo de Colab de punta a punta en CPU, con un PlantVillage sintético (sin red ni datos reales)."""

import json

import pytest  # noqa: F401  (fixtures)
import torch

from foliares.models.mobilenet import MobileNetV3SmallGAP
from foliares.training import prueba_humo as ph
from foliares.training.checkpoint import guardar_checkpoint

CFG = {"experimento": "humo_test", "semilla": 42, "determinista": False, "n_imagenes": 8, "tamano": 64,
       "pesos_imagenet": False, "lr": 0.01, "momentum": 0.9, "pasos_antes": 2, "pasos_despues": 2,
       "tolerancia_perdida": 1e-3, "sonda_determinismo": True, "pasos_sonda": 1}


@pytest.fixture(autouse=True)
def restaurar_modos_globales():
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


def _fase_1(tmp_path, pv_sintetico):
    splits, data_root = pv_sintetico
    salida, ckpt = tmp_path / "salida", tmp_path / "ckpt"
    return salida, ckpt, ph.fase_entrenar(CFG, tmp_path, splits, data_root, salida, ckpt, "r1")


def test_fase_1_y_2_reanudan_igual_que_la_corrida_continua(tmp_path, pv_sintetico):
    splits, data_root = pv_sintetico
    salida, ckpt, _ = _fase_1(tmp_path, pv_sintetico)
    f2 = ph.fase_reanudar(CFG, tmp_path, splits, data_root, salida, ckpt, "r1")
    assert f2["reanudacion_ok"] and f2["igual_bit_a_bit"] and f2["max_abs_dif_perdidas"] == 0.0
    assert all(f2["comprobaciones"].values())
    assert f2["semillas"]["semilla"] == CFG["semilla"] + 1                   # semilla distinta a propósito


def test_los_json_tienen_lo_que_exige_claude_md(tmp_path, pv_sintetico):
    splits, data_root = pv_sintetico
    salida, ckpt, _ = _fase_1(tmp_path, pv_sintetico)
    ph.fase_reanudar(CFG, tmp_path, splits, data_root, salida, ckpt, "r1")
    for nombre in ("fase1_entrenar.json", "fase2_reanudar.json"):
        d = json.loads((salida / "humo_test" / "r1" / nombre).read_text(encoding="utf-8"))
        assert {"config", "commit", "cambios_sin_commitear", "sha256_splits", "semillas", "torch", "dispositivo",
                "tiempo_s", "memoria_pico_cpu_mb", "memoria_pico_gpu_mb"} <= set(d)
        assert d["config"]["experimento"] == "humo_test" and "plantvillage_particion.csv" in d["sha256_splits"]


def test_el_lote_es_solo_train_de_plantvillage(tmp_path, pv_sintetico):
    _, _, f1 = _fase_1(tmp_path, pv_sintetico)
    assert f1["n_imagenes"] == 8 and all("/train_" in r and r.startswith("raw/plantvillage/") for r in f1["rutas_lote"])
    assert f1["uso_de_datos"].endswith("particion == train")


def test_sonda_de_determinismo_en_cpu_es_exacta(tmp_path, pv_sintetico):
    _, _, f1 = _fase_1(tmp_path, pv_sintetico)
    s = f1["sonda_determinismo"]
    assert s["determinista_False"]["igual_bit_a_bit"] and s["determinista_True"]["igual_bit_a_bit"]


def test_un_checkpoint_que_no_es_el_de_la_fase_1_no_se_acepta(tmp_path, pv_sintetico):
    splits, data_root = pv_sintetico
    salida, ckpt, _ = _fase_1(tmp_path, pv_sintetico)
    otro = MobileNetV3SmallGAP(2)                                            # otro modelo en el mismo lugar del checkpoint
    guardar_checkpoint(ckpt / ph.NOMBRE_CKPT, otro, torch.optim.SGD(otro.parameters(), lr=0.1), paso=2)
    f2 = ph.fase_reanudar(CFG, tmp_path, splits, data_root, salida, ckpt, "r1")
    assert f2["reanudacion_ok"] is False                                     # el sha256 y los pesos cargados no coinciden
    assert f2["comprobaciones"]["checkpoint_sha256_igual_al_de_la_fase_1"] is False
    assert f2["comprobaciones"]["pesos_cargados_iguales_a_los_guardados"] is False
