"""Sesión 12, fase 1: medición informativa en PV-val (no toca ningún test)."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from foliares.eval import medicion_pv_val as M
from foliares.utils.paths import cargar_rutas

REPO = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO / "configs" / "06_app.yaml").read_text(encoding="utf-8"))
CAL = yaml.safe_load((REPO / CFG["calibracion"]).read_text(encoding="utf-8"))
MODELO = yaml.safe_load((REPO / CAL["config_modelo"]).read_text(encoding="utf-8"))
LEIDOS: list = []
ESCALA = [200.0]


class _DS:
    def __init__(self, splits, data_root, dataset, particion, rol, *a, **k):
        LEIDOS.append((dataset, particion, rol))
        self.y = np.repeat(np.arange(12), 20)
        self.z = np.zeros((len(self.y), 12), dtype=np.float32)
        self.z[np.arange(len(self.y)), self.y] = ESCALA[0]      # siempre acierta; el margen decide si supera τ con la T de la config


def _falsos(monkeypatch):
    LEIDOS.clear()
    monkeypatch.setattr(M, "DatasetParticion", _DS)
    monkeypatch.setattr(M, "cargar_modelo_base", lambda *a: None)
    monkeypatch.setattr(M, "predecir", lambda m, ds, b, d: {"logits": ds.z, "y": ds.y})
    monkeypatch.setattr(M, "_copiar_al_repo", lambda *a: None)


@pytest.mark.parametrize("escala,cobertura", [(200.0, 1.0), (0.1, 0.0)])
def test_solo_lee_pv_val_en_rol_validacion_y_cuenta_las_que_superan_tau(monkeypatch, tmp_path, escala, cobertura):
    _falsos(monkeypatch)
    ESCALA[0] = escala
    R = cargar_rutas()
    cal = {**CAL, "bootstrap": {**CAL["bootstrap"], "replicas": 30}}
    d = M.medir_pv_val(CFG, cal, MODELO, REPO, R.splits, R.data_root, tmp_path, "r")
    assert LEIDOS == [("plantvillage", "val", "validacion")]
    assert d["resultados"]["cobertura"]["estimacion"] == cobertura and d["n_pv_val"] == 240
    if cobertura:
        assert d["resultados"]["acierto_respondidas"]["estimacion"] == 1.0
    else:                                        # §g.7: sin respondidas el acierto no está definido
        assert d["resultados"]["acierto_respondidas"]["estimacion"] is None
    ref = d["referencia_sin_calibrar_T1"]                  # T = 1: con logits de margen 0,1 la p máx es ~1/12 (no responde) y con 200 sí
    assert set(ref) == {"cobertura", "n_respondidas", "acierto_respondidas"} and ref["cobertura"]["estimacion"] == (1.0 if escala > 1 else 0.0)
    for clave in ("commit", "cambios_sin_commitear", "config", "sha256_splits", "semillas", "torch", "dispositivo", "tiempo_s",
                  "memoria_pico_cpu_mb", "temperatura", "tau", "confianza_maxima_calibrada"):
        assert clave in d
    assert json.loads((tmp_path / "06_app" / "r" / "pv_val_informativo.json").read_text(encoding="utf-8"))["fase"] == "pv_val_informativo"


def test_sin_temperatura_en_la_config_no_mide(monkeypatch, tmp_path):
    _falsos(monkeypatch)
    R = cargar_rutas()
    with pytest.raises(ValueError, match="temperatura"):
        M.medir_pv_val(CFG, {**CAL, "temperatura": None}, MODELO, REPO, R.splits, R.data_root, tmp_path, "r")
    assert LEIDOS == []


def test_el_modulo_no_pide_ninguna_particion_test():
    fuente = (REPO / "src" / "foliares" / "eval" / "medicion_pv_val.py").read_text(encoding="utf-8")
    assert '"test"' not in fuente and "'test'" not in fuente
