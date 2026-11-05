"""IC por clase desde predicciones guardadas: coherencia con el bootstrap simple y casos con resultado conocido."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foliares.eval import bootstrap as B
from foliares.eval import metricas as M
from foliares.eval.ic_por_clase import ic_de_predicciones, metricas_vector

REPO = Path(__file__).resolve().parents[1]
NOMBRES = ["a", "b", "c"]


def _pred(seed=0):
    g = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), [30, 20, 10])
    p = np.where(g.random(len(y)) < 0.7, y, g.integers(0, 3, len(y)))
    return pd.DataFrame({"etiqueta": y, "pred": p})


def test_el_bootstrap_multiple_da_el_mismo_ic_que_el_bootstrap_simple():
    d = _pred()
    y, p = d.etiqueta.to_numpy(), d.pred.to_numpy()
    multi = B.bootstrap_multiple(y, p, metricas_vector(NOMBRES), 300, 42)
    simple = B.bootstrap(y, p, lambda yy, pp: M.macro_f1(yy, pp, 3), 300, 42)
    for k in ("estimacion", "ic_bajo", "ic_alto", "media_replicas"):
        assert multi["macro_f1"][k] == pytest.approx(simple[k], abs=1e-12)
    simple_rec = B.bootstrap(y, p, lambda yy, pp: M.recall_por_clase(M.matriz_confusion(yy, pp, 3))[1], 300, 42)
    assert multi["recall|b"]["ic_bajo"] == pytest.approx(simple_rec["ic_bajo"], abs=1e-12)


def test_estimaciones_a_mano_y_el_ic_las_rodea():
    d = pd.DataFrame({"etiqueta": [0, 0, 0, 0, 1, 1, 2, 2], "pred": [0, 0, 0, 1, 1, 1, 0, 2]})
    r = ic_de_predicciones(d, NOMBRES, 500, 42, 0.95)
    assert r["n"] == 8 and r["exactitud"]["estimacion"] == pytest.approx(6 / 8)
    a = r["por_clase"]["a"]
    assert a["soporte"] == 4 and a["recall"]["estimacion"] == pytest.approx(3 / 4) and a["precision"]["estimacion"] == pytest.approx(3 / 4)
    c = r["por_clase"]["c"]
    assert c["recall"]["estimacion"] == 0.5 and c["precision"]["estimacion"] == 1.0
    for m in (r["exactitud"], a["recall"], c["f1"]):
        assert m["ic_bajo"] <= m["estimacion"] <= m["ic_alto"] and m["n_replicas"] == 500


def test_el_recall_de_cada_clase_es_estable_con_el_remuestreo_estratificado():
    # Con el remuestreo estratificado, una clase con todos los aciertos tiene recall 1 en TODAS las réplicas (IC degenerado).
    d = pd.DataFrame({"etiqueta": [0] * 10 + [1] * 10, "pred": [0] * 10 + [0] * 10})
    r = ic_de_predicciones(d, ["a", "b"], 200, 42, 0.95)
    assert r["por_clase"]["a"]["recall"]["ic_bajo"] == r["por_clase"]["a"]["recall"]["ic_alto"] == 1.0
    assert r["por_clase"]["b"]["recall"]["estimacion"] == 0.0 and r["por_clase"]["b"]["precision"]["estimacion"] == 0.0


def test_misma_semilla_mismo_resultado():
    a = ic_de_predicciones(_pred(), NOMBRES, 100, 42, 0.95)
    assert a == ic_de_predicciones(_pred(), NOMBRES, 100, 42, 0.95) and a != ic_de_predicciones(_pred(), NOMBRES, 100, 7, 0.95)


def test_el_script_exige_salida_y_los_runs(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "ic_por_clase.py"), "--config", str(REPO / "configs" / "03_brecha.yaml")],
                       capture_output=True, text=True, cwd=REPO, env={**os.environ, "DATA_ROOT": str(tmp_path)})
    assert r.returncode == 2 and "--salida" in r.stderr and "--run-dev" in r.stderr and "--run-tests" in r.stderr
