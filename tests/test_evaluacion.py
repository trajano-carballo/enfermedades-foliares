"""Evaluación de la sesión 9: regla (a) de la geometría, guarda antes de leer tests, análisis de errores y script."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from foliares.eval import evaluacion as E
from foliares.eval.guarda_accesos import AccesoNoRegistrado

REPO = Path(__file__).resolve().parents[1]
BS = {"replicas": 300, "semilla": 42, "nivel": 0.95}
Y = np.repeat(np.arange(3), 40)


def _con_aciertos(frac, semilla):
    """Predicciones que aciertan aprox. `frac` de las imágenes."""
    g = np.random.default_rng(semilla)
    return np.where(g.random(len(Y)) < frac, Y, (Y + 1) % 3)


def test_con_ninguna_que_califica_queda_aplastar():
    base = _con_aciertos(0.7, 1)
    r = E.regla_geometria(Y, {"aplastar": base, "recorte": base.copy(), "relleno": base.copy()}, "aplastar", 3, BS)
    assert r["geometria_elegida"] == "aplastar" and r["califican"] == [] and "ambiguo" not in r


def test_una_peor_con_el_ic_entero_debajo_de_cero_no_califica():
    r = E.regla_geometria(Y, {"aplastar": Y.copy(), "recorte": _con_aciertos(0.4, 3), "relleno": Y.copy()}, "aplastar", 3, BS)
    c = r["diferencia_contra_aplastar"]["recorte"]
    assert c["ic_alto"] < 0 and "recorte" not in r["califican"] and r["geometria_elegida"] == "aplastar"


def test_con_el_ic_que_incluye_el_cero_no_califica():
    base = _con_aciertos(0.7, 1)
    casi = base.copy()
    casi[:3] = Y[:3]                                     # apenas mejor: el IC pareado de la diferencia incluye el 0
    r = E.regla_geometria(Y, {"aplastar": base, "recorte": casi, "relleno": base.copy()}, "aplastar", 3, BS)
    c = r["diferencia_contra_aplastar"]["recorte"]
    assert c["ic_bajo"] <= 0 <= c["ic_alto"] and r["califican"] == [] and r["geometria_elegida"] == "aplastar"


def test_una_claramente_mejor_reemplaza_a_aplastar():
    r = E.regla_geometria(Y, {"aplastar": _con_aciertos(0.5, 1), "recorte": Y.copy(), "relleno": _con_aciertos(0.5, 2)},
                          "aplastar", 3, BS)
    assert r["geometria_elegida"] == "recorte" and r["califican"] == ["recorte"]


def test_si_dos_califican_gana_la_de_mayor_diferencia_media():
    r = E.regla_geometria(Y, {"aplastar": _con_aciertos(0.3, 1), "recorte": _con_aciertos(0.8, 2), "relleno": Y.copy()},
                          "aplastar", 3, BS)
    d = r["diferencia_contra_aplastar"]
    assert set(r["califican"]) == {"recorte", "relleno"} and r["geometria_elegida"] == "relleno"
    assert d["relleno"]["media_replicas"] > d["recorte"]["media_replicas"]


def test_matriz_normalizada_por_fila():
    assert E.matriz_normalizada([[2, 2], [0, 0]]) == [[0.5, 0.5], [0.0, 0.0]]


def test_analisis_de_errores_por_especie():
    nombres = ["Potato___Early", "Potato___Late", "Tomato___Late"]
    m = [[5, 1, 0],          # 1 error dentro de la especie (papa)
         [0, 4, 3],          # 3 errores entre especies (papa -> tomate)
         [0, 0, 6]]
    a = E.analisis_errores(m, nombres)
    assert (a["errores"], a["misma_especie"], a["entre_especies"]) == (4, 1, 3)
    assert a["clase_que_mas_errores_atrae"] == "Tomato___Late" and a["proporcion_hacia_esa_clase"] == 0.75


def test_los_tests_no_se_evaluan_sin_geometria_fijada(tmp_path):
    cfg = {**yaml.safe_load((REPO / "configs" / "03_brecha.yaml").read_text(encoding="utf-8")), "geometria": None}
    cfg_modelo = yaml.safe_load((REPO / "configs" / "02_modelo_base.yaml").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="Falta fijar `geometria`"):
        E.evaluar_tests(cfg, cfg_modelo, "configs/03_brecha.yaml", tmp_path, tmp_path, tmp_path, tmp_path / "s", "r", {})


def test_la_guarda_corre_antes_de_leer_cualquier_imagen(tmp_path):
    cfg = {**yaml.safe_load((REPO / "configs" / "03_brecha.yaml").read_text(encoding="utf-8")), "geometria": "aplastar"}
    cfg_modelo = yaml.safe_load((REPO / "configs" / "02_modelo_base.yaml").read_text(encoding="utf-8"))
    with pytest.raises(AccesoNoRegistrado):
        E.evaluar_tests(cfg, cfg_modelo, "configs/03_brecha.yaml", tmp_path, tmp_path, tmp_path, tmp_path / "salida", "r",
                        {"PV-test": 1, "PD-test": 2})
    assert not (tmp_path / "salida").exists()


def test_el_script_exige_las_filas_de_acceso_para_los_tests(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "evaluar.py"), "--config", str(REPO / "configs" / "03_brecha.yaml"),
                        "--salida", str(tmp_path), "--fase", "tests"], capture_output=True, text=True, cwd=REPO,
                       env={**os.environ, "DATA_ROOT": str(tmp_path)})
    assert r.returncode == 2 and "exige --fila-accesos-pv y --fila-accesos-pd" in r.stderr
    assert list(tmp_path.iterdir()) == []
