"""Diagnóstico de PD-dev: casos chicos con resultado conocido y consistencia del mapeo real con docs/mapeo_clases.md."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from foliares.eval import diagnostico as D

REPO = Path(__file__).resolve().parents[1]
NOMBRES = ["a", "b", "c"]


def test_el_mapeo_de_la_config_coincide_con_la_tabla_del_doc():
    doc = D.pares_del_doc((REPO / "docs" / "mapeo_clases.md").read_text(encoding="utf-8"))
    cm = yaml.safe_load((REPO / "configs" / "02_modelo_base.yaml").read_text(encoding="utf-8"))
    assert all(doc.get(c["pv"]) == c["plantdoc"] for c in cm["clases"]), {c["pv"]: doc.get(c["pv"]) for c in cm["clases"]}


def test_permutacion_detecta_un_corrimiento_sistematico():
    # cada clase se predice como la siguiente: la diagonal nunca es el máximo y la reasignación óptima lo arregla
    m = np.array([[0, 9, 1], [1, 0, 9], [9, 1, 0]])
    r = D.chequeo_permutacion(m, NOMBRES)
    assert r["filas_con_la_diagonal_como_maximo"] == 0 and r["columnas_mayoritarias_distintas"] == 3
    assert r["asignacion_optima"]["exactitud"] == pytest.approx(27 / 30) and r["asignacion_optima"]["imagenes_ganadas"] == 27
    assert r["asignacion_optima"]["permutacion_distinta_de_la_identidad"] == {"a": "b", "b": "c", "c": "a"}


def test_permutacion_en_un_colapso_a_una_clase_no_es_un_corrimiento():
    m = np.array([[1, 0, 9], [0, 1, 9], [0, 0, 10]])        # todo va a la columna c
    r = D.chequeo_permutacion(m, NOMBRES)
    assert r["filas_con_la_diagonal_como_maximo"] == 1 and r["clases_con_la_diagonal_como_maximo"] == ["c"]
    assert r["columnas_mayoritarias_distintas"] == 1 and r["asignacion_optima"]["imagenes_ganadas"] == 0


def test_permutacion_con_modelo_perfecto():
    r = D.chequeo_permutacion(np.diag([5, 5, 5]), NOMBRES)
    assert r["filas_con_la_diagonal_como_maximo"] == 3 and r["asignacion_optima"]["permutacion_distinta_de_la_identidad"] == {}


def _pred(y, p, probs):
    return pd.DataFrame({"etiqueta": y, "pred": p, "prob_pred": probs})


def test_metricas_pd_dev_a_mano():
    pred = _pred([0, 0, 1, 2], [2, 0, 2, 2], [0.9, 0.8, 0.7, 0.6])
    r = D.metricas_pd_dev(pred, NOMBRES, trivial=2)
    assert r["n"] == 4 and r["exactitud"] == 0.5 and r["distribucion_predichas"] == {"a": 1, "b": 0, "c": 3}
    assert r["clase_mas_predicha"] == {"clase": "c", "n": 3, "proporcion": 0.75}
    assert r["confianza_media"]["aciertos"] == pytest.approx(0.7) and r["confianza_media"]["errores"] == pytest.approx(0.8)
    assert r["trivial"]["clase"] == "c" and r["trivial"]["exactitud"] == 0.25 and r["trivial"]["recall_de_esa_clase"] == 1.0
    c = {x["clase"]: x for x in r["por_clase"]}
    assert c["a"]["recall"] == 0.5 and c["a"]["precision"] == 1.0 and c["c"]["precision"] == pytest.approx(1 / 3)


def test_mapeo_efectivo_detecta_una_etiqueta_corrida():
    pv, pdn = ["A___x", "B___y"], ["carpeta_a", "carpeta_b"]
    doc = {"A___x": "carpeta_a", "B___y": "carpeta_b"}
    filas = pd.DataFrame({"ruta": ["raw/plantdoc/train/carpeta_a/1.jpg", "raw/plantdoc/train/carpeta_b/2.jpg"],
                          "clase": ["carpeta_a", "carpeta_b"], "etiqueta": [0, 1]})
    pred = filas.copy()
    assert D.mapeo_efectivo(pv, pdn, doc, filas, pred)["etiqueta_igual_al_indice_de_la_carpeta"]
    mal = filas.assign(etiqueta=[1, 0])                        # índices cruzados
    r = D.mapeo_efectivo(pv, pdn, doc, mal, pred)
    assert not r["etiqueta_igual_al_indice_de_la_carpeta"] and not r["misma_etiqueta_que_las_predicciones"]
    assert not D.mapeo_efectivo(pv, pdn, {"A___x": "otra", "B___y": "carpeta_b"}, filas, pred)["todas_coinciden_con_el_doc"]


def test_buscar_features_usa_el_nombre_y_falla_si_no_esta(tmp_path):
    (tmp_path / "f.pt").write_bytes(b"x")
    assert D.buscar_features(tmp_path, "C:\\otra\\carpeta\\f.pt") == tmp_path / "f.pt"
    with pytest.raises(FileNotFoundError):
        D.buscar_features(tmp_path, "no_esta.pt")


def test_el_script_exige_salida_y_run_dev(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "diagnostico_dev.py"), "--config", str(REPO / "configs" / "03_brecha.yaml")],
                       capture_output=True, text=True, cwd=REPO, env={**os.environ, "DATA_ROOT": str(tmp_path)})
    assert r.returncode == 2 and "--salida" in r.stderr and "--run-dev" in r.stderr
