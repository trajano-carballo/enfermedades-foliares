"""Sesión 10: misma muestra de CAM que la sesión 9, figura base vs. variante y scripts de comparación y figuras."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from foliares.eval import figuras as FIG
from foliares.explain.muestra import seleccion_desde_muestra

REPO = Path(__file__).resolve().parents[1]
SLOT = {"PlantDoc-dev (1.ª)": 0, "PlantDoc-dev (2.ª)": 1, "PlantVillage-val (contraste)": 2}


def test_la_seleccion_reproduce_la_muestra_fija_en_las_mismas_celdas():
    filas = pd.DataFrame({"ruta": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]})
    muestra = [{"ruta": "c.jpg", "origen": "plantdoc-dev", "fila_grilla": "PlantDoc-dev (1.ª)", "tipo": "acierto"},
               {"ruta": "a.jpg", "origen": "plantdoc-dev", "fila_grilla": "PlantDoc-dev (2.ª)", "tipo": "error"},
               {"ruta": "z.jpg", "origen": "plantvillage-val", "fila_grilla": "PlantVillage-val (contraste)", "tipo": None}]
    s = seleccion_desde_muestra(muestra, filas, "plantdoc-dev", SLOT)
    assert s.fila.tolist() == [2, 0] and s.slot.tolist() == [0, 1] and s.tipo.tolist() == ["acierto", "error"]


def test_una_ruta_que_ya_no_esta_hace_fallar_la_seleccion():
    with pytest.raises(ValueError, match="no están en el Dataset"):
        seleccion_desde_muestra([{"ruta": "x.jpg", "origen": "plantdoc-dev", "fila_grilla": "PlantDoc-dev (1.ª)"}],
                                pd.DataFrame({"ruta": ["a.jpg"]}), "plantdoc-dev", SLOT)


def test_la_muestra_de_la_sesion_9_esta_versionada_y_tiene_36_imagenes():
    import json
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "04_evaluacion.yaml").read_text(encoding="utf-8"))
    m = json.loads((REPO / cfg["cam"]["muestra_sesion9"]).read_text(encoding="utf-8"))["muestra"]
    assert len(m) == 36 and {x["origen"] for x in m} == {"plantdoc-dev", "plantvillage-val"} and set(x["fila_grilla"] for x in m) == set(SLOT)


def _tests_json(pv, pd_, brecha):
    ic = lambda e: {"estimacion": e, "ic_bajo": e - 0.02, "ic_alto": e + 0.02}  # noqa: E731
    return {"resultados": {"PV-test": {"base_macro_f1_ic": ic(pv)}, "PD-test": {"base_macro_f1_ic": ic(pd_)}},
            "brecha_macro_f1_pv_test_menos_pd_test": ic(brecha)}


def test_la_figura_base_vs_variante_se_genera(tmp_path):
    f = FIG.figura_comparacion(_tests_json(0.93, 0.13, 0.80), _tests_json(0.95, 0.12, 0.83), tmp_path / "c.png")
    assert f.is_file() and f.stat().st_size > 5000


@pytest.mark.parametrize("script,args", [("comparar_con_base.py", ("--run-tests",)), ("figuras_sesion10.py", ("--run-tests", "--run-ic"))])
def test_los_scripts_exigen_sus_argumentos(tmp_path, script, args):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--config", str(REPO / "configs" / "04_evaluacion.yaml")],
                       capture_output=True, text=True, cwd=REPO, env={**os.environ, "DATA_ROOT": str(tmp_path)})
    assert r.returncode == 2 and "--salida" in r.stderr and all(a in r.stderr for a in args)
