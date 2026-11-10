"""Sesión 12, fase 4: latencia de la app (percentiles, muestra fija, cronometraje) sin tocar ningún test."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from foliares.data.dataset import filas_particion
from foliares.eval import latencia as L
from foliares.models.mobilenet import MobileNetV3SmallGAP
from foliares.utils.paths import cargar_rutas

REPO = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO / "configs" / "06_app.yaml").read_text(encoding="utf-8"))
CAL = yaml.safe_load((REPO / CFG["calibracion"]).read_text(encoding="utf-8"))
MODELO = yaml.safe_load((REPO / CAL["config_modelo"]).read_text(encoding="utf-8"))


def test_percentiles_y_resumen_en_ms():
    assert L.percentil([1, 2, 3, 4, 5], 50) == 3 and L.percentil(list(range(1, 101)), 95) == pytest.approx(95.05)
    r = L.resumen_ms([0.010, 0.020, 0.030])                     # segundos -> milisegundos
    assert r["n"] == 3 and r["p50_ms"] == pytest.approx(20) and r["min_ms"] == pytest.approx(10) and r["max_ms"] == pytest.approx(30)
    assert r["p95_ms"] == pytest.approx(29)


def test_la_muestra_es_fija_con_la_semilla_sin_repetidos_y_cambia_con_otra():
    filas = pd.DataFrame({"ruta": [f"i{i:03d}.jpg" for i in range(200)]})
    a, b = L.muestra_fija(filas, 50, 42), L.muestra_fija(filas, 50, 42)
    assert a == b and len(set(a)) == 50 and a == sorted(a) and a != L.muestra_fija(filas, 50, 43)
    with pytest.raises(ValueError):
        L.muestra_fija(filas, 201, 42)


def test_la_muestra_del_repo_sale_de_pv_val_y_pd_dev_y_alcanza():
    R = cargar_rutas()
    nombres, nombres_pd = [c["pv"] for c in MODELO["clases"]], [c["plantdoc"] for c in MODELO["clases"]]
    pv = filas_particion(R.splits, "plantvillage", "val", "validacion", nombres)
    pdv = filas_particion(R.splits, "plantdoc", "dev", "evaluacion", nombres, nombres_pd)
    n = CFG["latencia"]["muestra_por_conjunto"]
    assert len(L.muestra_fija(pv, n, CFG["latencia"]["semilla"])) == n and len(L.muestra_fija(pdv, n, CFG["latencia"]["semilla"])) == n


def test_cronometrar_mide_cada_llamada_y_no_cuenta_el_calentamiento(tmp_path):
    rutas = []
    for i in range(3):
        r = tmp_path / f"{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (90, 120, 3), dtype=np.uint8)).save(r)
        rutas.append(r)
    torch.manual_seed(0)
    modelo = MobileNetV3SmallGAP(12, pesos_imagenet=False).eval()
    nombres = [f"c{i}" for i in range(12)]
    for con_cam in (False, True):
        t = L.cronometrar(modelo, rutas, nombres, 14.97, 0.5, "aplastar", 224, con_cam, calentamiento=2, repeticiones=2)
        assert len(t) == 6 and all(x > 0 for x in t)


def test_el_script_y_el_modulo_no_piden_ninguna_particion_test():
    for f in (REPO / "scripts" / "medir_latencia.py", REPO / "src" / "foliares" / "eval" / "latencia.py"):
        fuente = f.read_text(encoding="utf-8")
        for prohibido in ('"test"', "'test'", "PD-test", "PV-test"):
            assert prohibido not in fuente, (f.name, prohibido)


def test_el_script_exige_salida(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "medir_latencia.py"), "--config", str(REPO / "configs" / "06_app.yaml")],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 2 and "--salida" in r.stderr


def test_el_entorno_informa_cpu_hilos_y_torch():
    e = L.entorno_cpu()
    assert e["modelo_cpu"] and e["hilos_torch"] >= 1 and e["torch"] == torch.__version__ and e["dispositivo"] == "cpu"
