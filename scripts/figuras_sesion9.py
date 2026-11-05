"""Figuras de la sesión 9 (brecha con IC, las dos matrices, grilla de CAM) y el JSON de la muestra fija de CAM.

Uso:
  python scripts/figuras_sesion9.py --config configs/03_brecha.yaml --salida <dir fuera del repo> \\
      --run-dev <run_id fase dev> --run-tests <run_id fase tests> [--figuras docs/figuras_sesion9]
Brecha y matrices salen de `tests.json` (solo números). La grilla de CAM usa la muestra fija del protocolo (decisión c):
PlantDoc-dev, 2 por clase (1 acierto y 1 error cuando haya), semilla 42 sobre las predicciones guardadas de la geometría elegida,
más 1 por clase de PV-val. NUNCA imágenes de un test (se verifica antes de dibujar). Escribe `cam_muestra.json` en
`<salida>/03_brecha/<run_id>/` y lo copia a `results/03_brecha/<run_id>/`. Sin backward: CAM sale del mismo forward.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

from foliares.data.dataset import DatasetParticion
from foliares.eval import figuras as FIG
from foliares.eval.evaluacion import cargar_modelo_base, exigir_fuera_del_repo
from foliares.explain.muestra import elegir_pv_val, seleccionar_pd_dev
from foliares.utils.paths import cargar_rutas
from foliares.utils.sesion import Medidor, escribir_json, nuevo_run_id, registro_base

ROTULOS = {0: "PlantDoc-dev (1.ª)", 1: "PlantDoc-dev (2.ª)", 2: "PlantVillage-val (contraste)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path, help="carpeta fuera del repo")
    ap.add_argument("--run-dev", required=True)
    ap.add_argument("--run-tests", required=True)
    ap.add_argument("--figuras", type=Path, default=None, help="por defecto, docs/figuras_sesion9 del repo")
    a = ap.parse_args()
    R = cargar_rutas()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    cm = yaml.safe_load((R.repo / cfg["config_modelo"]).read_text(encoding="utf-8"))
    registro = registro_base(R.repo, R.splits, {"evaluacion": cfg, "modelo": cm}, {"cam": cfg["cam"]["semilla"]})
    salida = exigir_fuera_del_repo(a.salida, R.repo)
    figs = a.figuras or (R.repo / "docs" / "figuras_sesion9")
    figs.mkdir(parents=True, exist_ok=True)
    geo, pv, pdn = cfg["geometria"], [c["pv"] for c in cm["clases"]], [c["plantdoc"] for c in cm["clases"]]
    k = len(pv)
    tests = json.loads((salida / cfg["experimento"] / a.run_tests / "tests.json").read_text(encoding="utf-8"))
    if tests["geometria"] != geo:
        raise ValueError(f"tests.json se evaluó con {tests['geometria']!r} y la config dice {geo!r}")
    with Medidor() as med:
        escritas = [FIG.figura_brecha(tests, figs / "brecha_ic.png"), FIG.figura_matrices(tests, figs / "matrices_pv_pd_test.png")]
        pred = pd.read_csv(salida / cfg["experimento"] / a.run_dev / f"predicciones_pd_dev_{geo}.csv")
        ds_pd = DatasetParticion(R.splits, R.data_root, "plantdoc", "dev", "evaluacion", pv, pdn, geometria=geo, tamano=cfg["tamano"])
        ds_pv = DatasetParticion(R.splits, R.data_root, "plantvillage", "val", "validacion", pv, geometria=geo, tamano=cfg["tamano"])
        if list(ds_pd.filas.ruta) != list(pred.ruta):
            raise ValueError("Las predicciones guardadas no están en el orden del Dataset de PD-dev")
        sel_pd = seleccionar_pd_dev(pred, k, cfg["cam"]["semilla"], cfg["cam"]["por_clase_pd_dev"])
        sel_pd["slot"] = sel_pd.groupby("etiqueta").cumcount()
        sel_pv = elegir_pv_val(ds_pv.filas, k, cfg["cam"]["semilla"]).assign(slot=2)
        modelo = cargar_modelo_base(salida, cfg, k)
        celdas = (FIG.construir_celdas(modelo, ds_pd, sel_pd, "plantdoc-dev", ROTULOS)
                  + FIG.construir_celdas(modelo, ds_pv, sel_pv, "plantvillage-val", ROTULOS))
        for nombre, rango in (("cam_grilla_clases_1a6.png", range(0, 6)), ("cam_grilla_clases_7a12.png", range(6, 12))):
            escritas.append(FIG.grilla_cam([c for c in celdas if c["columna"] in rango], pv, figs / nombre,
                                           f"Clases {rango.start + 1} a {rango.stop}"))
    run_id = nuevo_run_id(f"{cfg['experimento']}_cam")
    muestra = [{"ruta": c["ruta"], "origen": c["origen"], "fila_grilla": c["fila_grilla"][1], "tipo": c["tipo"],
                "clase_real": pv[c["real"]], "clase_predicha": pv[c["pred"]], "probabilidad_predicha": c["prob"],
                "en_carpeta_test_oficial_de_plantdoc": "/plantdoc/test/" in c["ruta"]} for c in celdas]
    datos = {**registro, "fase": "cam", "run_id": run_id, "run_dev": a.run_dev, "run_tests": a.run_tests, "geometria": geo,
             "datos_mostrados": "PlantDoc-dev y PlantVillage-val; ningún test", "n_imagenes": len(muestra),
             "n_plantdoc_dev_de_la_carpeta_test_oficial": sum(m["en_carpeta_test_oficial_de_plantdoc"] for m in muestra),
             "muestra": muestra, "figuras": [str(p.relative_to(R.repo)) if R.repo in p.resolve().parents else str(p) for p in escritas],
             **med.resumen(), "nota_memoria": "memoria_pico_cpu_mb es el máximo del proceso entero"}
    ruta = escribir_json(salida / cfg["experimento"] / run_id / "cam_muestra.json", datos)
    destino = R.repo / "results" / cfg["experimento"] / run_id
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta, destino / "cam_muestra.json")
    print(json.dumps({"run_id": run_id, "cambios_sin_commitear": datos["cambios_sin_commitear"], "commit": datos["commit"],
                      "imagenes": len(muestra), "de_la_carpeta_test_oficial": datos["n_plantdoc_dev_de_la_carpeta_test_oficial"],
                      "figuras": datos["figuras"]}, indent=2, ensure_ascii=False))
    print("JSON:", ruta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
