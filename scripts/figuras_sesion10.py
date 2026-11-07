"""Figuras de la sesión 10 (base vs. variante, matrices de la variante, grilla de CAM de la variante) y su JSON.

Uso:
  python scripts/figuras_sesion10.py --config configs/04_evaluacion.yaml --salida <dir fuera del repo> \\
      --run-tests <run_id de la pasada única de la variante> --run-ic <run_id de ic_por_clase de la variante> [--figuras docs/figuras_sesion10]
La grilla de CAM usa la MISMA muestra fija de la sesión 9 (`cam.muestra_sesion9` de la config): las mismas imágenes de
PlantDoc-dev y PlantVillage-val, en las mismas celdas; nunca imágenes de un test. Escribe `cam_muestra.json` en
`<salida>/04_aumentacion/<run_id>/` y lo copia a `results/04_aumentacion/<run_id>/`.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

from foliares.data.dataset import DatasetParticion
from foliares.eval import figuras as FIG
from foliares.eval.evaluacion import cargar_modelo_base, exigir_fuera_del_repo
from foliares.explain.muestra import seleccion_desde_muestra
from foliares.utils.paths import cargar_rutas
from foliares.utils.sesion import Medidor, escribir_json, nuevo_run_id, registro_base

ROTULOS = {0: "PlantDoc-dev (1.ª)", 1: "PlantDoc-dev (2.ª)", 2: "PlantVillage-val (contraste)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path)
    ap.add_argument("--run-tests", required=True)
    ap.add_argument("--run-ic", required=True)
    ap.add_argument("--figuras", type=Path, default=None)
    a = ap.parse_args()
    R = cargar_rutas()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    cm = yaml.safe_load((R.repo / cfg["config_modelo"]).read_text(encoding="utf-8"))
    registro = registro_base(R.repo, R.splits, {"evaluacion": cfg, "modelo": cm}, {})
    salida = exigir_fuera_del_repo(a.salida, R.repo)
    figs = a.figuras or (R.repo / "docs" / "figuras_sesion10")
    figs.mkdir(parents=True, exist_ok=True)
    geo, pv, pdn = cfg["geometria"], [c["pv"] for c in cm["clases"]], [c["plantdoc"] for c in cm["clases"]]
    exp = salida / cfg["experimento"]
    t_var = json.loads((exp / a.run_tests / "tests.json").read_text(encoding="utf-8"))
    t_base = json.loads((salida / "03_brecha" / cfg["referencia_base"]["run_tests"] / "tests.json").read_text(encoding="utf-8"))
    ic_json = f"results/{cfg['experimento']}/{a.run_ic}/ic_por_clase.json"
    if not (exp / a.run_ic / "ic_por_clase.json").is_file():
        raise FileNotFoundError(f"No existe el JSON de IC por clase del run {a.run_ic}")
    muestra9 = json.loads((R.repo / cfg["cam"]["muestra_sesion9"]).read_text(encoding="utf-8"))["muestra"]
    slot = {v: k for k, v in ROTULOS.items()}
    with Medidor() as med:
        escritas = [FIG.figura_comparacion(t_base, t_var, figs / "base_vs_variante.png"),
                    FIG.figura_matrices(t_var, figs / "matrices_variante_pv_pd_test.png", ic_json=ic_json)]
        ds_pd = DatasetParticion(R.splits, R.data_root, "plantdoc", "dev", "evaluacion", pv, pdn, geometria=geo, tamano=cfg["tamano"])
        ds_pv = DatasetParticion(R.splits, R.data_root, "plantvillage", "val", "validacion", pv, geometria=geo, tamano=cfg["tamano"])
        modelo = cargar_modelo_base(salida, cfg, len(pv))
        sel_pd = seleccion_desde_muestra(muestra9, ds_pd.filas, "plantdoc-dev", slot)
        sel_pv = seleccion_desde_muestra(muestra9, ds_pv.filas, "plantvillage-val", slot)
        celdas = (FIG.construir_celdas(modelo, ds_pd, sel_pd, "plantdoc-dev", ROTULOS)
                  + FIG.construir_celdas(modelo, ds_pv, sel_pv, "plantvillage-val", ROTULOS))
        for nombre, rango in (("cam_variante_clases_1a6.png", range(0, 6)), ("cam_variante_clases_7a12.png", range(6, 12))):
            escritas.append(FIG.grilla_cam([c for c in celdas if c["columna"] in rango], pv, figs / nombre,
                                           f"Variante de la etapa 4, clases {rango.start + 1} a {rango.stop}"))
    muestra = [{"ruta": c["ruta"], "origen": c["origen"], "fila_grilla": c["fila_grilla"][1], "tipo_en_la_sesion_9": c["tipo"],
                "clase_real": pv[c["real"]], "clase_predicha": pv[c["pred"]], "probabilidad_predicha": c["prob"]} for c in celdas]
    misma = sorted((m["ruta"], m["fila_grilla"]) for m in muestra) == sorted((m["ruta"], m["fila_grilla"]) for m in muestra9)
    if not misma:
        raise ValueError("La muestra no es la misma de la sesión 9")
    run_id = nuevo_run_id(f"{cfg['experimento']}_cam")
    aciertos = {o: sum(m["clase_real"] == m["clase_predicha"] for m in muestra if m["origen"] == o) for o in ("plantdoc-dev", "plantvillage-val")}
    datos = {**registro, "fase": "cam_variante", "run_id": run_id, "run_tests": a.run_tests, "run_ic": a.run_ic, "geometria": geo,
             "modelo_evaluado": cfg["modelo_base"], "muestra_de": cfg["cam"]["muestra_sesion9"], "misma_muestra_que_la_sesion_9": misma,
             "datos_mostrados": "PlantDoc-dev y PlantVillage-val; ningún test", "n_imagenes": len(muestra), "aciertos": aciertos,
             "muestra": muestra, "figuras": [p.relative_to(R.repo).as_posix() if R.repo in p.resolve().parents else str(p) for p in escritas],
             **med.resumen()}
    ruta = escribir_json(exp / run_id / "cam_muestra.json", datos)
    destino = R.repo / "results" / cfg["experimento"] / run_id
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta, destino / "cam_muestra.json")
    print(json.dumps({k: datos[k] for k in ("run_id", "cambios_sin_commitear", "commit", "n_imagenes", "aciertos", "figuras")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
