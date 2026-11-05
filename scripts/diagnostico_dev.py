"""Diagnóstico de bugs sobre PlantDoc-dev (sesión 9). Solo PD-dev (predicciones guardadas, `aplastar`) y 20 imágenes de PV-val.

Uso:
  python scripts/diagnostico_dev.py --config configs/03_brecha.yaml --salida <dir fuera del repo> --run-dev <run_id de la fase dev>
Lee `<salida>/03_brecha/<run_dev>/predicciones_pd_dev_<geometría>.csv` (las que escribió `scripts/evaluar.py --fase dev`) y
escribe `diagnostico_dev.json` ahí mismo; además copia ese JSON a `results/03_brecha/<run_dev>/` del repo. No toca ningún test.
Chequea: mapeo efectivo de clases, recall/precisión por clase, matriz de cuentas, distribución de predichas, trivial, confianza,
permutación y control de preproceso (20 imágenes de PV-val por el mismo camino que PD-dev).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

from foliares.data.congelar import sha256_archivo
from foliares.data.dataset import DatasetParticion, filas_particion
from foliares.eval import diagnostico as D
from foliares.eval import metricas as M
from foliares.eval.evaluacion import cargar_modelo_base, exigir_fuera_del_repo
from foliares.utils.paths import cargar_rutas
from foliares.utils.sesion import Medidor, escribir_json, registro_base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path, help="carpeta fuera del repo (la de las sesiones 8 y 9)")
    ap.add_argument("--run-dev", required=True, help="run_id de la fase dev (carpeta dentro de <salida>/03_brecha/)")
    a = ap.parse_args()
    R = cargar_rutas()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    cm = yaml.safe_load((R.repo / cfg["config_modelo"]).read_text(encoding="utf-8"))
    registro = registro_base(R.repo, R.splits, {"evaluacion": cfg, "modelo": cm}, {})    # estado de git antes de escribir
    salida = exigir_fuera_del_repo(a.salida, R.repo)
    geo = cfg["geometria"]
    pv, pdn = [c["pv"] for c in cm["clases"]], [c["plantdoc"] for c in cm["clases"]]
    carpeta = salida / cfg["experimento"] / a.run_dev
    pred = pd.read_csv(carpeta / f"predicciones_pd_dev_{geo}.csv")
    ds_pd = DatasetParticion(R.splits, R.data_root, "plantdoc", "dev", "evaluacion", pv, pdn, geometria=geo, tamano=cfg["tamano"])
    ds_pv = DatasetParticion(R.splits, R.data_root, "plantvillage", "val", "validacion", pv, geometria=geo, tamano=cfg["tamano"])
    doc = D.pares_del_doc((R.repo / "docs" / "mapeo_clases.md").read_text(encoding="utf-8"))
    y_train = filas_particion(R.splits, "plantvillage", "train", "entrenamiento", pv).etiqueta.to_numpy()
    trivial = M.clase_mayoritaria(y_train, len(pv), cm["metricas"]["desempate_trivial"])
    run_base = cfg["modelo_base"]["run_id"]
    res_base = json.loads((R.repo / "results" / "02_modelo_base" / run_base / "resultados.json").read_text(encoding="utf-8"))
    ft = D.buscar_features(salida / "cache_features", res_base["features"]["val"]["archivo"])
    if sha256_archivo(ft) != res_base["features"]["val"]["meta"]["sha256"]:
        raise ValueError(f"El caché de features {ft} no tiene el sha256 que registra el run {run_base}")
    feats = torch.load(ft, map_location="cpu", weights_only=True)
    if not torch.equal(feats["y"], ds_pv.etiquetas):
        raise ValueError("Las etiquetas del caché de entrenamiento no coinciden con las de PV-val del Dataset")
    modelo = cargar_modelo_base(salida, cfg, len(pv))
    with Medidor() as med:
        mapeo = D.mapeo_efectivo(pv, pdn, doc, ds_pd.filas, pred)
        metr = D.metricas_pd_dev(pred, pv, trivial)
        perm = D.chequeo_permutacion(metr["matriz_cuentas"]["filas_verdadera_columnas_predicha"], pv)
        ctrl = D.control_preproceso(modelo, ds_pv, feats["X"], ds_pd, pred)
    datos = {**registro, "fase": "diagnostico_dev", "run_dev": a.run_dev, "geometria": geo, "modelo_base": cfg["modelo_base"],
             "datos_usados": "PlantDoc-dev (predicciones guardadas y 20 recalculadas) y 20 imágenes de PV-val; ningún test",
             "mapeo": mapeo, "pd_dev": metr, "permutacion": perm, "control_preproceso": ctrl, **med.resumen(),
             "cache_features_entrenamiento": ft.name, "rutas_relativas_a": "la carpeta --salida",
             "nota_memoria": "memoria_pico_cpu_mb es el máximo del proceso entero"}
    ruta = escribir_json(carpeta / "diagnostico_dev.json", datos)
    destino = R.repo / "results" / cfg["experimento"] / a.run_dev
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta, destino / "diagnostico_dev.json")
    print(json.dumps({"run_dev": a.run_dev, "cambios_sin_commitear": datos["cambios_sin_commitear"], "commit": datos["commit"],
                      "mapeo_coincide": mapeo["todas_coinciden_con_el_doc"],
                      "diagonal_maxima": f"{perm['filas_con_la_diagonal_como_maximo']}/{perm['de']}",
                      "clase_mas_predicha": metr["clase_mas_predicha"],
                      "max_abs_dif_logits_control": ctrl["max_abs_dif_logits"]}, indent=2, ensure_ascii=False))
    print("JSON:", ruta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
