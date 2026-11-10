"""Verificación de las imágenes externas (Wikimedia Commons) de la app (etapa 6, sesión 12, fase 3): completa sha256 y
resultado del chequeo de duplicados (phash <= umbral, 8 variantes) en docs/imagenes_externas.csv, comparando contra todo
PlantVillage (color y segmented) y todo PlantDoc (train y test oficiales). Es solo hash: no abre ni muestra imágenes de los
conjuntos ni calcula ninguna métrica. Escribe results/06_app/<run_id>/verificacion_externas.json.

Uso:
  python scripts/verificar_externas.py --config configs/06_app.yaml [--cache <carpeta fuera del repo>]
Con --cache, los hashes de los conjuntos se guardan ahí y la corrida siguiente no los recalcula (mismo resultado).
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from foliares.eval.evaluacion import exigir_fuera_del_repo
from foliares.data.externas import escribir_planilla, leer_planilla, verificar_externas
from foliares.utils.paths import cargar_rutas
from foliares.utils.sesion import Medidor, escribir_json, nuevo_run_id, registro_base

EXT = {".jpg", ".jpeg", ".png"}


def _imagenes(base: Path, *subcarpetas: str) -> list[Path]:
    return sorted(p for s in subcarpetas for p in (base / s).rglob("*") if p.is_file() and p.suffix.lower() in EXT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--cache", type=Path, default=None, help="carpeta fuera del repo para el caché de hashes de los conjuntos")
    a = ap.parse_args()
    R = cargar_rutas()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    e = cfg["externas"]
    registro = registro_base(R.repo, R.splits, {"app": cfg}, {})
    run_id = a.run_id or nuevo_run_id(f"{cfg['experimento']}_externas")
    planilla = R.repo / e["planilla"]
    refs = {"PlantVillage (color y segmented)": _imagenes(R.plantvillage, "raw/color", "raw/segmented"),
            "PlantDoc (train y test oficiales)": _imagenes(R.plantdoc, "train", "test")}
    cache = exigir_fuera_del_repo(a.cache, R.repo) if a.cache else None
    detalle = {}
    with Medidor() as med:
        filas = verificar_externas(R.data_root / e["carpeta"], leer_planilla(planilla), refs, e["umbral_phash"], R.data_root, cache, detalle)
    escribir_planilla(planilla, filas)
    datos = {**registro, "fase": "verificacion_externas", "run_id": run_id, "umbral_phash": e["umbral_phash"],
             "comparadas": {n: len(r) for n, r in refs.items()}, "imagenes": filas, "detalle_por_imagen": detalle["imagenes"],
             "cache_hashes": detalle["cache"], **med.resumen(),
             "nota": "solo hash: no se abrió ni mostró ninguna imagen de los conjuntos y no se calculó ninguna métrica",
             "nota_memoria": "memoria_pico_cpu_mb es el máximo del proceso entero"}
    destino = R.repo / "results" / cfg["experimento"] / run_id
    escribir_json(destino / "verificacion_externas.json", datos)
    print(json.dumps({"run_id": run_id, "cambios_sin_commitear": datos["cambios_sin_commitear"], "comparadas": datos["comparadas"],
                      "resultado": {f["archivo"]: f["resultado_duplicados"] for f in filas}}, indent=2, ensure_ascii=False))
    return 1 if any(f["resultado_duplicados"].startswith("DUPLICADA") for f in filas) else 0


if __name__ == "__main__":
    sys.exit(main())
