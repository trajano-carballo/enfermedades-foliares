"""Entrena el modelo base (etapa 2): Linear(576, 12) sobre features GAP precalculadas de PV-train, early stopping por PV-val.

Uso:
  python scripts/entrenar.py --config configs/02_modelo_base.yaml --salida <dir fuera del repo> [--cache <dir>] [--limite N]
Escribe `<salida>/<experimento>/<run_id>/resultados.json` y `mejor_checkpoint.pt` (fuera del repo) y, salvo con `--limite`,
copia los JSON chicos (sin pesos ni features) a `results/<experimento>/<run_id>/` del repo. Solo PV-train y PV-val.
Si PV-train supera el umbral de desbalance de la config (decisión b), se detiene antes de entrenar (código 3); si faltan
las definiciones de métricas o la salida está dentro del repo, código 2. En ambos casos no escribe nada.
"""

import os

# Antes de importar torch: cuBLAS lo necesita para los algoritmos deterministas en CUDA (ver docs/reproducibilidad.md).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

from foliares import train  # noqa: E402
from foliares.utils.paths import cargar_rutas  # noqa: E402
from foliares.utils.sesion import nuevo_run_id  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path, help="carpeta de resultados fuera del repo (en Colab, Drive)")
    ap.add_argument("--cache", type=Path, default=None, help="caché de features (por defecto, <salida>/cache_features)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limite", type=int, default=None, help="prueba corta: solo las primeras N imágenes de cada partición")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    R = cargar_rutas()
    run_id = a.run_id or nuevo_run_id(cfg["experimento"] + ("_prueba" if a.limite else ""))
    try:
        d = train.ejecutar(cfg, R.repo, R.splits, R.data_root, a.salida, a.cache or a.salida / "cache_features", run_id, a.limite)
    except train.DesbalanceExcedido as e:
        print(e, file=sys.stderr)
        return 3
    except ValueError as e:                     # definiciones sin fijar o salida dentro del repo: nada se escribió
        print(e, file=sys.stderr)
        return 2
    resumen = {"run_id": run_id, "commit": d["commit"], "cambios_sin_commitear": d["cambios_sin_commitear"],
               "dispositivo": d["dispositivo"], "epoca_elegida": d["entrenamiento"]["epoca_elegida"],
               "macro_f1_pv_val": d["pv_val"]["macro_f1"], "tiempo_extraccion_s": d["tiempo_extraccion_s"],
               "tiempo_entrenamiento_s": d["tiempo_entrenamiento_s"]}
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print("JSON:", a.salida / cfg["experimento"] / run_id / "resultados.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
