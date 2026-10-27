"""Prueba de humo de Colab en dos fases (cada una en un proceso nuevo).

Uso (el notebook `colab_arranque.ipynb` lo orquesta):
  python scripts/colab_humo.py --config configs/colab_humo.yaml --salida <dir> --ckpt-dir <dir> --run-id <id> --fase entrenar
  python scripts/colab_humo.py --config configs/colab_humo.yaml --salida <dir> --ckpt-dir <dir> --run-id <id> --fase reanudar
Escribe `<salida>/<experimento>/<run_id>/fase{1,2}_*.json` y el checkpoint en `--ckpt-dir` (en Colab: Drive).
`--salida` es obligatorio y no tiene un default dentro del repo: en Colab va a Drive (la sesión se corta) y así las salidas
sin seguimiento no cuentan como "cambios sin commitear" en la fase 2.
"""

import os

# Antes de importar torch: cuBLAS lo necesita para los algoritmos deterministas en CUDA (ver docs/reproducibilidad.md).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

from foliares.training import prueba_humo  # noqa: E402
from foliares.utils.paths import cargar_rutas  # noqa: E402
from foliares.utils.sesion import nuevo_run_id  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--fase", required=True, choices=["entrenar", "reanudar"])
    ap.add_argument("--salida", required=True, type=Path, help="carpeta de resultados (en Colab, en Drive; no dentro del repo)")
    ap.add_argument("--ckpt-dir", type=Path, default=None, help="por defecto, el directorio de la corrida")
    ap.add_argument("--run-id", default=None, help="obligatorio en `reanudar`; en `entrenar`, se genera si falta")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    R = cargar_rutas()
    salida = a.salida
    if a.fase == "reanudar" and not a.run_id:
        ap.error("--run-id es obligatorio en la fase reanudar")
    run_id = a.run_id or nuevo_run_id(cfg["experimento"])
    ckpt_dir = a.ckpt_dir or (salida / cfg["experimento"] / run_id)
    f = prueba_humo.fase_entrenar if a.fase == "entrenar" else prueba_humo.fase_reanudar
    datos = f(cfg, R.repo, R.splits, R.data_root, salida, ckpt_dir, run_id)
    resumen = {k: datos.get(k) for k in ("fase", "run_id", "commit", "cambios_sin_commitear", "torch", "dispositivo", "gpu",
                                         "tiempo_s", "memoria_pico_gpu_mb", "memoria_pico_cpu_mb", "reanudacion_ok",
                                         "igual_bit_a_bit", "max_abs_dif_perdidas", "comprobaciones", "sonda_determinismo")
               if k in datos}
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print("JSON:", salida / cfg["experimento"] / run_id)
    return 0 if datos.get("reanudacion_ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
