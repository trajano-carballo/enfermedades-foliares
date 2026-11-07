"""Entrena la variante de la etapa 4: fine-tuning parcial con aumentación dirigida (sesión 10).

Uso (en Colab lo orquesta notebooks/04_aumentacion_colab.ipynb):
  python scripts/entrenar_finetune.py --config configs/04_aumentacion.yaml --salida <results fuera del repo> \
      --ckpt-dir <checkpoints fuera del repo> --checkpoint-base <mejor_checkpoint.pt del modelo base> [--run-id ID] [--limite N]
Checkpoint por época en <ckpt-dir>/04_aumentacion/<run_id>/ (ultimo.pt y mejor.pt). Si ya existe ultimo.pt con el mismo
--run-id, retoma desde la última época completa. Escribe <salida>/04_aumentacion/<run_id>/resultados.json. Solo PV-train y
PV-val. --limite: prueba corta con pocas imágenes por clase (el JSON lo marca como prueba_corta).
"""

import os

# Antes de importar torch: cuBLAS lo necesita para los algoritmos deterministas en CUDA (ver docs/reproducibilidad.md).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

from foliares.eval.evaluacion import exigir_fuera_del_repo  # noqa: E402
from foliares.training import finetune  # noqa: E402
from foliares.utils.paths import cargar_rutas  # noqa: E402
from foliares.utils.sesion import nuevo_run_id  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path, help="carpeta de resultados fuera del repo (en Colab, Drive)")
    ap.add_argument("--ckpt-dir", required=True, type=Path, help="carpeta de checkpoints fuera del repo (en Colab, Drive)")
    ap.add_argument("--checkpoint-base", required=True, type=Path, help="mejor_checkpoint.pt del modelo base (sesión 8)")
    ap.add_argument("--run-id", default=None, help="para retomar una corrida cortada, el mismo run_id")
    ap.add_argument("--limite", type=int, default=None)
    a = ap.parse_args()
    R = cargar_rutas()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    cfg_modelo = yaml.safe_load((R.repo / cfg["config_modelo"]).read_text(encoding="utf-8"))
    salida, ckpt = exigir_fuera_del_repo(a.salida, R.repo), exigir_fuera_del_repo(a.ckpt_dir, R.repo)
    run_id = a.run_id or nuevo_run_id(cfg["experimento"] + ("_prueba" if a.limite else ""))
    d = finetune.entrenar(cfg, cfg_modelo, R.repo, R.splits, R.data_root, salida, ckpt, a.checkpoint_base, run_id, a.limite)
    print(json.dumps({k: d[k] for k in ("run_id", "commit", "cambios_sin_commitear", "dispositivo", "gpu", "torch", "cuda_runtime",
                                        "epoca_elegida", "epocas_corridas", "detenido_por", "reanudaciones_desde_epoca",
                                        "tiempo_s", "memoria_pico_gpu_mb", "memoria_pico_cpu_mb")}
                     | {"macro_f1_pv_val": d["pv_val"]["macro_f1"]}, indent=2, ensure_ascii=False))
    print("JSON:", salida / cfg["experimento"] / run_id / "resultados.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
