"""Medición informativa en PV-val de la app (etapa 6, sesión 12): fracción con probabilidad máxima calibrada ≥ τ y acierto de
las que superan τ, con el modelo, la T y τ de configs/05_calibracion.yaml, más la referencia sin calibrar (T = 1). No toca ningún test.

Uso:
  python scripts/medir_pv_val.py --config configs/06_app.yaml --salida <dir fuera del repo>
`--salida` es la carpeta de las sesiones 8–11 (el checkpoint se busca relativo a ella). El JSON se copia a results/06_app/<run_id>/.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from foliares.eval.medicion_pv_val import medir_pv_val
from foliares.utils.paths import cargar_rutas
from foliares.utils.sesion import nuevo_run_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    R = cargar_rutas()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    cfg_cal = yaml.safe_load((R.repo / cfg["calibracion"]).read_text(encoding="utf-8"))
    cfg_modelo = yaml.safe_load((R.repo / cfg_cal["config_modelo"]).read_text(encoding="utf-8"))
    run_id = a.run_id or nuevo_run_id(f"{cfg['experimento']}_pv_val")
    d = medir_pv_val(cfg, cfg_cal, cfg_modelo, R.repo, R.splits, R.data_root, a.salida, run_id)
    r4 = lambda v: None if v is None else round(v, 4)  # noqa: E731
    resumen = lambda r: {k: {"estimacion": r4(v["estimacion"]), "ic": [r4(v["ic_bajo"]), r4(v["ic_alto"])]} for k, v in r.items()}  # noqa: E731
    print(json.dumps({"run_id": run_id, "cambios_sin_commitear": d["cambios_sin_commitear"], "n_pv_val": d["n_pv_val"],
                      "T": d["temperatura"], "tau": d["tau"], "resultados": resumen(d["resultados"]),
                      "referencia_sin_calibrar_T1": resumen(d["referencia_sin_calibrar_T1"]),
                      "confianza_maxima_calibrada": d["confianza_maxima_calibrada"]}, indent=2, ensure_ascii=False))
    print("JSON en:", a.salida / cfg["experimento"] / run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
