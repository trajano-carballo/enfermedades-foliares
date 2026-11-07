"""Calibración por temperatura y abstención (etapa 5, sesión 11). Protocolo: docs/protocolo_evaluacion.md §g.

Uso:
  python scripts/calibrar.py --config configs/05_calibracion.yaml --salida <dir fuera del repo> --fase dev
      Regla 1 (modelo), T por grilla en PlantDoc-dev, ECE y métricas del sistema fuera de partición, con IC. No toca ningún test.
  python scripts/calibrar.py --config configs/05_calibracion.yaml --salida <dir> --fase test --fila-accesos N
      Una pasada por PD-test con modelo, T y τ de la config. Exige la fila N de docs/bitacora/accesos_test.md commiteada y
      coherente (guarda); si no, no evalúa (código 4). Si el óptimo de T cae en un borde de la grilla, sale con código 5.
`--salida` es la carpeta de las sesiones 8–10 (el checkpoint se busca relativo a ella). El JSON se copia a
results/05_calibracion/<run_id>/.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from foliares.eval import calibracion as C
from foliares.eval.guarda_accesos import AccesoNoRegistrado
from foliares.utils.paths import cargar_rutas
from foliares.utils.sesion import nuevo_run_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path)
    ap.add_argument("--fase", required=True, choices=["dev", "test"])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--fila-accesos", type=int, default=None, help="fila de accesos_test.md para PD-test (fase test)")
    a = ap.parse_args()
    if a.fase == "test" and a.fila_accesos is None:
        ap.error("la fase test exige --fila-accesos (fila commiteada de docs/bitacora/accesos_test.md)")
    R = cargar_rutas()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    cfg_modelo = yaml.safe_load((R.repo / cfg["config_modelo"]).read_text(encoding="utf-8"))
    run_id = a.run_id or nuevo_run_id(f"{cfg['experimento']}_{a.fase}")
    try:
        if a.fase == "dev":
            d = C.calibrar_dev(cfg, cfg_modelo, R.repo, R.splits, R.data_root, a.salida, run_id)
            despues = d["despues_fuera_de_particion"]
        else:
            config_rel = a.config.resolve().relative_to(R.repo.resolve()).as_posix()
            d = C.calibrar_test(cfg, cfg_modelo, config_rel, R.repo, R.splits, R.data_root, a.salida, run_id, a.fila_accesos)
            despues = d["despues_T_final"]
    except AccesoNoRegistrado as e:
        print(f"No se evalúa: {e}", file=sys.stderr)
        return 4
    except C.TemperaturaEnBorde as e:
        print(f"Se frena: {e}", file=sys.stderr)
        return 5
    resumen = {k: round(v["estimacion"], 4) for k, v in despues.items()}
    print(json.dumps({"run_id": run_id, "cambios_sin_commitear": d["cambios_sin_commitear"],
                      "T": d.get("temperatura_final", d.get("temperatura")), "antes_T_1": {k: round(v["estimacion"], 4) for k, v in d["antes_T_1"].items()},
                      "despues": resumen}, indent=2, ensure_ascii=False))
    print("JSON en:", a.salida / cfg["experimento"] / run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
