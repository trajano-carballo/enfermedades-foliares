"""Evaluación del modelo base (etapa 3, sesión 9). Protocolo: docs/protocolo_evaluacion.md.

Uso:
  python scripts/evaluar.py --config configs/03_brecha.yaml --salida <dir fuera del repo> --fase dev
      PlantDoc-dev con cada geometría candidata y la regla (a). No toca ningún test.
  python scripts/evaluar.py --config configs/04_evaluacion.yaml --salida <dir> --fase dev_informativo
      Etapa 4: la variante en PlantDoc-dev con la geometría fija, macro-F1 con IC y diferencia pareada contra el base.
  python scripts/evaluar.py --config configs/03_brecha.yaml --salida <dir> --fase tests --fila-accesos-pv N --fila-accesos-pd M
      Una pasada por PV-test y PD-test con la geometría de la config. Exige las filas N (PV-test) y M (PD-test) de
      docs/bitacora/accesos_test.md commiteadas y coherentes (guarda, decisión e); si no, no evalúa (código 4).
`--salida` es la misma carpeta de la sesión 8 (el checkpoint del base se busca relativo a ella). Los JSON chicos se copian a
results/<experimento>/<run_id>/ del repo.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from foliares.eval import evaluacion
from foliares.eval.guarda_accesos import AccesoNoRegistrado
from foliares.utils.paths import cargar_rutas
from foliares.utils.sesion import nuevo_run_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path, help="carpeta fuera del repo (la de la sesión 8)")
    ap.add_argument("--fase", required=True, choices=["dev", "dev_informativo", "tests"])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--fila-accesos-pv", type=int, default=None, help="fila de accesos_test.md para PV-test (fase tests)")
    ap.add_argument("--fila-accesos-pd", type=int, default=None, help="fila de accesos_test.md para PD-test (fase tests)")
    a = ap.parse_args()
    if a.fase == "tests" and (a.fila_accesos_pv is None or a.fila_accesos_pd is None):
        ap.error("la fase tests exige --fila-accesos-pv y --fila-accesos-pd (filas commiteadas de docs/bitacora/accesos_test.md)")
    R = cargar_rutas()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    cfg_modelo = yaml.safe_load((R.repo / cfg["config_modelo"]).read_text(encoding="utf-8"))
    run_id = a.run_id or nuevo_run_id(f"{cfg['experimento']}_{a.fase}")
    try:
        if a.fase == "dev_informativo":
            d = evaluacion.evaluar_dev_informativo(cfg, cfg_modelo, R.repo, R.splits, R.data_root, a.salida, run_id)
            c = d["comparacion_con_base"]
            print(json.dumps({"run_id": run_id, "cambios_sin_commitear": d["cambios_sin_commitear"], "variante": c["variante"],
                              "base": c["base"], "diferencia_variante_menos_base": c["diferencia_variante_menos_base"]}, indent=2))
        elif a.fase == "dev":
            d = evaluacion.evaluar_dev(cfg, cfg_modelo, R.repo, R.splits, R.data_root, a.salida, run_id)
            resumen = {g: round(r["estimacion"], 4) for g, r in d["regla_a"]["macro_f1_por_geometria"].items()}
            print(json.dumps({"run_id": run_id, "macro_f1_pd_dev": resumen, "califican": d["regla_a"]["califican"],
                              "elegida": d["regla_a"]["geometria_elegida"]}, indent=2, ensure_ascii=False))
        else:
            config_rel = a.config.resolve().relative_to(R.repo.resolve()).as_posix()
            d = evaluacion.evaluar_tests(cfg, cfg_modelo, config_rel, R.repo, R.splits, R.data_root, a.salida, run_id,
                                         {"PV-test": a.fila_accesos_pv, "PD-test": a.fila_accesos_pd})
            print(json.dumps({"run_id": run_id, "brecha": d["brecha_macro_f1_pv_test_menos_pd_test"]}, indent=2))
    except AccesoNoRegistrado as e:
        print(f"No se evalúa: {e}", file=sys.stderr)
        return 4
    print("JSON en:", a.salida / cfg["experimento"] / run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
