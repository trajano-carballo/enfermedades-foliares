"""Figuras de la sesión 11 (diagrama de fiabilidad y curva de cobertura en PlantDoc-dev) desde el JSON versionado de la fase dev.

Uso:
  python scripts/figuras_sesion11.py --config configs/05_calibracion.yaml --run-dev <run_id> [--figuras docs/figuras_sesion11]
Solo lee `results/05_calibracion/<run_dev>/calibracion_dev.json`: ninguna imagen y ningún test.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from foliares.eval import figuras as FIG
from foliares.utils.paths import cargar_rutas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-dev", required=True)
    ap.add_argument("--figuras", type=Path, default=None)
    a = ap.parse_args()
    R = cargar_rutas()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    dev = json.loads((R.repo / "results" / cfg["experimento"] / a.run_dev / "calibracion_dev.json").read_text(encoding="utf-8"))
    figs = a.figuras or (R.repo / "docs" / "figuras_sesion11")
    figs.mkdir(parents=True, exist_ok=True)
    print(FIG.figura_fiabilidad(dev, figs / "fiabilidad_pd_dev.png"))
    print(FIG.figura_cobertura(dev, figs / "cobertura_pd_dev.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
