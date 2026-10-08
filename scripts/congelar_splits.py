"""Congela los candidatos de particiones en data/splits/ (SPLITS_DIR) con sha256 y README de versión.

NO se ejecuta en las sesiones de trabajo con Claude Code: lo corre el equipo al cerrar la revisión humana.
Uso:  python scripts/congelar_splits.py --config configs/congelar_splits.yaml [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import yaml

from foliares.data.congelar import congelar
from foliares.utils.paths import cargar_rutas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true", help="valida y calcula sha256, no escribe nada")
    ap.add_argument("--permitir-provisional", action="store_true", help="congela aunque la revisión humana esté PROVISIONAL")
    ap.add_argument("--force", action="store_true", help="sobrescribe particiones ya congeladas (requiere decisión del equipo)")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    R = cargar_rutas()
    try:
        r = congelar(cfg, R.repo, R.interim, R.splits, a.force, a.permitir_provisional, a.dry_run)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    print("[dry-run]" if r["dry_run"] else f"Congelado en {R.splits}")
    print(r["resumen"])
    for n, h in r["sha256"].items():
        print(h, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
