"""Verifica un subconjunto ya descomprimido en DATA_ROOT contra sus MANIFIESTO_*.csv (existencia, tamaño, sha256).

Uso (Colab, tras descomprimir los .tar en /content/data):
  DATA_ROOT=/content/data python scripts/verificar_paquete.py
Opcional: --tar archivo.tar [...] verifica antes el sha256 de cada .tar contra su .sha256 (junto al .tar);
--rapido compara solo existencia y tamaño. Código de salida 0 = todo OK.
"""

import argparse
import sys
from pathlib import Path

from foliares.data import paquete
from foliares.utils.paths import cargar_rutas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=None, help="por defecto, DATA_ROOT de configs/paths.yaml / entorno")
    ap.add_argument("--tar", type=Path, nargs="*", default=[])
    ap.add_argument("--rapido", action="store_true")
    a = ap.parse_args()
    root = a.data_root or cargar_rutas().data_root
    ok = True
    for t in a.tar:
        bien = paquete.verificar_tar(t)
        print(f"{'OK ' if bien else 'MAL'} sha256 del tar: {t.name}")
        ok &= bien
    manifiestos = sorted(root.glob("MANIFIESTO_*.csv"))
    if not manifiestos:
        print(f"No hay MANIFIESTO_*.csv en {root}: ¿se descomprimieron los .tar ahí?", file=sys.stderr)
        return 1
    for m in manifiestos:
        r = paquete.verificar(root, m, a.rapido)
        print(f"{'OK ' if r['ok'] else 'MAL'} {r['manifiesto']}: {r['esperados']} esperados, faltan {len(r['faltan'])}, "
              f"tamaño distinto {len(r['tamano_distinto'])}, sha256 distinto {len(r['sha256_distinto'])}")
        for k in ("faltan", "tamano_distinto", "sha256_distinto"):
            for x in r[k][:5]:
                print(f"   {k}: {x}")
        ok &= r["ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
