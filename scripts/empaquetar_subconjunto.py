"""Empaqueta el subconjunto de 12 clases en un .tar por dataset (+ MANIFIESTO con sha256 por archivo).

NO se ejecuta en las sesiones de trabajo con Claude Code (poco disco): se corre en la PC con disco.
Requiere haber generado antes (notebooks 01 y 01b) en DATA_ROOT/interim:
  manifiesto_plantvillage.csv  y  manifiesto_plantdoc_depurado.csv
Uso:  python scripts/empaquetar_subconjunto.py --config configs/paquete_subconjunto.yaml --salida D:/paquetes
Salida: plantvillage_subconjunto.tar, plantdoc_subconjunto.tar y sus .sha256 (subirlos a Drive).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from foliares.data import paquete
from foliares.utils.paths import cargar_rutas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path, help="carpeta donde se escriben los .tar")
    ap.add_argument("--dataset", choices=["plantvillage", "plantdoc", "todos"], default="todos")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    R = cargar_rutas()
    trabajos = {
        "plantvillage": lambda: paquete.lista_plantvillage(pd.read_csv(R.interim / cfg["manifiesto_plantvillage"])),
        "plantdoc": lambda: paquete.lista_plantdoc(pd.read_csv(R.interim / cfg["manifiesto_plantdoc_depurado"])),
    }
    for ds, lista in trabajos.items():
        if a.dataset not in (ds, "todos"):
            continue
        rutas = lista()
        print(f"[{ds}] {len(rutas)} archivos; calculando sha256 y empaquetando…")
        r = paquete.empaquetar(rutas, R.data_root, a.salida / f"{ds}_subconjunto.tar", ds)
        print(f"[{ds}] {r['archivos']} archivos, {r['bytes'] / 2**20:.0f} MB, tar sha256 {r['tar_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
