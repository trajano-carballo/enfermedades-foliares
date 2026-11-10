"""Figuras y tablas del documento final (sesión 14) desde los JSON versionados de results/.

Uso:
  python scripts/figuras_finales.py --config configs/07_figuras_finales.yaml [--figuras <dir>]
Solo lee JSON de results/ (números ya medidos): no carga ningún modelo, no lee ninguna imagen y no toca ningún test. Las
grillas de CAM no están acá: necesitan checkpoint e imágenes y muestran fotos de PlantDoc (quedan en docs/figuras_sesion9 y 10).
Escribe en la carpeta de figuras los PNG y `tablas.md` (tablas + fuente, commit y estado del árbol de cada JSON).
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from foliares.eval import figuras as FIG
from foliares.eval import finales as F
from foliares.utils.git import info_repo
from foliares.utils.paths import raiz_repo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--figuras", type=Path, default=None, help="por defecto, `figuras` de la config (relativa al repo)")
    a = ap.parse_args(argv)
    repo = raiz_repo()
    estado = info_repo(repo)            # antes de escribir: los archivos que escribe este script no deben contar como cambios
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    for ruta in cfg["fuentes"].values():
        if not ruta.startswith("results/"):
            raise ValueError(f"Fuente fuera de results/: {ruta}")
    J = {k: json.loads((repo / v).read_text(encoding="utf-8")) for k, v in cfg["fuentes"].items()}
    figs = a.figuras or (repo / cfg["figuras"])
    figs.mkdir(parents=True, exist_ok=True)

    escritas = [
        FIG.figura_brecha(J["tests_base"], figs / "brecha_base_ic.png"),
        FIG.figura_comparacion(J["tests_base"], J["tests_variante"], figs / "brecha_base_vs_variante.png"),
        FIG.figura_matrices(J["tests_base"], figs / "matrices_base.png", ic_json=cfg["fuentes"]["ic_base"]),
        FIG.figura_matrices(J["tests_variante"], figs / "matrices_variante.png", ic_json=cfg["fuentes"]["ic_variante"]),
        FIG.figura_fiabilidad(J["calibracion_dev"], figs / "fiabilidad_pd_dev.png"),
        # PD-test: desde el JSON ya versionado de la pasada única de la fila 5 (no es un acceso nuevo)
        FIG.figura_fiabilidad(J["calibracion_test"], figs / "fiabilidad_pd_test.png", despues="despues_T_final",
                              titulo_despues=f"Después (T = {J['calibracion_test']['temperatura']:.2f}, fija)",
                              conjunto="PlantDoc-test (pasada única, fila 5)", n=J["calibracion_test"]["n_pd_test"]),
        FIG.figura_cobertura(J["calibracion_dev"], figs / "cobertura_pd_dev.png"),
        F.figura_latencia(J["latencia"], figs / "latencia_tamano.png"),
    ]

    fuentes = ["| Clave | JSON | Commit | Árbol limpio |", "|---|---|---|---|"]
    fuentes += [f"| {k} | `{cfg['fuentes'][k]}` | `{d['commit'][:7]}` | {'sí' if not d['cambios_sin_commitear'] else 'NO'} |" for k, d in J.items()]
    md = ["# Tablas del documento final", "",
          f"Generado por `scripts/figuras_finales.py --config {a.config.as_posix()}` desde los JSON de `results/` (sin cómputo nuevo). "
          "IC 95 % bootstrap estratificado, 2.000 réplicas, semilla 42.", "",
          f"Generado desde el commit `{estado['commit'][:7]}`; cambios sin commitear al generar: "
          f"{'no' if estado['cambios_sin_commitear'] is False else 'SÍ'}.", "",
          "## Macro-F1 base vs. variante (filas 1–4)", "", F.tabla_brecha(J["tests_base"], J["tests_variante"], J["comparacion"]), "",
          "## Calibración y abstención: PV-val contra PD-test (fila 5)", "", F.tabla_calibracion(J["pv_val_app"], J["calibracion_test"]), "",
          "## Latencia y tamaño de la app", "", F.tabla_latencia(J["latencia"]), "",
          "## Fuentes", "", *fuentes, "",
          "Las grillas de CAM no salen de results/ (necesitan checkpoint e imágenes) y muestran fotos de PlantDoc: quedan en "
          "`docs/figuras_sesion9/` y `docs/figuras_sesion10/`.", ""]
    escritas.append(figs / "tablas.md")
    (figs / "tablas.md").write_text("\n".join(md), encoding="utf-8")
    for p in escritas:
        print(p.relative_to(repo).as_posix() if repo in p.resolve().parents else p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
