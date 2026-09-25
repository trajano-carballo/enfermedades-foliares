"""Rescata del árbol de git los archivos de PlantDoc que Windows/NTFS no puede
materializar con su nombre original: nombres con '?' (no válido en NTFS) o
rutas que superan el límite de longitud de path de Windows.

No altera el árbol oficial ni pierde imágenes: extrae el blob con
`git cat-file` y lo escribe con un nombre local saneado. Guarda el mapeo
nombre original -> nombre local en docs/bitacora/plantdoc_archivos_renombrados.csv
para trazabilidad (ver docs/mapeo_clases.md).

Uso: python scripts/rescatar_archivos_plantdoc.py --rev <commit> --rutas-faltantes <txt>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path

MAX_PATH_COMPONENTE = 140  # margen bajo el límite de 260 caracteres de Windows para la ruta completa


def cargar_mapa_sha(repo: Path, rev: str) -> dict[str, str]:
    """path relativo -> sha1 del blob, vía `git ls-tree -r` (no toca el
    filesystem, evita el límite de longitud de ruta de Windows)."""
    salida = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", rev],
        capture_output=True,
        check=True,
        encoding="utf-8",
    ).stdout
    mapa = {}
    for line in salida.splitlines():
        meta, _, path = line.partition("\t")
        sha = meta.split()[2]
        mapa[path] = sha
    return mapa


def git_cat_file(repo: Path, sha: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "cat-file", "blob", sha],
        capture_output=True,
        check=True,
    ).stdout


def nombre_local_para(nombre_original: str) -> tuple[str, str]:
    """Devuelve (nombre_local, motivo)."""
    tiene_invalidos = any(c in nombre_original for c in '<>:"|?*')
    ruta_larga = len(nombre_original) > MAX_PATH_COMPONENTE

    nombre = nombre_original.replace("?", "_")
    if not ruta_larga and len(nombre) <= MAX_PATH_COMPONENTE:
        motivo = "caracter_invalido_ntfs" if tiene_invalidos else "ninguno"
        return nombre, motivo

    stem, punto, ext = nombre.rpartition(".")
    if not punto:
        stem, ext = nombre, ""
    ext = f".{ext}" if ext else ""
    h = hashlib.sha1(nombre_original.encode("utf-8")).hexdigest()[:8]
    recorte = MAX_PATH_COMPONENTE - len(ext) - len(h) - 1
    nombre_corto = f"{stem[:recorte]}_{h}{ext}"
    return nombre_corto, "ruta_demasiado_larga"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="data/raw/plantdoc", help="Ruta al clon local de PlantDoc")
    parser.add_argument("--rev", required=True, help="Commit exacto (ver docs/bitacora/decisiones.md)")
    parser.add_argument(
        "--rutas-faltantes",
        required=True,
        help="Archivo de texto con una ruta relativa por línea (dentro del repo) a rescatar",
    )
    parser.add_argument(
        "--csv-salida",
        default="docs/bitacora/plantdoc_archivos_renombrados.csv",
        help="CSV de trazabilidad nombre original -> nombre local",
    )
    args = parser.parse_args()

    repo = Path(args.repo)
    rutas = [
        line.strip()
        for line in Path(args.rutas_faltantes).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    mapa_sha = cargar_mapa_sha(repo, args.rev)

    filas = []
    for rel in rutas:
        partes = rel.split("/")
        split, clase, nombre_original = partes[0], partes[1], partes[2]
        nombre_local, motivo = nombre_local_para(nombre_original)
        destino = repo / split / clase / nombre_local
        destino.parent.mkdir(parents=True, exist_ok=True)
        contenido = git_cat_file(repo, mapa_sha[rel])
        destino.write_bytes(contenido)
        filas.append(
            {
                "split": split,
                "clase": clase,
                "nombre_original": nombre_original,
                "nombre_local": nombre_local,
                "motivo": motivo,
            }
        )

    Path(args.csv_salida).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "clase", "nombre_original", "nombre_local", "motivo"])
        w.writeheader()
        w.writerows(filas)

    print(f"Rescatados {len(filas)} archivos. Trazabilidad en {args.csv_salida}")


if __name__ == "__main__":
    main()
