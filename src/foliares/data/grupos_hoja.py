"""Grupos de hoja de PlantVillage (carpeta color) a partir de
leaf_grouping/filtered_leafmaps/<clase>.csv, y chequeos de sanidad.

Exploratorio (sesión 3): solo describe; no genera particiones ni escribe en data/splits/.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RE_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}___")
# prefijo = todo lo anterior al número final; el número puede traer decimales o sufijos.
RE_NOMBRE = re.compile(r"^(?P<prefijo>.*?)\s*(?P<num>\d+(?:\.\d+)?)(?P<resto>.*)\.(?:jpe?g|png)$", re.IGNORECASE)


def sin_uuid(nombre: str) -> str:
    return RE_UUID.sub("", nombre)


def parsear_nombre(nombre_sin_uuid: str) -> dict:
    """Separa sesión (prefijo), número y marcas especiales del nombre de archivo."""
    m = RE_NOMBRE.match(nombre_sin_uuid)
    if not m:
        return {"sesion": None, "num": None, "conflicted": "conflicted" in nombre_sin_uuid, "decimal": False}
    return {
        "sesion": m["prefijo"].strip(),
        "num": float(m["num"]),
        "conflicted": "conflicted" in nombre_sin_uuid,
        "decimal": "." in m["num"],
    }


def tabla_imagenes(color_dir: Path, leafmaps_dir: Path, clases: list[str]) -> pd.DataFrame:
    """Una fila por imagen de color/<clase>/: sesión, número, hoja (match exacto por
    nombre sin uuid) y flag `con_grupo`. La hoja es (clase, Leaf #): el '#' solo es
    único dentro de la clase."""
    filas = []
    for clase in clases:
        csv = Path(leafmaps_dir) / f"{clase}.csv"
        mapa = {}
        if csv.exists():
            df = pd.read_csv(csv, dtype=str)
            mapa = dict(zip(df["File Name"], df["Leaf #"]))
        for f in sorted((Path(color_dir) / clase).iterdir()):
            nombre = sin_uuid(f.name)
            p = parsear_nombre(nombre)
            hoja = mapa.get(nombre)
            filas.append({"clase": clase, "archivo": f.name, "nombre": nombre, **p,
                          "hoja": None if hoja is None else f"{clase}:::{hoja}",
                          "con_grupo": hoja is not None,
                          "csv_existe": csv.exists()})
    return pd.DataFrame(filas)


def bloques_contiguos(nums: list[float], tolerancia: int) -> list[list[float]]:
    """Agrupa números ordenados en bloques donde la diferencia entre consecutivos <= tolerancia."""
    nums = sorted(nums)
    bloques, actual = [], [nums[0]]
    for a, b in zip(nums, nums[1:]):
        if b - a <= tolerancia:
            actual.append(b)
        else:
            bloques.append(actual)
            actual = [b]
    bloques.append(actual)
    return bloques


def resumen_por_clase(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla 1b: imágenes, con/sin grupo, hojas distintas e imágenes por hoja."""
    out = []
    for clase, g in df.groupby("clase", sort=False):
        c = g[g.con_grupo]
        por_hoja = c.groupby("hoja").size()
        out.append({
            "clase": clase, "imagenes": len(g), "con_grupo": len(c), "sin_grupo": len(g) - len(c),
            "hojas": por_hoja.size,
            "img_por_hoja_media": round(por_hoja.mean(), 2) if len(por_hoja) else float("nan"),
            "img_por_hoja_min": int(por_hoja.min()) if len(por_hoja) else None,
            "img_por_hoja_max": int(por_hoja.max()) if len(por_hoja) else None,
        })
    return pd.DataFrame(out)


def contiguidad(df: pd.DataFrame, tolerancias=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    """Tabla 1e: sobre imágenes CON grupo, agrupa por (clase, sesión) en bloques de
    números consecutivos y mide cuántas hojas reales quedan partidas entre bloques."""
    c = df[df.con_grupo & df.num.notna()]
    filas = []
    for tol in tolerancias:
        n_bloques, mayor, partidas, hojas = 0, 0, 0, 0
        for (_, _), g in c.groupby(["clase", "sesion"]):
            bl = bloques_contiguos(g.num.tolist(), tol)
            n_bloques += len(bl)
            mayor = max(mayor, max(len(b) for b in bl))
            # bloque de cada imagen
            id_bloque = {}
            for i, b in enumerate(bl):
                for x in b:
                    id_bloque[x] = i
            bl_por_img = g.num.map(id_bloque)
            nb_por_hoja = bl_por_img.groupby(g.hoja).nunique()
            hojas += len(nb_por_hoja)
            partidas += int((nb_por_hoja > 1).sum())
        filas.append({"tolerancia": tol, "bloques": n_bloques, "bloque_mayor": mayor,
                      "hojas": hojas, "hojas_partidas": partidas,
                      "pct_hojas_partidas": round(100 * partidas / hojas, 2)})
    return pd.DataFrame(filas)


def contiguidad_clase(sub: pd.DataFrame, tolerancias=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    """Bloques de números consecutivos para un subconjunto (ej. Tomato mosaic virus)."""
    nums = sorted(sub.num.dropna().unique().tolist())
    return pd.DataFrame([{"tolerancia": t, "bloques": len(bl), "bloque_mayor": max(len(b) for b in bl)}
                         for t in tolerancias for bl in [bloques_contiguos(nums, t)]])
