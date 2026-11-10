"""Imágenes externas a los conjuntos (sesión 12, fase 3; decisión 2 del 2026-11-10): sha256 y chequeo de duplicados por hash
perceptual (phash, el mismo de `foliares.data.duplicados` y el mismo umbral ≤ 6 de las sesiones 4 y 5) contra PlantVillage y
PlantDoc. Cada imagen externa se prueba con las 8 variantes del grupo diedral (rotaciones y espejado); es solo hash: no se
muestra ninguna imagen ni se calcula ninguna métrica."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import imagehash
import numpy as np
import PIL

from foliares.data import duplicados
from foliares.data.congelar import sha256_archivo
from foliares.data.duplicados import hamming_matriz, hashes_uint64

COLUMNAS = ["archivo", "autor", "licencia", "url", "sha256", "resultado_duplicados"]


def leer_planilla(ruta: str | Path) -> list[dict]:
    with open(ruta, encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    if not filas or list(filas[0]) != COLUMNAS:
        raise ValueError(f"La planilla {ruta} no tiene las columnas {COLUMNAS} o no tiene filas")
    return filas


def escribir_planilla(ruta: str | Path, filas: list[dict]) -> None:
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, COLUMNAS, lineterminator="\n")
        w.writeheader()
        w.writerows(filas)


def hashes_con_cache(rutas: list[Path], nombre: str, cache: str | Path | None) -> tuple[list[Path], np.ndarray, str]:
    """`hashes_uint64(rutas)` con caché en disco (fuera del repo). La clave es el sha256 de las rutas con su tamaño, la versión de
    PIL e imagehash y el código de `duplicados.py`: si algo cambia, se recalcula. La clave no incluye el contenido de los archivos
    ni su fecha: un archivo reemplazado por otro del mismo tamaño en la misma ruta daría un acierto con los hashes viejos; con `data/`
    congelado es improbable. Devuelve (rutas_ok, hashes, "hit"|"miss"|"sin_cache")."""
    if cache is None:
        return (*hashes_uint64(list(rutas)), "sin_cache")
    clave = hashlib.sha256("\n".join(f"{r.as_posix()}|{r.stat().st_size}" for r in rutas).encode()
                           + f"|PIL {PIL.__version__}|imagehash {imagehash.__version__}".encode()
                           + Path(duplicados.__file__).read_bytes()).hexdigest()
    archivo = Path(cache) / f"hashes_{re.sub(r'[^A-Za-z0-9]+', '_', nombre).strip('_')}_{clave[:16]}.npz"
    if archivo.is_file():
        z = np.load(archivo, allow_pickle=False)
        return [Path(x) for x in z["rutas"]], z["hashes"], "hit"
    ok, h = hashes_uint64(list(rutas))
    archivo.parent.mkdir(parents=True, exist_ok=True)
    np.savez(archivo, rutas=np.array([r.as_posix() for r in ok]), hashes=h)
    return ok, h, "miss"


def verificar_externas(carpeta: str | Path, filas: list[dict], referencias: dict[str, list[Path]], umbral: int = 6,
                       base: str | Path | None = None, cache: str | Path | None = None, detalle: dict | None = None) -> list[dict]:
    """Completa `sha256` y `resultado_duplicados` de cada fila (copia). `referencias`: {nombre del conjunto: rutas}. Si alguna
    externa es duplicada (distancia ≤ umbral en alguna de las 8 variantes), el resultado empieza con "DUPLICADA" y nombra la
    ruta de referencia (relativa a `base`, p. ej. DATA_ROOT, para no guardar rutas absolutas) y "descartada"; si no,
    "sin duplicados" y la distancia mínima por conjunto. Si se pasa `detalle` (dict), se llena con el phash propio de cada
    externa y, por conjunto, la imagen más cercana y su distancia (`imagenes`), y con el estado del caché (`cache`)."""
    carpeta = Path(carpeta)
    rel = lambda r: (r.resolve().relative_to(Path(base).resolve()) if base else r).as_posix()  # noqa: E731
    rutas = [carpeta / f["archivo"] for f in filas]
    faltan = [str(r) for r in rutas if not r.is_file()]
    if faltan:
        raise FileNotFoundError(f"Faltan imágenes externas: {faltan}")
    ok_ext, h_ext = hashes_uint64(rutas, con_variantes=True)
    if len(ok_ext) != len(rutas):
        raise ValueError("Alguna imagen externa no se pudo leer")
    minimos, cercanas, estado = {}, {}, {}
    for nombre, refs in referencias.items():
        ok_ref, h_ref, estado[nombre] = hashes_con_cache(refs, nombre, cache)
        d = hamming_matriz(h_ext.reshape(-1), h_ref).reshape(len(rutas), 8, len(ok_ref)).min(axis=1)   # (externas, referencias)
        minimos[nombre], cercanas[nombre] = d.min(axis=1), [ok_ref[i] for i in d.argmin(axis=1)]
        minimos[nombre + "_n"] = len(ok_ref)
    salida = []
    for i, f in enumerate(filas):
        dup = [(n, int(minimos[n][i]), cercanas[n][i]) for n in referencias if minimos[n][i] <= umbral]
        if dup:
            txt = "DUPLICADA (phash <= %d, 8 variantes): " % umbral + "; ".join(f"{n}: {d} contra {rel(r)}" for n, d, r in dup) + "; descartada"
        else:
            txt = ("sin duplicados (phash <= %d, 8 variantes); distancia minima: " % umbral
                   + "; ".join(f"{n} {int(minimos[n][i])} de {minimos[n + '_n']} imagenes" for n in referencias))
        salida.append({**f, "sha256": sha256_archivo(rutas[i]), "resultado_duplicados": txt})
    if detalle is not None:
        detalle["cache"] = estado
        detalle["imagenes"] = [{"archivo": f["archivo"], "phash_propio": f"{int(h_ext[i, 0]):016x}",
                                "mas_cercana": {n: {"ruta": rel(cercanas[n][i]), "distancia": int(minimos[n][i])} for n in referencias}}
                               for i, f in enumerate(filas)]
    return salida
