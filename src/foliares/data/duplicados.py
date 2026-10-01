"""Detección de casi-duplicados entre PlantVillage y PlantDoc por hashing
perceptual (imagehash). Acotada por el caller a las clases candidatas
(tomate, papa, pimiento) para no gastar cómputo de más — ver sesión 2,
tarea 6: es un reporte de pares sospechosos, no un filtro que se aplique solo.
"""

from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image


def calcular_hashes(rutas: list[Path], algoritmo: str = "phash") -> dict[Path, imagehash.ImageHash]:
    """Calcula un hash perceptual por imagen. Las ilegibles se omiten (ya
    quedan reportadas aparte por el chequeo de integridad)."""
    fn = {"phash": imagehash.phash, "ahash": imagehash.average_hash, "dhash": imagehash.dhash}[algoritmo]
    hashes: dict[Path, imagehash.ImageHash] = {}
    for r in rutas:
        try:
            with Image.open(r) as img:
                hashes[r] = fn(img)
        except Exception:  # noqa: BLE001 - imagen ilegible, se omite del hashing
            continue
    return hashes


def pares_sospechosos(
    hashes_a: dict[Path, imagehash.ImageHash],
    hashes_b: dict[Path, imagehash.ImageHash],
    umbral_distancia: int = 6,
) -> list[tuple[Path, Path, int]]:
    """Compara cada hash de `hashes_a` (ej. PlantDoc) contra cada hash de
    `hashes_b` (ej. PlantVillage) y devuelve los pares con distancia de
    Hamming <= umbral_distancia, ordenados por distancia ascendente.

    O(n*m): se espera que el caller ya haya acotado ambos conjuntos a las
    clases candidatas antes de llamar."""
    pares: list[tuple[Path, Path, int]] = []
    for ra, ha in hashes_a.items():
        for rb, hb in hashes_b.items():
            dist = ha - hb
            if dist <= umbral_distancia:
                pares.append((ra, rb, dist))
    pares.sort(key=lambda t: t[2])
    return pares


# --- Sesión 3: comparación vectorizada y con las 8 variantes (4 rotaciones x espejado) ---
import numpy as np  # noqa: E402

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hash_a_uint64(h: imagehash.ImageHash) -> np.uint64:
    bits = h.hash.flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def variantes_d4(img: Image.Image) -> list[Image.Image]:
    """Las 8 variantes del grupo diedral: 4 rotaciones x {identidad, espejado horizontal}."""
    out = []
    for base in (img, img.transpose(Image.FLIP_LEFT_RIGHT)):
        for k in range(4):
            out.append(base.rotate(90 * k, expand=True) if k else base)
    return out


def hashes_uint64(rutas: list[Path], con_variantes: bool = False):
    """Devuelve (rutas_ok, array uint64 de shape (n,) o (n, 8))."""
    ok, filas = [], []
    for r in rutas:
        try:
            with Image.open(r) as img:
                img = img.convert("RGB")
                vs = variantes_d4(img) if con_variantes else [img]
                filas.append([hash_a_uint64(imagehash.phash(v)) for v in vs])
                ok.append(r)
        except Exception:  # noqa: BLE001
            continue
    a = np.array(filas, dtype=np.uint64)
    return ok, (a if con_variantes else a[:, 0])


def hamming_matriz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distancia de Hamming entre cada elemento de `a` (n,) y de `b` (m,) -> (n, m) uint8."""
    out = np.empty((len(a), len(b)), dtype=np.uint8)
    for i, x in enumerate(a):
        xor = np.bitwise_xor(b, x)
        out[i] = _POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
    return out
