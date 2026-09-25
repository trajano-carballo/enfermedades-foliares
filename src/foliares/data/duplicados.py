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
