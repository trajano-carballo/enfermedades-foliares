"""Preproceso de imágenes: UNA función para ambos dominios (invariante 3), parametrizada por geometría.

Geometrías (decisión e de la sesión 8; la de evaluación se elige en la sesión 9 con PlantDoc-dev):
- `aplastar`: resize directo a `tamano`×`tamano` (deforma si la imagen no es cuadrada). Es la del modelo base.
- `resize_recorte`: el lado corto a `tamano` (conserva la proporción) y recorte central de `tamano`×`tamano`.
- `relleno`: el lado largo a `tamano` (conserva la proporción) y relleno centrado hasta `tamano`×`tamano` con el color
  medio de ImageNet (que normalizado vale 0).
Después, a tensor float32 en [0, 1] y normalización con la media y el desvío de ImageNet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

GEOMETRIAS = ("aplastar", "resize_recorte", "relleno")
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET = (0.229, 0.224, 0.225)
_RELLENO = tuple(round(255 * m) for m in MEDIA_IMAGENET)


def _geometria(im: Image.Image, geometria: str, tamano: int) -> Image.Image:
    w, h = im.size
    if geometria == "aplastar":
        return im.resize((tamano, tamano), Image.BILINEAR)
    if geometria == "resize_recorte":
        escala = tamano / min(w, h)
        nw, nh = max(tamano, round(w * escala)), max(tamano, round(h * escala))
        im = im.resize((nw, nh), Image.BILINEAR)
        izq, arr = (nw - tamano) // 2, (nh - tamano) // 2
        return im.crop((izq, arr, izq + tamano, arr + tamano))
    if geometria == "relleno":
        escala = tamano / max(w, h)
        nw, nh = min(tamano, max(1, round(w * escala))), min(tamano, max(1, round(h * escala)))
        lienzo = Image.new("RGB", (tamano, tamano), _RELLENO)
        lienzo.paste(im.resize((nw, nh), Image.BILINEAR), ((tamano - nw) // 2, (tamano - nh) // 2))
        return lienzo
    raise ValueError(f"Geometría desconocida: {geometria!r}; opciones: {GEOMETRIAS}")


def preprocesar(imagen: Image.Image | str | Path, geometria: str, tamano: int = 224) -> torch.Tensor:
    """Imagen (PIL o ruta) -> tensor (3, tamano, tamano) normalizado con ImageNet."""
    if geometria not in GEOMETRIAS:
        raise ValueError(f"Geometría desconocida: {geometria!r}; opciones: {GEOMETRIAS}")
    if isinstance(imagen, (str, Path)):
        with Image.open(imagen) as im:
            return preprocesar(im.convert("RGB"), geometria, tamano)
    a = np.asarray(_geometria(imagen.convert("RGB"), geometria, tamano), dtype=np.float32) / 255.0
    t = torch.from_numpy(a.copy()).permute(2, 0, 1)
    return (t - torch.tensor(MEDIA_IMAGENET).view(3, 1, 1)) / torch.tensor(STD_IMAGENET).view(3, 1, 1)
