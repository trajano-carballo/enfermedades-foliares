"""Aumentación de PV-train (decisión c de la sesión 10). SOLO para entrenamiento: PV-val, PlantDoc y los tests usan `preprocesar`
con la geometría del protocolo (invariante 3).

- `ninguna`: el pipeline de train es exactamente `preprocesar(ruta, geometria, tamano)`, el del modelo base.
- `dirigida`: lectura igual que `preprocesar` (`leer_rgb`: EXIF y alfa sobre blanco), `RandomResizedCrop(tamano, scale)`,
  `RandomHorizontalFlip`, `RandomRotation(grados)` y `ColorJitter(brillo, contraste, saturación)` de torchvision, y la misma
  conversión a tensor y normalización de ImageNet que `preprocesar`. El azar sale del generador global de torch (por época se
  fija con una semilla derivada; ver `foliares.training.finetune`).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from foliares.data.preproceso import MEDIA_IMAGENET, STD_IMAGENET, leer_rgb, preprocesar

TIPOS = ("ninguna", "dirigida")


def a_tensor_normalizado(im: Image.Image) -> torch.Tensor:
    """La misma conversión que el final de `preprocesar`: float32 en [0, 1] y normalización de ImageNet."""
    a = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(a.copy()).permute(2, 0, 1)
    return (t - torch.tensor(MEDIA_IMAGENET).view(3, 1, 1)) / torch.tensor(STD_IMAGENET).view(3, 1, 1)


class AumentacionDirigida:
    """Transformación de PV-train como objeto (se puede serializar: los procesos de lectura del DataLoader la reciben por pickle)."""

    def __init__(self, cfg: dict, tamano: int):
        self.t = T.Compose([T.RandomResizedCrop(tamano, scale=tuple(cfg["escala_recorte"])),
                            T.RandomHorizontalFlip(),
                            T.RandomRotation(cfg["rotacion_grados"]),
                            T.ColorJitter(brightness=cfg["brillo"], contrast=cfg["contraste"], saturation=cfg["saturacion"])])

    def __call__(self, ruta: str | Path) -> torch.Tensor:
        with Image.open(ruta) as im:
            return a_tensor_normalizado(self.t(leer_rgb(im)))


def transformacion_train(cfg: dict, geometria: str, tamano: int) -> Callable[[str | Path], torch.Tensor]:
    """Función ruta -> tensor para PV-train según `cfg` (sección `aumentacion` de la config). Siempre serializable."""
    tipo = cfg.get("tipo")
    if tipo == "ninguna":
        return partial(preprocesar, geometria=geometria, tamano=tamano)
    if tipo != "dirigida":
        raise ValueError(f"Aumentación desconocida: {tipo!r}; opciones: {TIPOS}")
    return AumentacionDirigida(cfg, tamano)
