"""Lote de PlantVillage-TRAIN para la prueba de humo de Colab.

Guarda de invariantes (1 y 2 de CLAUDE.md): esta función solo lee `plantvillage_particion.csv`, solo toma filas con
`particion == "train"` y `clase_eval`, y rechaza cualquier ruta que no sea de PlantVillage. No hay parámetro para elegir
otra partición: PlantDoc y los tests no se pueden pedir por acá.

El preproceso (resize + normalización ImageNet) es SOLO para la prueba de humo; el preproceso definitivo se decide en la
etapa 2 (con PlantDoc-dev) y es el mismo para ambos dominios.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from foliares.utils.seeds import rng

ARCHIVO_SPLIT = "plantvillage_particion.csv"
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET = (0.229, 0.224, 0.225)


def clases_eval(splits_dir: str | Path) -> list[str]:
    """Las 12 clases de evaluación, ordenadas (índice de clase = posición en esta lista)."""
    df = _leer(splits_dir)
    return sorted(df.clase[_es_eval(df)].unique())


def _leer(splits_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(splits_dir) / ARCHIVO_SPLIT, dtype=str, keep_default_na=False)


def _es_eval(df: pd.DataFrame) -> pd.Series:
    return df.clase_eval.str.lower() == "true"


def elegir_rutas_train(splits_dir: str | Path, n: int, semilla: int) -> pd.DataFrame:
    """`n` filas de PlantVillage-train (clases de evaluación), elegidas con una semilla derivada y estable."""
    df = _leer(splits_dir)
    train = df[_es_eval(df) & (df.particion == "train")].sort_values("ruta_color").reset_index(drop=True)
    if len(train) < n:
        raise ValueError(f"Hay {len(train)} imágenes de train y se pidieron {n}")
    elegidas = train.iloc[np.sort(rng(semilla, "lote_humo").choice(len(train), size=n, replace=False))]
    ajenas = [r for r in elegidas.ruta_color if not r.startswith("raw/plantvillage/") or "plantdoc" in r.lower()]
    if ajenas or set(elegidas.particion) != {"train"}:
        raise RuntimeError(f"El lote de humo solo puede ser de PlantVillage-train; rutas ajenas: {ajenas[:3]}")
    return elegidas


def preprocesar_humo(ruta: str | Path, tamano: int) -> torch.Tensor:
    with Image.open(ruta) as im:
        a = np.asarray(im.convert("RGB").resize((tamano, tamano), Image.BILINEAR), dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1)
    media, std = torch.tensor(MEDIA_IMAGENET).view(3, 1, 1), torch.tensor(STD_IMAGENET).view(3, 1, 1)
    return (t - media) / std


def cargar_lote(splits_dir: str | Path, data_root: str | Path, n: int, semilla: int, tamano: int = 224):
    """(x, y, rutas): x (n, 3, tamano, tamano) float32, y (n,) int64 con el índice de `clases_eval`, rutas relativas."""
    filas = elegir_rutas_train(splits_dir, n, semilla)
    indice = {c: i for i, c in enumerate(clases_eval(splits_dir))}
    x = torch.stack([preprocesar_humo(Path(data_root) / r, tamano) for r in filas.ruta_color])
    y = torch.tensor([indice[c] for c in filas.clase], dtype=torch.int64)
    return x, y, list(filas.ruta_color)
