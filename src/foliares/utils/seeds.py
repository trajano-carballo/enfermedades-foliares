"""Semillas: fijado global (python, numpy, torch) y semillas derivadas estables."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def fijar_semillas(semilla: int, determinista: bool = False) -> dict:
    """Fija las semillas de python, numpy y (si está instalado) torch.

    `determinista=True` además pide a torch algoritmos deterministas (más lento).
    Devuelve qué se fijó, para registrarlo junto a los resultados.
    PYTHONHASHSEED solo afecta a procesos hijos: el intérprete actual ya arrancó.
    """
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)
    fijado = {"semilla": semilla, "python": True, "numpy": True, "torch": False, "cuda": False}
    try:
        import torch
    except ImportError:
        return fijado
    torch.manual_seed(semilla)
    fijado["torch"] = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
        fijado["cuda"] = True
    if determinista:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return fijado


def semilla_derivada(base: int, *claves) -> int:
    """Semilla de 32 bits estable (independiente del orden de ejecución y de la máquina)
    a partir de una semilla base y claves (p. ej. clase, sesión). No usa hash() de python."""
    texto = "|".join([str(base), *map(str, claves)])
    return int.from_bytes(hashlib.sha256(texto.encode("utf-8")).digest()[:4], "big")


def rng(base: int, *claves) -> np.random.Generator:
    """Generador de numpy con semilla derivada de (base, claves)."""
    return np.random.default_rng(semilla_derivada(base, *claves))
