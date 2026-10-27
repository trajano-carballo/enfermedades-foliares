"""Checkpoints reanudables (obligatorios en Colab gratuito: la sesión se corta).

Guarda modelo, optimizador, paso y los estados de los generadores (python, numpy, torch y CUDA) para reanudar donde
quedó. Todo se guarda con tipos simples (tensores, listas, números) para poder cargar con `weights_only=True`: un
checkpoint en una carpeta compartida de Drive no debería poder ejecutar código al abrirse. La escritura es atómica
(archivo temporal + `os.replace`): una sesión cortada a mitad de `torch.save` no deja un checkpoint truncado.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from foliares.data.congelar import sha256_archivo

FORMATO = 1


def estados_rng() -> dict:
    """Estado de los generadores de python, numpy, torch y (si hay) CUDA, solo con tipos seguros para `weights_only`."""
    version, estado, gauss = random.getstate()
    _, clave, pos, tiene_gauss, gauss_np = np.random.get_state()
    return {"python": {"version": version, "estado": list(estado), "gauss": gauss},
            "numpy": {"clave": torch.from_numpy(clave.astype(np.int64)), "pos": int(pos),
                      "tiene_gauss": int(tiene_gauss), "gauss": float(gauss_np)},
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restaurar_rng(estados: dict) -> None:
    """Los estados de los generadores viven en CPU: `numpy`, `torch.set_rng_state` y `random` no aceptan tensores en la GPU
    (con `torch.load(map_location=cuda)` el checkpoint entero, estados de RNG incluidos, llega a la GPU), así que todos
    los tensores se traen a CPU acá, sea cual sea el `map_location` con el que se cargó."""
    py, npy = estados["python"], estados["numpy"]
    random.setstate((py["version"], tuple(py["estado"]), py["gauss"]))
    np.random.set_state(("MT19937", npy["clave"].cpu().numpy().astype(np.uint32), npy["pos"], npy["tiene_gauss"], npy["gauss"]))
    torch.set_rng_state(estados["torch"].cpu())
    if estados.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in estados["cuda"]])


def guardar_checkpoint(ruta: str | Path, modelo: torch.nn.Module, optimizador: torch.optim.Optimizer, paso: int,
                       extra: dict | None = None) -> dict:
    """Escribe el checkpoint de forma atómica. Devuelve {ruta, bytes, sha256} para registrarlo en el JSON de la corrida.
    `extra` solo con tipos simples (str, números, listas, dicts)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(ruta.name + ".tmp")
    try:
        torch.save({"formato": FORMATO, "paso": int(paso), "modelo": modelo.state_dict(),
                    "optimizador": optimizador.state_dict(), "rng": estados_rng(), "extra": extra or {}}, tmp)
        os.replace(tmp, ruta)
    finally:
        tmp.unlink(missing_ok=True)
    return {"ruta": str(ruta), "bytes": ruta.stat().st_size, "sha256": sha256_archivo(ruta)}


def cargar_checkpoint(ruta: str | Path, modelo: torch.nn.Module, optimizador: torch.optim.Optimizer | None = None,
                      map_location="cpu", restaurar_generadores: bool = True) -> dict:
    """Carga modelo (y optimizador y generadores si se piden). Devuelve {paso, extra}.

    Por defecto el checkpoint se lee a CPU: `load_state_dict` copia cada tensor al dispositivo de los parámetros del
    modelo y del optimizador, así que no hace falta (ni conviene) cargarlo entero en la GPU."""
    ck = torch.load(ruta, map_location=map_location, weights_only=True)
    if ck.get("formato") != FORMATO:
        raise ValueError(f"Formato de checkpoint desconocido: {ck.get('formato')!r} (se esperaba {FORMATO})")
    modelo.load_state_dict(ck["modelo"])
    if optimizador is not None:
        optimizador.load_state_dict(ck["optimizador"])
    if restaurar_generadores:
        restaurar_rng(ck["rng"])
    return {"paso": ck["paso"], "extra": ck["extra"]}
