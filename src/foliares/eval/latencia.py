"""Latencia de la app de punta a punta, con y sin CAM (sesión 12, fase 4). Mide `predecir_imagen` (leer el archivo, preproceso,
forward, calibración, top-3 y, con CAM, el mapa superpuesto) en CPU, con calentamiento aparte. No incluye la capa de red de
Gradio. Solo usa imágenes de PV-val, PD-dev y externas (nunca de un test) y no calcula ninguna métrica de desempeño."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np

from foliares.inferencia import predecir_imagen


def percentil(valores, q: float) -> float:
    """Percentil `q` (0-100) por interpolación lineal (el `percentile` por defecto de numpy)."""
    return float(np.percentile(np.asarray(valores, dtype=float), q))


def resumen_ms(tiempos_s) -> dict:
    ms = np.asarray(tiempos_s, dtype=float) * 1000
    return {"n": int(len(ms)), "p50_ms": percentil(ms, 50), "p95_ms": percentil(ms, 95), "media_ms": float(ms.mean()),
            "min_ms": float(ms.min()), "max_ms": float(ms.max())}


def muestra_fija(filas, n: int, semilla: int) -> list[str]:
    """`n` rutas de `filas` (DataFrame con `ruta`) sin reposición, con el generador de `semilla`, en el orden de `filas`."""
    if n > len(filas):
        raise ValueError(f"La muestra pide {n} imágenes y la partición tiene {len(filas)}")
    idx = np.sort(np.random.default_rng(semilla).choice(len(filas), size=n, replace=False))
    return filas.ruta.iloc[idx].tolist()


def cronometrar(modelo, rutas, nombres, temperatura, tau, geometria, tamano, con_cam: bool, calentamiento: int, repeticiones: int) -> list[float]:
    """Segundos de cada llamada a `predecir_imagen` (todas las `rutas`, `repeticiones` veces). Antes, `calentamiento` llamadas
    sin medir sobre las primeras `rutas`."""
    llamar = lambda r: predecir_imagen(modelo, r, nombres, temperatura, tau, geometria, tamano, con_cam=con_cam)  # noqa: E731
    for i in range(calentamiento):
        llamar(rutas[i % len(rutas)])
    tiempos = []
    for _ in range(repeticiones):
        for r in rutas:
            t = time.perf_counter()
            llamar(r)
            tiempos.append(time.perf_counter() - t)
    return tiempos


def modelo_cpu() -> str:
    """Nombre del procesador (Windows: registro; Linux: /proc/cpuinfo; si no, `platform.processor()`)."""
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            return next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except Exception:  # noqa: BLE001
        return platform.processor() or "desconocido"


def entorno_cpu() -> dict:
    import torch

    return {"modelo_cpu": modelo_cpu(), "nucleos_logicos": os.cpu_count(), "hilos_torch": torch.get_num_threads(),
            "torch": torch.__version__, "dispositivo": "cpu"}
