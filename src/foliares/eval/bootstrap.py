"""Intervalos bootstrap (decisión b de la sesión 9, `docs/protocolo_evaluacion.md`).

- Remuestreo de imágenes con reposición, estratificado por clase: cada réplica conserva cuántas imágenes tiene cada clase.
- IC por percentiles (2,5 y 97,5 para el 95 %), 2.000 réplicas y semilla 42 por defecto (los fija la config).
- Pareado: las mismas réplicas de índices para dos predicciones sobre las mismas imágenes; IC de la diferencia por réplica.
- Independiente: dos conjuntos distintos (p. ej. PV-test y PD-test), cada uno con su propio generador; diferencia por réplica.
La estimación puntual es la métrica sobre la muestra completa; `media_replicas` es el promedio de las réplicas.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Metrica = Callable[[np.ndarray, np.ndarray], float]


def _np(v) -> np.ndarray:
    return np.asarray(v.cpu() if hasattr(v, "cpu") else v)


def indices_estratificados(y, n_replicas: int, rng: np.random.Generator) -> np.ndarray:
    """(n_replicas, N): en cada réplica, para cada clase, tantos índices de esa clase (con reposición) como tiene."""
    y = _np(y)
    grupos = [np.flatnonzero(y == c) for c in np.unique(y)]
    return np.concatenate([g[rng.integers(0, len(g), size=(n_replicas, len(g)))] for g in grupos], axis=1)


def ic_percentil(valores, nivel: float = 0.95) -> tuple[float, float]:
    a = (1 - nivel) / 2 * 100
    lo, hi = np.percentile(_np(valores), [a, 100 - a])
    return float(lo), float(hi)


def _resumen(puntual: float, replicas: np.ndarray, nivel: float) -> dict:
    lo, hi = ic_percentil(replicas, nivel)
    return {"estimacion": float(puntual), "ic_bajo": lo, "ic_alto": hi, "nivel": nivel, "n_replicas": len(replicas),
            "media_replicas": float(replicas.mean())}


def bootstrap(y, p, metrica: Metrica, n_replicas: int = 2000, semilla: int = 42, nivel: float = 0.95) -> dict:
    y, p = _np(y), _np(p)
    idx = indices_estratificados(y, n_replicas, np.random.default_rng(semilla))
    reps = np.array([metrica(y[i], p[i]) for i in idx])
    return _resumen(metrica(y, p), reps, nivel)


def bootstrap_pareado(y, p_a, p_b, metrica: Metrica, n_replicas: int = 2000, semilla: int = 42, nivel: float = 0.95) -> dict:
    """IC de metrica(p_a) − metrica(p_b) con las MISMAS réplicas de imágenes. `excluye_0` si el IC no contiene el 0."""
    y, p_a, p_b = _np(y), _np(p_a), _np(p_b)
    if not (len(y) == len(p_a) == len(p_b)):
        raise ValueError("El bootstrap pareado necesita las mismas imágenes en las dos predicciones")
    idx = indices_estratificados(y, n_replicas, np.random.default_rng(semilla))
    reps = np.array([metrica(y[i], p_a[i]) - metrica(y[i], p_b[i]) for i in idx])
    r = _resumen(metrica(y, p_a) - metrica(y, p_b), reps, nivel)
    return {**r, "excluye_0": bool(r["ic_bajo"] > 0 or r["ic_alto"] < 0)}


def bootstrap_diferencia_independiente(y1, p1, y2, p2, metrica: Metrica, n_replicas: int = 2000, semilla: int = 42,
                                       nivel: float = 0.95) -> dict:
    """IC de metrica(conjunto 1) − metrica(conjunto 2), remuestreando cada conjunto por separado (generadores
    `default_rng([semilla, 1])` y `default_rng([semilla, 2])`) y restando réplica a réplica."""
    y1, p1, y2, p2 = map(_np, (y1, p1, y2, p2))
    i1 = indices_estratificados(y1, n_replicas, np.random.default_rng([semilla, 1]))
    i2 = indices_estratificados(y2, n_replicas, np.random.default_rng([semilla, 2]))
    reps = np.array([metrica(y1[a], p1[a]) - metrica(y2[b], p2[b]) for a, b in zip(i1, i2)])
    r = _resumen(metrica(y1, p1) - metrica(y2, p2), reps, nivel)
    return {**r, "excluye_0": bool(r["ic_bajo"] > 0 or r["ic_alto"] < 0)}
