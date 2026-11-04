"""Métricas de clasificación: matriz de confusión, recall y F1 por clase, macro-F1 con y sin mosaic, clasificador trivial.

Convenciones: la matriz tiene filas = clase verdadera y columnas = clase predicha. Precisión, recall o F1 con denominador 0
valen 0 (como `zero_division=0` de scikit-learn). Dos definiciones quedan como parámetro explícito, sin default, porque el
equipo no las fijó todavía (sesión 8): el modo del macro-F1 "sin mosaic" y el desempate del clasificador trivial.
"""

from __future__ import annotations

import numpy as np

MODOS_SIN_CLASE = ("promedio_sin_la_clase", "sin_imagenes_de_la_clase")
DESEMPATES = ("primera_en_orden", "error")


def _a_numpy(v) -> np.ndarray:
    return np.asarray(v.cpu() if hasattr(v, "cpu") else v, dtype=np.int64)


def matriz_confusion(y, p, n_clases: int) -> np.ndarray:
    y, p = _a_numpy(y), _a_numpy(p)
    if y.shape != p.shape:
        raise ValueError(f"y y p tienen formas distintas: {y.shape} vs {p.shape}")
    m = np.zeros((n_clases, n_clases), dtype=np.int64)
    np.add.at(m, (y, p), 1)
    return m


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def recall_por_clase(m: np.ndarray) -> np.ndarray:
    return _div(np.diag(m).astype(float), m.sum(axis=1).astype(float))


def precision_por_clase(m: np.ndarray) -> np.ndarray:
    return _div(np.diag(m).astype(float), m.sum(axis=0).astype(float))


def f1_por_clase(m: np.ndarray) -> np.ndarray:
    p, r = precision_por_clase(m), recall_por_clase(m)
    return _div(2 * p * r, p + r)


def macro_f1(y, p, n_clases: int, clases: list[int] | None = None) -> float:
    """Promedio simple del F1 por clase sobre `clases` (todas por defecto), calculado sobre todas las imágenes."""
    f1 = f1_por_clase(matriz_confusion(y, p, n_clases))
    return float(f1[list(range(n_clases)) if clases is None else clases].mean())


def macro_f1_sin_clase(y, p, n_clases: int, clase: int, modo: str) -> float:
    """Macro-F1 "sin" una clase (p. ej. mosaic). `modo` (sin default, lo decide el equipo):
    - `promedio_sin_la_clase`: F1 por clase sobre todas las imágenes; se promedian las otras clases.
    - `sin_imagenes_de_la_clase`: se quitan las imágenes cuya clase verdadera es `clase` y se promedian las otras clases
      (las predicciones de `clase` sobre imágenes de otras clases siguen contando como error)."""
    otras = [c for c in range(n_clases) if c != clase]
    if modo == "promedio_sin_la_clase":
        return macro_f1(y, p, n_clases, otras)
    if modo == "sin_imagenes_de_la_clase":
        y, p = _a_numpy(y), _a_numpy(p)
        quedan = y != clase
        return macro_f1(y[quedan], p[quedan], n_clases, otras)
    raise ValueError(f"Modo desconocido: {modo!r}; opciones: {MODOS_SIN_CLASE}")


def clase_mayoritaria(y_train, n_clases: int, desempate: str) -> int:
    """Índice de la clase más frecuente en train. `desempate` (sin default): `primera_en_orden` toma la de menor índice;
    `error` falla si hay empate."""
    cuenta = np.bincount(_a_numpy(y_train), minlength=n_clases)
    empatadas = np.flatnonzero(cuenta == cuenta.max())
    if len(empatadas) > 1:
        if desempate == "error":
            raise ValueError(f"Empate de la clase mayoritaria entre los índices {empatadas.tolist()} ({cuenta.max()} imágenes)")
        if desempate != "primera_en_orden":
            raise ValueError(f"Desempate desconocido: {desempate!r}; opciones: {DESEMPATES}")
    return int(empatadas[0])


def resumen(y, p, nombres: list[str], clase_excluible: int | None = None, modo_sin: str | None = None) -> dict:
    """Todo lo que se reporta de una evaluación: macro-F1 (con y, si se pide, sin `clase_excluible`), exactitud, recall y F1
    por clase, matriz de confusión y conteos."""
    n = len(nombres)
    m = matriz_confusion(y, p, n)
    r = {"n": int(m.sum()), "exactitud": float(np.trace(m) / m.sum()) if m.sum() else 0.0,
         "macro_f1": float(f1_por_clase(m).mean()),
         "recall_por_clase": dict(zip(nombres, map(float, recall_por_clase(m)))),
         "f1_por_clase": dict(zip(nombres, map(float, f1_por_clase(m)))),
         "soporte_por_clase": dict(zip(nombres, map(int, m.sum(axis=1)))),
         "matriz_confusion": {"orden": nombres, "filas_verdadera_columnas_predicha": m.tolist()}}
    if clase_excluible is not None and modo_sin is not None:
        r["macro_f1_sin_" + nombres[clase_excluible]] = macro_f1_sin_clase(y, p, n, clase_excluible, modo_sin)
        r["modo_sin"] = modo_sin
    return r
