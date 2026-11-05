"""IC bootstrap de la exactitud y del recall, la precisión y el F1 por clase, desde predicciones por imagen YA guardadas.

Cierra el punto M1 de la review de la sesión 9 (invariante 6: toda métrica OOD con IC): no reevalúa el modelo ni lee imágenes;
parte de los CSV de predicciones que escribió `scripts/evaluar.py` y usa `foliares.eval.bootstrap` (remuestreo con reposición
estratificado por clase verdadera, mismas réplicas para todas las métricas). Convención: precisión, recall o F1 con denominador 0
valen 0 (como en `foliares.eval.metricas`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foliares.eval import bootstrap as B
from foliares.eval import metricas as M


def metricas_vector(nombres: list[str]):
    """Función `(y, p) -> {nombre: valor}` con exactitud, macro-F1 y recall/precisión/F1 de cada clase (claves `recall|clase`...)."""
    k = len(nombres)

    def f(y, p):
        m = M.matriz_confusion(y, p, k)
        rec, pre, f1 = M.recall_por_clase(m), M.precision_por_clase(m), M.f1_por_clase(m)
        out = {"exactitud": float(np.trace(m) / m.sum()), "macro_f1": float(f1.mean())}
        for i, n in enumerate(nombres):
            out[f"recall|{n}"], out[f"precision|{n}"], out[f"f1|{n}"] = float(rec[i]), float(pre[i]), float(f1[i])
        return out
    return f


def ic_de_predicciones(pred: pd.DataFrame, nombres: list[str], replicas: int, semilla: int, nivel: float) -> dict:
    """`pred` con columnas `etiqueta` y `pred`. Devuelve exactitud, macro-F1 y, por clase, soporte, recall, precisión y F1, cada
    una con estimación e IC percentil."""
    y, p = pred.etiqueta.to_numpy(), pred.pred.to_numpy()
    r = B.bootstrap_multiple(y, p, metricas_vector(nombres), replicas, semilla, nivel)
    sub = lambda k: {c: r[f"{k}|{n}"] for c, n in enumerate(nombres)}  # noqa: E731
    soporte = np.bincount(y, minlength=len(nombres))
    return {"n": int(len(y)), "exactitud": r["exactitud"], "macro_f1": r["macro_f1"],
            "por_clase": {n: {"soporte": int(soporte[c]), "recall": sub("recall")[c], "precision": sub("precision")[c],
                              "f1": sub("f1")[c]} for c, n in enumerate(nombres)}}
