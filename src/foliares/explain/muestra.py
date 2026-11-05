"""Muestra fija para la grilla de CAM (decisión c de la sesión 9, `docs/protocolo_evaluacion.md`).

PlantDoc-dev: 2 imágenes por clase (1 acierto y 1 error cuando haya; si no, 2 aciertos o 2 errores), elegidas con semilla 42
DESPUÉS de predecir con la geometría elegida (se parte de las predicciones guardadas); más 1 imagen por clase de PV-val como
contraste. Nunca imágenes de un test: `exigir_particion_permitida` lo verifica antes de mostrar nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foliares.utils.seeds import rng

PARTICIONES_PERMITIDAS = {("plantdoc", "dev"), ("plantvillage", "val")}


def exigir_particion_permitida(dataset) -> None:
    """Solo PlantDoc-dev y PlantVillage-val pueden mostrarse en la grilla; cualquier test (o train) se rechaza."""
    par = (getattr(dataset, "dataset", None), getattr(dataset, "particion", None))
    if par not in PARTICIONES_PERMITIDAS:
        raise ValueError(f"La grilla de CAM solo puede mostrar PlantDoc-dev y PlantVillage-val, no {par}: nunca imágenes de un test")


def seleccionar_pd_dev(pred: pd.DataFrame, n_clases: int, semilla: int, por_clase: int = 2) -> pd.DataFrame:
    """`pred` con columnas `etiqueta` y `pred` (una fila por imagen, en el orden del Dataset). Devuelve las filas elegidas con
    `fila` (posición en `pred`) y `tipo` (acierto o error)."""
    pred = pred.reset_index(drop=True).assign(fila=lambda d: d.index)
    elegidas = []
    for c in range(n_clases):
        sub = pred[pred.etiqueta == c]
        aciertos, errores = sub[sub.pred == c], sub[sub.pred != c]
        g = rng(semilla, "cam_pd_dev", c)

        def tomar(df: pd.DataFrame, k: int, tipo: str) -> pd.DataFrame:
            if len(df) == 0 or k == 0:
                return df.iloc[0:0].assign(tipo=tipo)
            return df.iloc[np.sort(g.choice(len(df), size=min(k, len(df)), replace=False))].assign(tipo=tipo)

        if len(aciertos) and len(errores):
            k_ac = por_clase // 2
            elegidas += [tomar(aciertos, k_ac, "acierto"), tomar(errores, por_clase - k_ac, "error")]
        elif len(aciertos):
            elegidas.append(tomar(aciertos, por_clase, "acierto"))
        else:
            elegidas.append(tomar(errores, por_clase, "error"))
    return pd.concat(elegidas, ignore_index=True)


def elegir_pv_val(filas: pd.DataFrame, n_clases: int, semilla: int) -> pd.DataFrame:
    """1 imagen por clase de PV-val (`filas` = las filas del Dataset, con `etiqueta`), al azar con semilla derivada."""
    filas = filas.reset_index(drop=True).assign(fila=lambda d: d.index)
    return pd.concat([filas[filas.etiqueta == c].iloc[[int(rng(semilla, "cam_pv_val", c).integers(int((filas.etiqueta == c).sum())))]]
                      for c in range(n_clases)], ignore_index=True)
