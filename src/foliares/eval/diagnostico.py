"""Diagnóstico de bugs sobre PlantDoc-dev (sesión 9): ¿la caída del modelo base es un error de mapeo, de permutación o de
preproceso, o es real? Solo PD-dev (con las predicciones ya guardadas, geometría `aplastar`) y 20 imágenes de PV-val como
control. No toca ningún test y no cambia el modelo, la geometría ni las reglas.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from foliares.eval import metricas as M


def pares_del_doc(texto: str) -> dict[str, str]:
    """Clase de PlantVillage -> carpeta de PlantDoc, leído de las tablas de `docs/mapeo_clases.md` (primera y tercera celda)."""
    pares = {}
    for linea in texto.splitlines():
        celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
        if len(celdas) >= 3 and celdas[0].startswith("`") and celdas[0].endswith("`"):
            m = re.match(r"`([^`]+)`", celdas[2])
            if m:
                pares[celdas[0].strip("`")] = m.group(1)
    return pares


def mapeo_efectivo(pv: list[str], pdn: list[str], doc: dict[str, str], filas: pd.DataFrame, pred: pd.DataFrame,
                   por_clase: int = 3) -> dict:
    """Índice -> clase PV -> carpeta de PD según la config y según el doc; y consistencia de las filas del Dataset de PD-dev
    (clase del CSV = carpeta de la ruta, etiqueta = índice de esa carpeta, orden y etiquetas = las de las predicciones)."""
    indices = [{"indice": i, "clase_pv": pv[i], "carpeta_pd_config": pdn[i], "carpeta_pd_doc": doc.get(pv[i]),
                "coincide": doc.get(pv[i]) == pdn[i]} for i in range(len(pv))]
    carpeta = filas.ruta.str.split("/").str[3]
    indice_de = {n: i for i, n in enumerate(pdn)}
    muestra = []
    for i in range(len(pv)):
        for _, r in filas[filas.etiqueta == i].sort_values("ruta").head(por_clase).iterrows():
            c = r.ruta.split("/")[3]
            muestra.append({"ruta": r.ruta, "carpeta": c, "etiqueta": int(r.etiqueta), "clase_pv": pv[int(r.etiqueta)],
                            "coincide_con_el_doc": doc.get(pv[int(r.etiqueta)]) == c})
    return {"indices": indices, "todas_coinciden_con_el_doc": all(x["coincide"] for x in indices),
            "clase_del_csv_igual_a_la_carpeta": bool((carpeta == filas.clase).all()),
            "etiqueta_igual_al_indice_de_la_carpeta": bool((filas.etiqueta == carpeta.map(indice_de)).all()),
            "mismo_orden_que_las_predicciones": list(filas.ruta) == list(pred.ruta),
            "misma_etiqueta_que_las_predicciones": bool((filas.etiqueta.values == pred.etiqueta.values).all()),
            "muestra_3_por_clase": muestra, "muestra_coincide_con_el_doc": all(x["coincide_con_el_doc"] for x in muestra)}


def metricas_pd_dev(pred: pd.DataFrame, nombres: list[str], trivial: int) -> dict:
    y, p = pred.etiqueta.values, pred.pred.values
    k = len(nombres)
    m = M.matriz_confusion(y, p, k)
    rec, pre, f1 = M.recall_por_clase(m), M.precision_por_clase(m), M.f1_por_clase(m)
    mt = M.matriz_confusion(y, np.full_like(y, trivial), k)
    j = int(m.sum(0).argmax())
    acierto = y == p
    return {"n": int(len(y)), "exactitud": float(np.trace(m) / m.sum()), "macro_f1": float(f1.mean()),
            "por_clase": [{"indice": i, "clase": nombres[i], "verdaderas": int(m[i].sum()), "predichas": int(m[:, i].sum()),
                           "recall": float(rec[i]), "precision": float(pre[i]), "f1": float(f1[i])} for i in range(k)],
            "matriz_cuentas": {"orden": nombres, "filas_verdadera_columnas_predicha": m.tolist()},
            "distribucion_predichas": dict(zip(nombres, map(int, m.sum(0)))),
            "distribucion_verdaderas": dict(zip(nombres, map(int, m.sum(1)))),
            "clase_mas_predicha": {"clase": nombres[j], "n": int(m.sum(0)[j]), "proporcion": float(m.sum(0)[j] / m.sum())},
            "trivial": {"clase": nombres[trivial], "exactitud": float(np.trace(mt) / mt.sum()),
                        "macro_f1": float(M.f1_por_clase(mt).mean()), "recall_de_esa_clase": float(M.recall_por_clase(mt)[trivial])},
            "azar_uniforme_exactitud": 1 / k,
            "confianza_media": {"clase_predicha": float(pred.prob_pred.mean()), "aciertos": float(pred.prob_pred[acierto].mean()),
                                "errores": float(pred.prob_pred[~acierto].mean())}}


def chequeo_permutacion(m: np.ndarray, nombres: list[str]) -> dict:
    """¿Hay un corrimiento sistemático de índices? Mira si la diagonal es el máximo de cada fila, a dónde va la mayoría de cada
    fila y cuánto sube la exactitud con la mejor reasignación posible de las columnas (asignación óptima)."""
    from scipy.optimize import linear_sum_assignment

    m = np.asarray(m)
    k = len(m)
    diag = [i for i in range(k) if int(m[i].argmax()) == i]
    filas = [{"clase": nombres[i], "columna_mayoritaria": nombres[int(m[i].argmax())], "cuentas": int(m[i].max()),
              "de": int(m[i].sum()), "en_la_diagonal": int(m[i].argmax()) == i} for i in range(k)]
    r, c = linear_sum_assignment(-m)
    return {"filas_con_la_diagonal_como_maximo": len(diag), "de": k, "clases_con_la_diagonal_como_maximo": [nombres[i] for i in diag],
            "columnas_mayoritarias_distintas": len({f["columna_mayoritaria"] for f in filas}), "por_fila": filas,
            "asignacion_optima": {"exactitud": float(m[r, c].sum() / m.sum()), "exactitud_identidad": float(np.trace(m) / m.sum()),
                                  "imagenes_ganadas": int(m[r, c].sum() - np.trace(m)),
                                  "permutacion_distinta_de_la_identidad": {nombres[a]: nombres[b] for a, b in zip(r, c) if a != b}}}


@torch.inference_mode()
def control_preproceso(modelo, ds_pv_val, features_entrenamiento: torch.Tensor, ds_pd_dev, pred_pd_dev: pd.DataFrame,
                       n: int = 20, semilla: int = 42) -> dict:
    """Mismo camino que PD-dev (Dataset + preprocesar + modelo completo desde el checkpoint) sobre `n` imágenes de PV-val: las
    features y los logits tienen que coincidir con los del entrenamiento (`features_entrenamiento`, misma fila, mismo `fc`).
    También recalcula `n` probabilidades de PD-dev y las compara con las guardadas."""
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    idx = np.sort(np.random.default_rng(semilla).choice(len(ds_pv_val), n, replace=False))
    x = torch.stack([ds_pv_val[int(i)][0] for i in idx])
    logits = modelo(x)
    logits_ref = modelo.fc(features_entrenamiento[idx])
    feats = torch.flatten(modelo.gap(modelo.mapas(x)), 1)
    ref = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features.state_dict()
    propio = modelo.features.state_dict()
    xs = torch.stack([ds_pd_dev[i][0] for i in range(n)])
    k = logits.shape[1]
    p_nuevas = torch.softmax(modelo(xs), 1).numpy()
    p_guardadas = pred_pd_dev[[f"p_{j}" for j in range(k)]].values[:n]
    return {"n_pv_val": n, "indices_pv_val": idx.tolist(),
            "max_abs_dif_features": float((feats - features_entrenamiento[idx]).abs().max()),
            "max_abs_dif_logits": float((logits - logits_ref).abs().max()),
            "predicciones_iguales": bool((logits.argmax(1) == logits_ref.argmax(1)).all()),
            "aciertos_pv_val": int((logits.argmax(1) == ds_pv_val.etiquetas[idx]).sum()),
            "max_abs_dif_pesos_backbone_vs_imagenet": max(float((v - propio[kk]).abs().max()) for kk, v in ref.items()
                                                          if v.dtype.is_floating_point),
            "modelo_en_modo_eval": not modelo.training,
            "backbone_requires_grad": any(p.requires_grad for p in modelo.features.parameters()),
            "n_pd_dev_recalculadas": n, "max_abs_dif_probabilidades_pd_dev": float(np.abs(p_nuevas - p_guardadas).max())}


def buscar_features(cache_dir: Path, archivo_del_run: str) -> Path:
    """Archivo de features del run de la sesión 8 (el JSON guarda su ruta, absoluta o relativa: se usa el nombre)."""
    ruta = Path(cache_dir) / Path(str(archivo_del_run).replace("\\", "/")).name
    if not ruta.is_file():
        raise FileNotFoundError(f"No está el caché de features del entrenamiento: {ruta}")
    return ruta
