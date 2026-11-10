"""Camino de UNA imagen para la app (etapa 6, sesión 12): preproceso, inferencia, calibración por temperatura, cartel de
confianza baja y CAM. La app (`app/app.py`) solo importa esto: no reimplementa nada. Es el mismo preproceso (`preprocesar`),
los mismos pesos y la misma calibración (`softmax_t`) que usa la evaluación (invariante 3).

Decisión del equipo (2026-11-10, opción B): se muestra siempre el top-k calibrado y el CAM; si la probabilidad máxima
calibrada es < τ, `baja_confianza` es True y la app agrega el cartel. Responde si p ≥ τ (igual que `metricas_sistema`).
"""

from __future__ import annotations

import numpy as np
import torch

from foliares.data.preproceso import MEDIA_IMAGENET, STD_IMAGENET, preprocesar
from foliares.eval.calibracion import softmax_t
from foliares.explain import cam as C


@torch.inference_mode()
def predecir_imagen(modelo, imagen, nombres: list[str], temperatura: float, tau: float, geometria: str, tamano: int = 224,
                    con_cam: bool = True, k: int = 3) -> dict:
    """`imagen`: ruta o PIL. Devuelve el top-k calibrado [(clase, probabilidad)], la probabilidad máxima, `baja_confianza`
    (p máx < τ) y, con `con_cam`, el CAM de la clase predicha superpuesto sobre lo que el modelo vio (uint8, tamano×tamano×3)."""
    x = preprocesar(imagen, geometria, tamano)[None]
    modelo.eval()
    if con_cam:
        out = C.cam_de_modelo(modelo, x)         # un solo forward: logits y mapas
        logits = out["logits"]
    else:
        logits = modelo(x)
    p = softmax_t(logits.float().numpy(), temperatura)[0]
    orden = np.argsort(-p)[:k]
    resultado = {"top": [(nombres[i], float(p[i])) for i in orden], "prob_max": float(p.max()),
                 "baja_confianza": bool(p.max() < tau), "cam": None}
    if con_cam:
        cam01 = C.reescalar(out["cam"], tamano)[0].numpy()
        base = C.desnormalizar(x, MEDIA_IMAGENET, STD_IMAGENET)[0]
        resultado["cam"] = (C.superponer(base, cam01) * 255).round().astype(np.uint8)
    return resultado
