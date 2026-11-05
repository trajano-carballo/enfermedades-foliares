"""CAM (Class Activation Mapping) para el cabezal GAP -> Linear (invariante 4). Es la única explicabilidad del proyecto.

CAM_c = sum_k w_ck * A_k sobre el último mapa de features (576×7×7 a 224 px). Sale del MISMO forward que la predicción, sin
backward y sin gradientes (Grad-CAM está descartado del alcance). Nota: el CAM no incluye el sesgo de `fc`; el promedio espacial
del CAM más el sesgo da el logit de la clase.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def cam(mapas: torch.Tensor, pesos: torch.Tensor, clases: torch.Tensor) -> torch.Tensor:
    """mapas (B, K, H, W), pesos (C, K), clases (B,) -> (B, H, W) con CAM_c = sum_k w_ck A_k (sin gradientes)."""
    with torch.no_grad():
        w = pesos.detach()[clases]
        return torch.einsum("bk,bkhw->bhw", w, mapas.detach())


def reescalar(mapa_cam: torch.Tensor, tamano: int) -> torch.Tensor:
    """(B, H, W) -> (B, tamano, tamano) por interpolación bilineal y normalizado a [0, 1] POR IMAGEN (min-max): sirve para
    ver DÓNDE se activa, no cuánto; una imagen sin variación queda en ceros."""
    with torch.no_grad():
        m = F.interpolate(mapa_cam.detach()[:, None].float(), size=(tamano, tamano), mode="bilinear", align_corners=False)[:, 0]
        lo, hi = m.amin(dim=(1, 2), keepdim=True), m.amax(dim=(1, 2), keepdim=True)
        return torch.where(hi - lo > 1e-12, (m - lo) / (hi - lo).clamp_min(1e-12), torch.zeros_like(m))


def superponer(imagen: np.ndarray, cam01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """imagen (H, W, 3) en [0, 1] y cam01 (H, W) en [0, 1] -> imagen con el mapa de calor (jet) mezclado con `alpha`."""
    from matplotlib import colormaps

    calor = colormaps["jet"](np.clip(cam01, 0, 1))[..., :3]
    return np.clip((1 - alpha) * imagen + alpha * calor, 0, 1)


def desnormalizar(x: torch.Tensor, media, std) -> np.ndarray:
    """(B, 3, H, W) normalizado con ImageNet -> (B, H, W, 3) en [0, 1], lo que el modelo realmente vio."""
    m, s = torch.tensor(media).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1)
    return (x.detach().cpu() * s + m).clamp(0, 1).permute(0, 2, 3, 1).numpy()


@torch.inference_mode()
def cam_de_modelo(modelo, x: torch.Tensor, clases: torch.Tensor | None = None) -> dict:
    """Un forward del modelo completo: logits, probabilidades, predicción y el CAM de `clases` (por defecto, la predicha).
    Todo bajo `inference_mode`: no hay grafo de autograd ni backward."""
    modelo.eval()
    mapas = modelo.mapas(x)
    logits = modelo.fc(torch.flatten(modelo.gap(mapas), 1))
    probs = torch.softmax(logits, 1)
    pred = probs.argmax(1)
    objetivo = pred if clases is None else clases
    return {"cam": cam(mapas, modelo.fc.weight, objetivo), "logits": logits, "probs": probs, "pred": pred, "clase_cam": objetivo}
