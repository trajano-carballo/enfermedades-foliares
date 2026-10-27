"""MobileNetV3-Small con cabezal GAP -> Linear (invariante 4 de CLAUDE.md).

El cabezal es `AdaptiveAvgPool2d(1) -> Flatten -> Linear(576, n_clases)`, sin capa oculta: es lo que habilita CAM
(`CAM_c = sum_k w_ck * A_k` sobre el mapa de 576x7x7 a 224 px). NO se reutiliza el `classifier` de torchvision
(Linear 576->1024 intermedia). Sin dropout: agregarlo (o no) es una decisión de la etapa 2.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

CANALES_FEATURES = 576


class MobileNetV3SmallGAP(nn.Module):
    """`features` de torchvision (con o sin pesos de ImageNet) + GAP + Linear. `forward` devuelve los logits."""

    def __init__(self, n_clases: int, pesos_imagenet: bool = False):
        super().__init__()
        pesos = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pesos_imagenet else None
        self.features = mobilenet_v3_small(weights=pesos).features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(CANALES_FEATURES, n_clases)

    def mapas(self, x: torch.Tensor) -> torch.Tensor:
        """Último mapa de features (B, 576, 7, 7 a 224 px): la entrada de CAM."""
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.gap(self.mapas(x)), 1))
