"""Invariante 4: el cabezal es GAP -> Linear, sin capa oculta (habilita CAM)."""

import torch
from torch import nn

from foliares.models.mobilenet import CANALES_FEATURES, MobileNetV3SmallGAP


def test_cabezal_gap_linear_sin_capa_oculta():
    m = MobileNetV3SmallGAP(n_clases=12)
    assert not hasattr(m, "classifier")                       # no se reutiliza el classifier de torchvision
    lineales = [x for x in m.modules() if isinstance(x, nn.Linear)]
    assert len(lineales) == 1 and (lineales[0].in_features, lineales[0].out_features) == (CANALES_FEATURES, 12)
    assert isinstance(m.gap, nn.AdaptiveAvgPool2d) and m.gap.output_size == 1
    assert not any(isinstance(x, nn.Dropout) for x in m.modules())


def test_forma_de_los_mapas_y_los_logits_y_cam_posible():
    m = MobileNetV3SmallGAP(n_clases=12).eval()
    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        a, logits = m.mapas(x), m(x)
    assert a.shape == (2, 576, 7, 7) and logits.shape == (2, 12)
    # CAM_c = sum_k w_ck A_k reproduce el logit de la clase tras promediar espacialmente (sale del mismo forward)
    cam = torch.einsum("ck,bkhw->bchw", m.fc.weight, a)
    assert torch.allclose(cam.mean(dim=(2, 3)) + m.fc.bias, logits, atol=1e-4)
