"""Grillas de miniaturas para inspección visual (sesión 3). Solo imágenes, sin modelo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def grilla(items: list[tuple[Path, str]], ncols: int = 6, lado: float = 1.6, titulo: str = "",
           guardar: Path | None = None):
    """items = [(ruta, rótulo)]. Muestra (y opcionalmente guarda) la grilla."""
    n = len(items)
    nrows = max(1, -(-n // ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * lado, nrows * (lado + 0.25)), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (ruta, rot) in zip(axs.ravel(), items):
        try:
            with Image.open(ruta) as im:
                im = im.convert("RGB")
                im.thumbnail((160, 160))
                ax.imshow(im)
        except Exception:  # noqa: BLE001
            ax.text(0.5, 0.5, "ilegible", ha="center")
        ax.set_title(rot, fontsize=6)
    fig.suptitle(titulo, fontsize=9)
    fig.tight_layout()
    if guardar is not None:
        Path(guardar).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(guardar, dpi=70)
    return fig
