"""Conteo e integridad de imágenes por clase.

Funciones puras de inventario: no entrenan nada, no calculan métricas de
desempeño. Pensadas para orquestarse desde notebooks/00_exploracion_datasets.ipynb
(etapa 0, exploratorio, nada congelado).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def contar_por_clase(directorio: Path) -> dict[str, int]:
    """Cuenta archivos por subcarpeta de primer nivel (una subcarpeta = una clase)."""
    directorio = Path(directorio)
    conteos: dict[str, int] = {}
    for clase_dir in sorted(p for p in directorio.iterdir() if p.is_dir()):
        conteos[clase_dir.name] = sum(
            1 for f in clase_dir.iterdir() if f.is_file() and f.suffix in EXTENSIONES_IMAGEN
        )
    return conteos


@dataclass
class HallazgoIntegridad:
    ruta: str
    clase: str
    tipo: str  # "ilegible" | "resolucion_atipica" | "escala_de_grises_en_color"
    detalle: str = ""


@dataclass
class ResumenIntegridad:
    n_archivos: int = 0
    hallazgos: list[HallazgoIntegridad] = field(default_factory=list)
    resoluciones: dict[tuple[int, int], int] = field(default_factory=dict)

    def contar_por_tipo(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for h in self.hallazgos:
            out[h.tipo] = out.get(h.tipo, 0) + 1
        return out


def _es_probablemente_gris(img: Image.Image, tolerancia: int = 4) -> bool:
    """Heurística vectorizada: convierte a RGB y mide la máxima diferencia entre
    canales en todos los píxeles. Una imagen a color guardada como RGB pero sin
    variación de canal (por debajo de `tolerancia`) se marca como sospechosa de
    estar en escala de grises."""
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    diff = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
    return bool(diff.max() <= tolerancia)


def verificar_integridad(
    directorio: Path,
    resoluciones_esperadas: set[tuple[int, int]] | None = None,
    chequear_escala_de_grises: bool = False,
) -> ResumenIntegridad:
    """Recorre `directorio` (una subcarpeta por clase) y reporta:
    - archivos ilegibles (no abren con PIL o están truncados),
    - resoluciones fuera de `resoluciones_esperadas` (si se pasa; si no, solo
      se tabulan todas las resoluciones vistas para que el equipo las revise),
    - (opcional) imágenes en escala de grises dentro de un directorio "color".
    """
    directorio = Path(directorio)
    resumen = ResumenIntegridad()

    for clase_dir in sorted(p for p in directorio.iterdir() if p.is_dir()):
        clase = clase_dir.name
        for f in sorted(clase_dir.iterdir()):
            if not (f.is_file() and f.suffix in EXTENSIONES_IMAGEN):
                continue
            resumen.n_archivos += 1
            try:
                with Image.open(f) as img:
                    img.verify()
                with Image.open(f) as img:
                    tam = img.size
                    modo = img.mode
                    resumen.resoluciones[tam] = resumen.resoluciones.get(tam, 0) + 1
                    if resoluciones_esperadas is not None and tam not in resoluciones_esperadas:
                        resumen.hallazgos.append(
                            HallazgoIntegridad(str(f), clase, "resolucion_atipica", f"{tam[0]}x{tam[1]}")
                        )
                    if chequear_escala_de_grises:
                        if modo == "L":
                            resumen.hallazgos.append(
                                HallazgoIntegridad(str(f), clase, "escala_de_grises_en_color", "modo=L")
                            )
                        elif modo == "RGB" and _es_probablemente_gris(img):
                            resumen.hallazgos.append(
                                HallazgoIntegridad(str(f), clase, "escala_de_grises_en_color", "RGB con canales iguales")
                            )
            except Exception as e:  # noqa: BLE001 - se reporta cualquier fallo de lectura
                resumen.hallazgos.append(HallazgoIntegridad(str(f), clase, "ilegible", str(e)))

    return resumen


def contar_por_clase_split(directorio_split: Path) -> dict[str, int]:
    """Alias semántico de contar_por_clase para un split (train/ o test/) de PlantDoc."""
    return contar_por_clase(directorio_split)


def tamano_en_disco_mb(directorio: Path) -> float:
    directorio = Path(directorio)
    total = 0
    for root, _dirs, files in os.walk(directorio):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)
