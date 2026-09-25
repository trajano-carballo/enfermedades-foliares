"""Reproduce la lógica real de `_generate_examples` en el loader oficial de
PlantVillage (data/raw/plantvillage/plant_village.py, commit fijado en
docs/bitacora/decisiones.md) para calcular el leaf_id de cada imagen a partir de
leaf_grouping/leaf-map.json.

Uso: verificación formal de cobertura de agrupamiento de hoja por cultivo
(tarea 3 de la sesión 2 — evidencia para la bitácora, no decide nada por sí sola).
"""

from __future__ import annotations

import json
from pathlib import Path


def calcular_leaf_id(file_name: str, class_name: str, leaf_map: dict) -> str:
    """Replica exacta de la lógica de leaf_id en plant_village.py::_generate_examples."""
    image_identifier = file_name.replace("_final_masked", "")
    if "___" in image_identifier:
        image_identifier = image_identifier.split("___")[-1]

    image_identifier = image_identifier.split("copy")[0]
    image_identifier = (
        image_identifier.replace(".jpg", "")
        .replace(".JPG", "")
        .replace(".png", "")
        .replace(".PNG", "")
    )
    image_identifier = image_identifier.strip()

    lookup_key = image_identifier.lower().strip()

    if lookup_key in leaf_map:
        suggestions = leaf_map[lookup_key]
        if len(suggestions) == 1:
            return suggestions[0]
        for suggestion in suggestions:
            if class_name in suggestion:
                return suggestion
        return f"fallback_{image_identifier}"
    return f"fallback_{image_identifier}"


def cargar_leaf_map(leaf_map_path: Path) -> dict:
    with open(leaf_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cobertura_por_clase(
    directorio_color: Path, leaf_map: dict, clases: list[str]
) -> dict[str, dict]:
    """Para cada clase en `clases`, calcula cuántas imágenes en
    `directorio_color/<clase>/` obtienen un leaf_id real (mapeado) vs. un
    leaf_id 'fallback_...' (no encontrado o ambiguo sin resolver) según la
    lógica oficial del loader.

    Devuelve {clase: {"n": int, "mapeadas": int, "fallback": int, "cobertura_pct": float}}.
    """
    directorio_color = Path(directorio_color)
    resultado: dict[str, dict] = {}
    for clase in clases:
        clase_dir = directorio_color / clase
        n = mapeadas = fallback = 0
        if not clase_dir.is_dir():
            resultado[clase] = {"n": 0, "mapeadas": 0, "fallback": 0, "cobertura_pct": float("nan")}
            continue
        for f in clase_dir.iterdir():
            if not f.is_file():
                continue
            n += 1
            leaf_id = calcular_leaf_id(f.name, clase, leaf_map)
            if leaf_id.startswith("fallback_"):
                fallback += 1
            else:
                mapeadas += 1
        cobertura = (100.0 * mapeadas / n) if n else float("nan")
        resultado[clase] = {"n": n, "mapeadas": mapeadas, "fallback": fallback, "cobertura_pct": cobertura}
    return resultado


def cobertura_por_cultivo(
    directorio_color: Path, leaf_map: dict, clases_por_cultivo: dict[str, list[str]]
) -> dict[str, dict]:
    """Agrega cobertura_por_clase a nivel de cultivo (ej. 'maiz', 'tomate', 'papa', 'pimiento')."""
    out: dict[str, dict] = {}
    for cultivo, clases in clases_por_cultivo.items():
        por_clase = cobertura_por_clase(directorio_color, leaf_map, clases)
        n = sum(v["n"] for v in por_clase.values())
        mapeadas = sum(v["mapeadas"] for v in por_clase.values())
        out[cultivo] = {
            "n": n,
            "mapeadas": mapeadas,
            "fallback": n - mapeadas,
            "cobertura_pct": (100.0 * mapeadas / n) if n else float("nan"),
            "por_clase": por_clase,
        }
    return out
