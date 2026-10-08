"""Manifiesto de PlantVillage (clases candidatas: tomate, papa, pimiento).

Una fila por imagen de la carpeta color, con su par segmented, sesión, hoja y grupo.
Las rutas se guardan relativas a DATA_ROOT (posix) para que sirvan en cualquier máquina.
No decide ninguna partición.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from foliares.data import grupos_hoja as gh
from foliares.data.taxonomia import PAR_CANDIDATO_PLANTDOC, clases_plantvillage_de_cultivo, es_clase_eval_pv
from foliares.utils.paths import Rutas

SUFIJO_SEGMENTED = "_final_masked.jpg"
SIN_SESION = "(sin_sesion)"
COLUMNAS = ["ruta_color", "ruta_segmented", "segmented_existe", "segmented_irregular", "clase", "cultivo", "clase_comun", "clase_eval",
            "clase_plantdoc_candidata", "sesion", "num", "leaf_id", "tiene_grupo", "uuid", "archivo"]


def clases_candidatas() -> list[str]:
    return [c for k in ("tomate", "papa", "pimiento") for c in clases_plantvillage_de_cultivo(k)]


def nombre_segmented(archivo_color: str) -> str:
    """Convención verificada: `<stem de color>_final_masked.jpg` (la extensión de color
    puede ser .JPG/.jpg; la de segmented es siempre .jpg)."""
    return Path(archivo_color).stem + SUFIJO_SEGMENTED


def construir_manifiesto(rutas: Rutas) -> tuple[pd.DataFrame, dict]:
    """Devuelve (manifiesto, control). `control` trae el chequeo del par color/segmented."""
    color_dir = rutas.plantvillage / "raw" / "color"
    seg_dir = rutas.plantvillage / "raw" / "segmented"
    leafmaps = rutas.plantvillage / "leaf_grouping" / "filtered_leafmaps"
    clases = clases_candidatas()
    df = gh.tabla_imagenes(color_dir, leafmaps, clases)

    seg_por_clase = {c: {f.name for f in (seg_dir / c).iterdir()} for c in clases}
    df["segmented"] = [nombre_segmented(a) for a in df.archivo]
    df["segmented_existe"] = [s in seg_por_clase[c] for s, c in zip(df.segmented, df.clase)]
    # Excepción conocida: algún segmented perdió el prefijo `uuid___`. Se enlaza solo si el
    # nombre sin uuid existe en esa clase, y queda marcado como irregular.
    df["segmented_irregular"] = False
    for i in df.index[~df.segmented_existe]:
        alt = gh.sin_uuid(df.at[i, "segmented"])
        if alt in seg_por_clase[df.at[i, "clase"]]:
            df.at[i, "segmented"], df.at[i, "segmented_existe"], df.at[i, "segmented_irregular"] = alt, True, True

    base_pv = rutas.relativa(rutas.plantvillage)
    df["ruta_color"] = [f"{base_pv}/raw/color/{c}/{a}" for c, a in zip(df.clase, df.archivo)]
    df["ruta_segmented"] = [f"{base_pv}/raw/segmented/{c}/{s}" if ok else ""
                            for c, s, ok in zip(df.clase, df.segmented, df.segmented_existe)]
    df["cultivo"] = df.clase.str.split("___").str[0]
    df["clase_plantdoc_candidata"] = df.clase.map(PAR_CANDIDATO_PLANTDOC).fillna("")
    df["clase_comun"] = df.clase_plantdoc_candidata != ""
    df["clase_eval"] = df.clase.map(es_clase_eval_pv)
    df["sesion"] = df.sesion.fillna(SIN_SESION)
    df["leaf_id"] = df.hoja.fillna("")
    df["tiene_grupo"] = df.con_grupo
    df["uuid"] = df.archivo.str.extract(r"^(.*?)___", expand=False).fillna("")

    # Control del par: color sin segmented y segmented sin color, por clase.
    filas = []
    for c in clases:
        enlazados = set(df.segmented[(df.clase == c) & df.segmented_existe])
        filas.append({"clase": c, "color": int((df.clase == c).sum()), "segmented_en_disco": len(seg_por_clase[c]),
                      "par_irregular": int(((df.clase == c) & df.segmented_irregular).sum()),
                      "color_sin_par": int(((df.clase == c) & ~df.segmented_existe).sum()),
                      "segmented_sin_color": len(seg_por_clase[c] - enlazados)})
    control = pd.DataFrame(filas)
    return df[COLUMNAS].reset_index(drop=True), {"par_color_segmented": control}
