"""Escritura de salidas de texto con fin de línea LF, igual en Windows, Linux y Colab.

Los CSV, `.meta.json` y demás archivos que escribe el código se comparan por sha256 entre PC
(y se versionan con `eol=lf`, ver `.gitattributes`): si cada sistema escribiera su propio fin de
línea, el mismo contenido daría hashes distintos. Todo lo que el código escribe pasa por acá.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def escribir_texto(ruta: str | Path, texto: str, encoding: str = "utf-8") -> Path:
    """Escribe `texto` sin traducir `\n` (en Windows `write_text` por defecto escribiría CRLF)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding=encoding, newline="\n")
    return ruta


def escribir_csv(df: pd.DataFrame, ruta: str | Path, bom: bool = False, **kwargs) -> Path:
    """`df.to_csv(index=False)` con LF. `bom=True` agrega el BOM de UTF-8 (para que Excel abra bien las tildes)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("index", False)
    df.to_csv(ruta, lineterminator="\n", encoding="utf-8-sig" if bom else "utf-8", **kwargs)
    return ruta


def escribir_json(ruta: str | Path, objeto, **kwargs) -> Path:
    """JSON con sangría de 2 espacios, sin escapar tildes, con LF."""
    kwargs.setdefault("indent", 2)
    kwargs.setdefault("ensure_ascii", False)
    return escribir_texto(ruta, json.dumps(objeto, **kwargs))
