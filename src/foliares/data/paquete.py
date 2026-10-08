"""Empaquetado del subconjunto de trabajo (12 clases) en un .tar por dataset, con manifiesto sha256.

Los nombres dentro del .tar son la `ruta` de los manifiestos (relativa a DATA_ROOT, con los nombres
locales saneados de PlantDoc), así que al descomprimir en DATA_ROOT las rutas de los CSV valen tal cual,
también en Linux/Colab. Sin compresión (los JPG ya están comprimidos).
"""

from __future__ import annotations

import csv
import tarfile
from pathlib import Path

import pandas as pd

from foliares.data.congelar import sha256_archivo


def lista_plantvillage(manifiesto: pd.DataFrame) -> list[str]:
    """Color + segmented de las clases de evaluación (clase_eval)."""
    m = manifiesto[manifiesto.clase_eval]
    return sorted(set(m.ruta_color) | set(r for r in m.ruta_segmented.fillna("") if r))


def lista_plantdoc(depurado: pd.DataFrame) -> list[str]:
    """Todas las imágenes de las 12 clases (cualquier estado), para poder cambiar exclusiones sin re-empaquetar."""
    return sorted(depurado.ruta[depurado.estado != "fuera_de_alcance"])


def nombre_manifiesto(dataset: str) -> str:
    return f"MANIFIESTO_{dataset}.csv"


def empaquetar(rutas: list[str], data_root: Path, tar_path: Path, dataset: str) -> dict:
    """Escribe `tar_path` (con el manifiesto como primer miembro) y `tar_path.sha256`. Devuelve un resumen."""
    faltan = [r for r in rutas if not (data_root / r).is_file()]
    if faltan:
        raise FileNotFoundError(f"{len(faltan)} archivos de la lista no existen en {data_root} (p. ej. {faltan[:3]})")
    filas = [{"ruta": r, "sha256": sha256_archivo(data_root / r), "bytes": (data_root / r).stat().st_size} for r in rutas]
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    man = tar_path.with_suffix(".manifiesto.tmp.csv")
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ruta", "sha256", "bytes"])
        w.writeheader()
        w.writerows(filas)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tf:
        tf.add(man, arcname=nombre_manifiesto(dataset))
        for r in rutas:
            tf.add(data_root / r, arcname=r, recursive=False)
    man.unlink()
    suma = sha256_archivo(tar_path)
    tar_path.with_name(tar_path.name + ".sha256").write_text(f"{suma}  {tar_path.name}\n", encoding="utf-8")
    return {"archivos": len(filas), "bytes": sum(f["bytes"] for f in filas), "tar_sha256": suma}


def verificar_tar(tar_path: Path) -> bool:
    """Compara el sha256 del .tar contra su archivo `.sha256` (antes de descomprimir)."""
    esperado = tar_path.with_name(tar_path.name + ".sha256").read_text(encoding="utf-8").split()[0]
    return sha256_archivo(tar_path) == esperado


def verificar(data_root: Path, manifiesto_csv: Path, rapido: bool = False) -> dict:
    """Verifica cada archivo del manifiesto en `data_root`: existencia, tamaño y (salvo `rapido`) sha256."""
    m = pd.read_csv(manifiesto_csv)
    faltan, tam_mal, hash_mal = [], [], []
    for r in m.itertuples():
        p = data_root / r.ruta
        if not p.is_file():
            faltan.append(r.ruta)
        elif p.stat().st_size != r.bytes:
            tam_mal.append(r.ruta)
        elif not rapido and sha256_archivo(p) != r.sha256:
            hash_mal.append(r.ruta)
    return {"manifiesto": manifiesto_csv.name, "esperados": len(m), "faltan": faltan, "tamano_distinto": tam_mal,
            "sha256_distinto": hash_mal, "ok": not (faltan or tam_mal or hash_mal)}
