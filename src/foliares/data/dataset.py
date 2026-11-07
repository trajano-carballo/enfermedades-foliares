"""Dataset sobre las particiones congeladas de `data/splits/`, con las guardas de los invariantes 1 y 2.

- Lee un CSV de `data/splits/` y una partición; las rutas se resuelven contra DATA_ROOT.
- PlantVillage: cada partición tiene un solo rol (train -> entrenamiento, val -> validacion, test -> evaluacion); filtra a
  las 12 clases de `clase_eval` y a la partición pedida; `excluida` y las filas sin partición no se
  pueden pedir. El CSV trae 15 clases: las 3 fuera de alcance tienen train/val/test y se descartan acá, explícitamente.
- PlantDoc: SOLO en el rol `evaluacion` (invariante 1: nunca en entrenamiento ni en validación/early stopping); solo filas
  `conservada` con `particion_propuesta` dev o test.
- Las clases que quedan tienen que ser exactamente las de la config (orden = índice); una clase fuera de alcance es un error.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import Dataset

from foliares.data.preproceso import preprocesar

ROLES = ("entrenamiento", "validacion", "evaluacion")
ROL_DE_PARTICION_PV = {"train": "entrenamiento", "val": "validacion", "test": "evaluacion"}
ARCHIVOS = {"plantvillage": "plantvillage_particion.csv", "plantdoc": "plantdoc_particion.csv"}


class ParticionInvalida(ValueError):
    """Pedido que rompe un invariante o no corresponde a una partición utilizable."""


def filas_particion(splits_dir: str | Path, dataset: str, particion: str, rol: str, clases_pv: list[str],
                    clases_pd: list[str] | None = None) -> pd.DataFrame:
    """Filas (`ruta`, `clase`, `etiqueta`) de una partición, ya filtradas y con la etiqueta = índice en `clases_pv`."""
    if rol not in ROLES:
        raise ParticionInvalida(f"Rol desconocido: {rol!r}; opciones: {ROLES}")
    if dataset not in ARCHIVOS:
        raise ParticionInvalida(f"Dataset desconocido: {dataset!r}; opciones: {list(ARCHIVOS)}")
    if dataset == "plantdoc" and rol != "evaluacion":
        raise ParticionInvalida(f"PlantDoc no puede usarse en el rol {rol!r}: nunca entra al entrenamiento ni a la "
                                "validación/early stopping de PlantVillage (invariante 1 de CLAUDE.md)")
    if particion in ("", "excluida"):
        raise ParticionInvalida(f"La partición {particion!r} no se puede usar: las filas excluidas o sin partición se descartan")
    df = pd.read_csv(Path(splits_dir) / ARCHIVOS[dataset], dtype=str, keep_default_na=False)
    if dataset == "plantvillage":
        if particion not in ("train", "val", "test"):
            raise ParticionInvalida(f"PlantVillage no tiene la partición {particion!r} (train, val o test)")
        if rol != ROL_DE_PARTICION_PV[particion]:
            raise ParticionInvalida(f"PlantVillage-{particion} solo puede usarse en el rol {ROL_DE_PARTICION_PV[particion]!r}, "
                                    f"no en {rol!r} (train entrena, val valida y elige el checkpoint, test solo evalúa)")
        df = df[(df.clase_eval.str.lower() == "true") & (df.particion == particion)]
        filas = pd.DataFrame({"ruta": df.ruta_color, "clase": df.clase})
        nombres = list(clases_pv)
    else:
        if particion not in ("dev", "test"):
            raise ParticionInvalida(f"PlantDoc no tiene la partición {particion!r} (dev o test)")
        if clases_pd is None or len(clases_pd) != len(clases_pv):
            raise ParticionInvalida("Para PlantDoc hace falta el mapeo de clases (`plantdoc` de cada clase en la config)")
        df = df[(df.estado == "conservada") & (df.particion_propuesta == particion)]
        filas = pd.DataFrame({"ruta": df.ruta, "clase": df.clase})
        nombres = list(clases_pd)
    presentes = set(filas.clase)
    fuera = sorted(presentes - set(nombres))
    if fuera:
        raise ParticionInvalida(f"Clases fuera de alcance en {dataset}/{particion}: {fuera} (no están en la config)")
    faltan = sorted(set(nombres) - presentes)
    if faltan:
        raise ParticionInvalida(f"Clases de la config sin imágenes en {dataset}/{particion}: {faltan}")
    indice = {c: i for i, c in enumerate(nombres)}
    filas = filas.sort_values("ruta").reset_index(drop=True)
    return filas.assign(etiqueta=filas.clase.map(indice).astype("int64"))


class DatasetParticion(Dataset):
    """Imágenes de una partición congelada, preprocesadas con la geometría de la config. Devuelve (x, etiqueta)."""

    def __init__(self, splits_dir: str | Path, data_root: str | Path, dataset: str, particion: str, rol: str,
                 clases_pv: list[str], clases_pd: list[str] | None = None, geometria: str = "aplastar", tamano: int = 224,
                 preproceso: Callable | None = None):
        self.filas = filas_particion(splits_dir, dataset, particion, rol, clases_pv, clases_pd)
        self.data_root = Path(data_root)
        self.dataset, self.particion, self.rol = dataset, particion, rol
        self._pre = preproceso or partial(preprocesar, geometria=geometria, tamano=tamano)   # serializable (workers)

    def __len__(self) -> int:
        return len(self.filas)

    def __getitem__(self, i: int):
        f = self.filas.iloc[i]
        return self._pre(self.data_root / f.ruta), torch.tensor(f.etiqueta, dtype=torch.int64)

    @property
    def rutas(self) -> list[str]:
        return list(self.filas.ruta)

    @property
    def etiquetas(self) -> torch.Tensor:
        return torch.tensor(self.filas.etiqueta.to_numpy(), dtype=torch.int64)
