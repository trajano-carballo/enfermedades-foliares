"""Fixtures compartidas: un PlantVillage sintético en un directorio temporal (imágenes chicas + CSV de splits)."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image


def crear_pv_sintetico(raiz, n_train=12, n_val=3, n_test=3, n_excl=2):
    """Devuelve (splits_dir, data_root). Clases A y B de evaluación, C fuera de la evaluación; filas de train,
    val, test y excluida; todo bajo raw/plantvillage/raw/color/<clase>/."""
    data_root, splits = raiz / "data", raiz / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    gen = np.random.default_rng(0)
    filas = []

    def agregar(clase, evaluacion, particion, k):
        for i in range(k):
            ruta = f"raw/plantvillage/raw/color/{clase}/{particion}_{i}.JPG"
            (data_root / ruta).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(gen.integers(0, 255, (40, 30, 3), dtype=np.uint8)).save(data_root / ruta)
            filas.append({"ruta_color": ruta, "clase": clase, "clase_eval": str(evaluacion), "particion": particion})

    for clase in ("Clase___A", "Clase___B"):
        for particion, k in (("train", n_train), ("val", n_val), ("test", n_test), ("excluida", n_excl)):
            agregar(clase, True, particion, k)
    agregar("Clase___C", False, "train", 4)          # fuera de las 12 clases: no puede entrar al lote
    pd.DataFrame(filas).to_csv(splits / "plantvillage_particion.csv", index=False, lineterminator="\n")
    return splits, data_root


@pytest.fixture
def pv_sintetico(tmp_path):
    return crear_pv_sintetico(tmp_path)
