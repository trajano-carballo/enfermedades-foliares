"""Guarda de invariantes: el lote de humo es solo PlantVillage-train de las clases de evaluación."""

import pandas as pd
import pytest
import torch

from foliares.data.lote_humo import cargar_lote, clases_eval, elegir_rutas_train


def test_solo_train_de_las_clases_de_evaluacion(pv_sintetico):
    splits, _ = pv_sintetico
    todas = elegir_rutas_train(splits, n=24, semilla=42)           # las 24 imágenes de train de A y B
    assert set(todas.particion) == {"train"} and set(todas.clase) == {"Clase___A", "Clase___B"}
    assert not todas.ruta_color.str.contains("val_|test_|excluida_|Clase___C").any()


def test_pedir_mas_de_las_que_hay_en_train_falla(pv_sintetico):
    splits, _ = pv_sintetico
    with pytest.raises(ValueError):
        elegir_rutas_train(splits, n=25, semilla=42)               # val y test no se suman para completar


def test_semilla_estable_y_distinta_cambia(pv_sintetico):
    splits, _ = pv_sintetico
    a = list(elegir_rutas_train(splits, 8, 42).ruta_color)
    assert a == list(elegir_rutas_train(splits, 8, 42).ruta_color) and a != list(elegir_rutas_train(splits, 8, 43).ruta_color)


def test_rechaza_rutas_que_no_son_de_plantvillage(pv_sintetico):
    splits, _ = pv_sintetico
    df = pd.read_csv(splits / "plantvillage_particion.csv", dtype=str, keep_default_na=False)
    df.loc[(df.particion == "train") & (df.clase == "Clase___A"), "ruta_color"] = "raw/plantdoc/train/Tomato leaf/x.jpg"
    df.to_csv(splits / "plantvillage_particion.csv", index=False)
    with pytest.raises(RuntimeError, match="solo puede ser de PlantVillage-train"):
        elegir_rutas_train(splits, n=24, semilla=42)


def test_forma_y_etiquetas(pv_sintetico):
    splits, data_root = pv_sintetico
    x, y, rutas = cargar_lote(splits, data_root, n=6, semilla=42, tamano=32)
    assert x.shape == (6, 3, 32, 32) and x.dtype == torch.float32 and y.dtype == torch.int64 and len(rutas) == 6
    assert clases_eval(splits) == ["Clase___A", "Clase___B"] and set(y.tolist()) <= {0, 1}
    assert all("/train_" in r for r in rutas)
