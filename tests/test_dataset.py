"""Dataset sobre data/splits/: filtra a las clases de evaluación, descarta excluida/vacías y protege el invariante 1."""

import pandas as pd
import pytest
import torch

from foliares.data.dataset import DatasetParticion, ParticionInvalida, filas_particion

CLASES = ["Clase___A", "Clase___B"]


def test_filtra_clase_eval_y_particion(pv_sintetico):
    splits, _ = pv_sintetico
    f = filas_particion(splits, "plantvillage", "train", "entrenamiento", CLASES)
    assert len(f) == 24 and set(f.clase) == set(CLASES)               # Clase___C (fuera de clase_eval) no entra
    assert f.ruta.str.contains("/train_").all() and f.etiqueta.tolist() == [CLASES.index(c) for c in f.clase]


@pytest.mark.parametrize("particion", ["excluida", ""])
def test_excluida_y_vacia_no_se_pueden_pedir(pv_sintetico, particion):
    splits, _ = pv_sintetico
    with pytest.raises(ParticionInvalida):
        filas_particion(splits, "plantvillage", particion, "entrenamiento", CLASES)


def test_falla_si_entra_una_clase_fuera_de_alcance(pv_sintetico):
    splits, _ = pv_sintetico
    df = pd.read_csv(splits / "plantvillage_particion.csv", dtype=str, keep_default_na=False)
    df.loc[df.clase == "Clase___C", "clase_eval"] = "True"            # una clase fuera de alcance marcada como de evaluación
    df.to_csv(splits / "plantvillage_particion.csv", index=False)
    with pytest.raises(ParticionInvalida, match="fuera de alcance"):
        filas_particion(splits, "plantvillage", "train", "entrenamiento", CLASES)


def test_falla_si_a_la_config_le_sobra_una_clase(pv_sintetico):
    splits, _ = pv_sintetico
    with pytest.raises(ParticionInvalida, match="sin imágenes"):
        filas_particion(splits, "plantvillage", "train", "entrenamiento", CLASES + ["Clase___Z"])


@pytest.mark.parametrize("rol", ["entrenamiento", "validacion"])
def test_plantdoc_no_se_puede_usar_para_entrenar_ni_validar(pv_sintetico, rol):
    splits, _ = pv_sintetico
    with pytest.raises(ParticionInvalida, match="invariante 1"):
        filas_particion(splits, "plantdoc", "dev", rol, CLASES, ["Pd A", "Pd B"])


def test_rol_desconocido_falla(pv_sintetico):
    splits, _ = pv_sintetico
    with pytest.raises(ParticionInvalida, match="Rol desconocido"):
        filas_particion(splits, "plantvillage", "train", "ajuste", CLASES)


def test_items_y_etiquetas(pv_sintetico):
    splits, data_root = pv_sintetico
    d = DatasetParticion(splits, data_root, "plantvillage", "val", "validacion", CLASES, geometria="aplastar", tamano=32)
    x, y = d[0]
    assert len(d) == 6 and x.shape == (3, 32, 32) and y.dtype == torch.int64
    assert torch.equal(d.etiquetas, torch.tensor(d.filas.etiqueta.to_numpy()))


@pytest.mark.parametrize("particion,rol", [("train", "validacion"), ("train", "evaluacion"), ("val", "entrenamiento"),
                                            ("val", "evaluacion"), ("test", "entrenamiento"), ("test", "validacion")])
def test_plantvillage_cada_particion_solo_en_su_rol(pv_sintetico, particion, rol):
    # M1 (review de la sesión 8): train -> entrenamiento, val -> validacion, test -> evaluacion; cualquier otra combinación falla.
    splits, _ = pv_sintetico
    with pytest.raises(ParticionInvalida, match="solo puede usarse en el rol"):
        filas_particion(splits, "plantvillage", particion, rol, CLASES)


@pytest.mark.parametrize("particion,rol", [("train", "entrenamiento"), ("val", "validacion"), ("test", "evaluacion")])
def test_plantvillage_combinaciones_permitidas(pv_sintetico, particion, rol):
    splits, _ = pv_sintetico
    assert len(filas_particion(splits, "plantvillage", particion, rol, CLASES)) > 0
