"""Tests sintéticos de la lógica de duplicados de PlantDoc (sin imágenes)."""

import numpy as np
import pandas as pd

from foliares.data import plantdoc_chequeos as pc


def _meta():
    return pd.DataFrame({
        "split": ["train", "train", "test", "train", "train", "train"],
        "clase": ["A", "A", "A", "B", "B", "A"],
        "ruta": [f"r{k}" for k in range(6)],
        "nombre_original": [f"n{k}" for k in range(6)],
    })


def _pares(meta, lista, dist=0):
    filas = []
    for i, j in lista:
        a, b = meta.iloc[i], meta.iloc[j]
        filas.append({"i": i, "j": j, "categoria": pc.categoria(a, b), "distancia": dist})
    return pd.DataFrame(filas)


def test_categorias():
    m = _meta()
    assert pc.categoria(m.iloc[0], m.iloc[2]) == pc.CATEGORIAS[0]
    assert pc.categoria(m.iloc[0], m.iloc[1]) == pc.CATEGORIAS[1]
    assert pc.categoria(m.iloc[0], m.iloc[3]) == pc.CATEGORIAS[3]


def test_componentes_encadena_pares():
    comp = pc.componentes(5, [(0, 1), (1, 2), (3, 4)])
    assert comp[0] == comp[1] == comp[2] and comp[3] == comp[4] and comp[0] != comp[3]


def test_nunca_se_saca_de_test():
    m = _meta()
    p = _pares(m, [(0, 2), (0, 1)])           # train0 = test2 = train1 (misma clase)
    sacar = pc.sacar_de_dev(m, p)
    assert sacar == {0, 1} and all(m.split.iloc[k] == "train" for k in sacar)


def test_train_train_conserva_una():
    m = _meta()
    sacar = pc.sacar_de_dev(m, _pares(m, [(0, 1)]))
    assert sacar == {1}                        # se conserva la de ruta menor


def test_clases_distintas_solo_si_se_pide_y_saca_todas():
    m = _meta()
    p = _pares(m, [(0, 3)])
    assert pc.sacar_de_dev(m, p) == set()
    assert pc.sacar_de_dev(m, p, incluir_clases_distintas=True) == {0, 3}


def test_conteos():
    m = _meta()
    t = pc.conteos_dev_test(m, {1}).set_index("clase")
    assert t.loc["A", "dev_antes"] == 3 and t.loc["A", "dev_despues"] == 2 and t.loc["A", "test"] == 1


def test_distancias_detecta_variante_rotada():
    h = np.array([[5, 7, 9, 11, 13, 15, 17, 19], [7, 5, 9, 11, 13, 15, 17, 19]], dtype=np.uint64)
    D, var = pc.distancias(h)
    assert D[0, 1] == 0 and D[1, 0] == 0
