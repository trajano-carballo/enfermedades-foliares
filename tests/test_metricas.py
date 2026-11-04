"""Métricas sobre casos chicos calculados a mano."""

import numpy as np
import pytest

from foliares.eval import metricas as M

#        clase verdadera: 0 0 0 1 1 2
Y = np.array([0, 0, 0, 1, 1, 2])
P = np.array([0, 0, 1, 1, 2, 2])


def test_matriz_de_confusion_filas_verdadera_columnas_predicha():
    assert M.matriz_confusion(Y, P, 3).tolist() == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]


def test_recall_precision_f1_por_clase():
    m = M.matriz_confusion(Y, P, 3)
    assert np.allclose(M.recall_por_clase(m), [2 / 3, 1 / 2, 1])
    assert np.allclose(M.precision_por_clase(m), [1, 1 / 2, 1 / 2])
    assert np.allclose(M.f1_por_clase(m), [0.8, 0.5, 2 / 3])


def test_macro_f1_y_clase_sin_predicciones_vale_cero():
    assert M.macro_f1(Y, P, 3) == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert M.macro_f1([0, 1], [0, 0], 2) == pytest.approx((2 / 3 + 0) / 2)     # clase 1 sin aciertos: F1 = 0


def test_macro_f1_sin_clase_en_los_dos_modos():
    # Una imagen de la clase 2 predicha como 0: baja la precisión de la clase 0 solo si esa imagen se queda.
    y, p = [0, 1, 2, 2], [0, 1, 0, 2]
    # promedio sin la clase 2, con todas las imágenes: F1 clase 0 = 2/3 (P = 1/2, R = 1), clase 1 = 1
    assert M.macro_f1_sin_clase(y, p, 3, 2, "promedio_sin_la_clase") == pytest.approx((2 / 3 + 1) / 2)
    # sin las imágenes de la clase 2: y = [0, 1], p = [0, 1] -> F1 = 1 en las dos
    assert M.macro_f1_sin_clase(y, p, 3, 2, "sin_imagenes_de_la_clase") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        M.macro_f1_sin_clase(y, p, 3, 2, "otro")


def test_clase_mayoritaria_y_desempate():
    assert M.clase_mayoritaria([0, 1, 1, 2], 3, "error") == 1
    with pytest.raises(ValueError, match="Empate"):
        M.clase_mayoritaria([0, 0, 1, 1], 3, "error")
    assert M.clase_mayoritaria([0, 0, 1, 1], 3, "primera_en_orden") == 0
    with pytest.raises(ValueError, match="Desempate desconocido"):
        M.clase_mayoritaria([0, 0, 1, 1], 3, "al_azar")


def test_trivial_predice_siempre_la_mayoritaria():
    clase = M.clase_mayoritaria([2, 2, 2, 0, 1], 3, "error")
    r = M.resumen(Y, np.full_like(Y, clase), ["a", "b", "c"])
    assert r["recall_por_clase"] == {"a": 0.0, "b": 0.0, "c": 1.0} and r["macro_f1"] == pytest.approx((2 / 7) / 3)


def test_resumen_trae_todo():
    r = M.resumen(Y, P, ["a", "b", "c"], 2, "promedio_sin_la_clase")
    assert r["n"] == 6 and r["exactitud"] == pytest.approx(4 / 6) and r["macro_f1_sin_c"] == pytest.approx(0.65)
    assert r["matriz_confusion"]["filas_verdadera_columnas_predicha"] == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    assert r["soporte_por_clase"] == {"a": 3, "b": 2, "c": 1}
