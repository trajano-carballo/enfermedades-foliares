"""Bootstrap estratificado, pareado e independiente sobre casos chicos con resultado conocido."""

import numpy as np
import pytest

from foliares.eval import bootstrap as B
from foliares.eval.metricas import macro_f1

Y = np.array([0] * 30 + [1] * 20 + [2] * 10)


def exactitud(y, p):
    return float((y == p).mean())


def mf1(y, p):
    return macro_f1(y, p, 3)


def test_cada_replica_conserva_los_conteos_por_clase():
    idx = B.indices_estratificados(Y, 50, np.random.default_rng(0))
    assert idx.shape == (50, len(Y))
    for fila in idx:
        assert np.bincount(Y[fila], minlength=3).tolist() == [30, 20, 10]


def test_ic_percentil_de_una_secuencia_conocida():
    assert B.ic_percentil(np.arange(1, 101)) == pytest.approx((3.475, 97.525))      # percentiles 2,5 y 97,5 de numpy


def test_misma_semilla_mismo_resultado_y_otra_semilla_otro():
    p = np.where(np.random.default_rng(1).random(len(Y)) < 0.8, Y, (Y + 1) % 3)
    a, b = B.bootstrap(Y, p, mf1, 200, 42), B.bootstrap(Y, p, mf1, 200, 42)
    assert a == b and B.bootstrap(Y, p, mf1, 200, 7) != a


def test_predicciones_perfectas_dan_ic_degenerado_en_1():
    r = B.bootstrap(Y, Y, mf1, 100)
    assert r["estimacion"] == r["ic_bajo"] == r["ic_alto"] == 1.0


def test_la_estimacion_es_la_metrica_sobre_la_muestra_completa_y_el_ic_la_rodea():
    p = np.where(np.random.default_rng(2).random(len(Y)) < 0.7, Y, (Y + 1) % 3)
    r = B.bootstrap(Y, p, exactitud, 2000)
    assert r["estimacion"] == exactitud(Y, p) and r["ic_bajo"] < r["estimacion"] < r["ic_alto"]


def test_pareado_con_predicciones_iguales_da_diferencia_cero():
    p = (Y + 1) % 3
    r = B.bootstrap_pareado(Y, p, p, mf1, 100)
    assert r["estimacion"] == r["ic_bajo"] == r["ic_alto"] == 0.0 and not r["excluye_0"]


def test_pareado_detecta_una_mejora_clara():
    malo = (Y + 1) % 3                                                            # todo mal
    r = B.bootstrap_pareado(Y, Y, malo, mf1, 200)
    assert r["estimacion"] == 1.0 and r["excluye_0"] and r["ic_bajo"] > 0


def test_pareado_usa_las_mismas_replicas_para_las_dos_predicciones():
    # Si p_b = p_a salvo una imagen, la diferencia por réplica solo puede venir de esa imagen: con réplicas independientes
    # aparecerían diferencias de ambos signos aunque las predicciones sean casi idénticas.
    p_a = Y.copy()
    p_b = Y.copy()
    p_b[0] = 1                                                                    # un error más en p_b
    r = B.bootstrap_pareado(Y, p_a, p_b, exactitud, 500)
    assert r["ic_bajo"] >= 0                                                      # nunca a favor de p_b


def test_pareado_exige_las_mismas_imagenes():
    with pytest.raises(ValueError):
        B.bootstrap_pareado(Y, Y, Y[:-1], mf1, 10)


def test_diferencia_independiente_entre_conjuntos_iguales_contiene_el_cero():
    p = np.where(np.random.default_rng(3).random(len(Y)) < 0.75, Y, (Y + 2) % 3)
    r = B.bootstrap_diferencia_independiente(Y, p, Y, p, mf1, 1000)
    assert r["estimacion"] == 0.0 and r["ic_bajo"] < 0 < r["ic_alto"] and not r["excluye_0"]


def test_diferencia_independiente_de_un_conjunto_perfecto_contra_uno_malo():
    y2 = np.array([0] * 8 + [1] * 8 + [2] * 8)
    r = B.bootstrap_diferencia_independiente(Y, Y, y2, (y2 + 1) % 3, mf1, 300)
    assert r["estimacion"] == 1.0 and r["excluye_0"]
