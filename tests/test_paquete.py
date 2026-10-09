"""Empaquetado y verificación sobre un árbol sintético en tmp_path (no toca los datasets reales)."""

import tarfile

import pandas as pd
import pytest

from foliares.data import paquete


@pytest.fixture
def arbol(tmp_path):
    root = tmp_path / "root"
    for r, contenido in {"raw/ds/a/1.jpg": b"uno", "raw/ds/a/2 x.jpg": b"dos dos", "raw/ds/seg/1_m.jpg": b"s"}.items():
        (root / r).parent.mkdir(parents=True, exist_ok=True)
        (root / r).write_bytes(contenido)
    return root, ["raw/ds/a/1.jpg", "raw/ds/a/2 x.jpg", "raw/ds/seg/1_m.jpg"]


def test_empaquetar_nombres_manifiesto_y_verificar(arbol, tmp_path):
    root, rutas = arbol
    tar = tmp_path / "out" / "ds_subconjunto.tar"
    r = paquete.empaquetar(rutas, root, tar, "ds")
    assert r["archivos"] == 3 and paquete.verificar_tar(tar)
    with tarfile.open(tar) as tf:
        nombres = tf.getnames()
        assert nombres[0] == "MANIFIESTO_ds.csv" and set(nombres[1:]) == set(rutas)   # nombre = ruta
        destino = tmp_path / "colab"
        tf.extractall(destino, filter="data")
    m = pd.read_csv(destino / "MANIFIESTO_ds.csv")
    assert list(m.columns) == ["ruta", "sha256", "bytes"] and len(m) == 3
    assert paquete.verificar(destino, destino / "MANIFIESTO_ds.csv")["ok"]


def test_verificar_detecta_faltante_tamano_y_hash(arbol, tmp_path):
    root, rutas = arbol
    tar = tmp_path / "t.tar"
    paquete.empaquetar(rutas, root, tar, "ds")
    with tarfile.open(tar) as tf:
        tf.extractall(tmp_path / "x", filter="data")
    x = tmp_path / "x"
    (x / rutas[0]).unlink()
    (x / rutas[1]).write_bytes(b"otro largo")
    (x / rutas[2]).write_bytes(b"z")                      # mismo tamaño, distinto contenido
    r = paquete.verificar(x, x / "MANIFIESTO_ds.csv")
    assert not r["ok"] and r["faltan"] == [rutas[0]] and r["tamano_distinto"] == [rutas[1]] and r["sha256_distinto"] == [rutas[2]]
    assert paquete.verificar(x, x / "MANIFIESTO_ds.csv", rapido=True)["sha256_distinto"] == []


def test_tar_alterado_no_verifica(arbol, tmp_path):
    root, rutas = arbol
    tar = tmp_path / "t.tar"
    paquete.empaquetar(rutas, root, tar, "ds")
    with open(tar, "ab") as f:
        f.write(b"x")
    assert not paquete.verificar_tar(tar)


def test_listas_por_clase_eval():
    m = pd.DataFrame({"clase_eval": [True, False], "ruta_color": ["c1", "c2"], "ruta_segmented": ["s1", "s2"]})
    assert paquete.lista_plantvillage(m) == ["c1", "s1"]
    d = pd.DataFrame({"estado": ["conservada", "excluida_duplicada", "fuera_de_alcance"], "ruta": ["a", "b", "c"]})
    assert paquete.lista_plantdoc(d) == ["a", "b"]
