"""Sesión 12, fase 3: chequeo de duplicados y sha256 de las imágenes externas (solo hash)."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from foliares.data.duplicados import hashes_uint64 as hashes_reales
from foliares.data.externas import COLUMNAS, escribir_planilla, leer_planilla, verificar_externas

REPO = Path(__file__).resolve().parents[1]


def _foto(ruta, semilla, tam=(160, 120)):
    rng = np.random.default_rng(semilla)
    base = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)           # baja frecuencia: el phash la distingue de otra semilla
    Image.fromarray(base).resize(tam, Image.BICUBIC).save(ruta, quality=95)
    return ruta


@pytest.fixture
def mundo(tmp_path):
    ext = tmp_path / "ext"
    ext.mkdir()
    refs = tmp_path / "refs"
    refs.mkdir()
    ref_dup, ref_otra = _foto(refs / "a.jpg", 1), _foto(refs / "b.jpg", 2)
    _foto(ext / "nueva.jpg", 3)
    Image.open(ref_dup).transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True).save(ext / "copia_rotada.jpg", quality=90)
    filas = [{"archivo": n, "autor": "x", "licencia": "CC0", "url": "u", "sha256": "", "resultado_duplicados": ""}
             for n in ("nueva.jpg", "copia_rotada.jpg")]
    return ext, filas, {"conjunto": [ref_dup, ref_otra]}


def test_detecta_la_copia_rotada_y_espejada_y_no_la_imagen_nueva(mundo):
    ext, filas, refs = mundo
    r = verificar_externas(ext, filas, refs, umbral=6, base=ext.parent)
    assert r[0]["resultado_duplicados"].startswith("sin duplicados") and "distancia minima" in r[0]["resultado_duplicados"]
    dup = r[1]["resultado_duplicados"]
    assert dup.startswith("DUPLICADA") and "contra refs/a.jpg" in dup and dup.endswith("descartada")   # ruta relativa a `base`
    assert str(ext.parent) not in dup and ":" not in dup.split("contra")[1]


def test_completa_el_sha256_y_no_toca_el_resto(mundo):
    ext, filas, refs = mundo
    r = verificar_externas(ext, filas, refs)
    assert r[0]["sha256"] == hashlib.sha256((ext / "nueva.jpg").read_bytes()).hexdigest()
    assert [(f["archivo"], f["autor"], f["licencia"], f["url"]) for f in r] == [(f["archivo"], f["autor"], f["licencia"], f["url"]) for f in filas]


def test_una_imagen_que_falta_frena(mundo):
    ext, filas, refs = mundo
    with pytest.raises(FileNotFoundError):
        verificar_externas(ext, [*filas, {**filas[0], "archivo": "no_esta.jpg"}], refs)


def test_la_planilla_guarda_comas_y_parentesis_y_exige_las_columnas(tmp_path):
    p = tmp_path / "p.csv"
    f = {"archivo": "a.jpg", "autor": "Ana, Univ.", "licencia": "CC BY 3.0 US", "url": "https://x/File:A_(1).jpg", "sha256": "", "resultado_duplicados": ""}
    escribir_planilla(p, [f])
    assert leer_planilla(p) == [f]
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        leer_planilla(p)


def test_la_planilla_del_repo_tiene_las_columnas_y_los_datos_de_commons():
    filas = leer_planilla(REPO / "docs" / "imagenes_externas.csv")
    assert list(filas[0]) == COLUMNAS
    validas = [f for f in filas if not f["resultado_duplicados"].startswith("DUPLICADA")]   # las duplicadas quedan marcadas, descartadas
    assert 3 <= len(validas) <= 5 and all(f["sha256"] and f["resultado_duplicados"] for f in filas)
    assert all(f["autor"] and f["licencia"] and f["url"].startswith("https://commons.wikimedia.org/") for f in filas)


def test_cada_externa_se_compara_con_su_propio_phash_y_su_propia_referencia_mas_cercana(tmp_path):
    """Dos externas distintas, cada una copia (retocada) de una referencia distinta: el hash, la imagen más cercana y la distancia son de cada una."""
    ext, refs = tmp_path / "ext", tmp_path / "refs"
    ext.mkdir(), refs.mkdir()
    r = [_foto(refs / f"r{i}.jpg", 10 + i) for i in range(4)]
    for i, k in enumerate((1, 3)):
        Image.open(r[k]).save(ext / f"e{i}.jpg", quality=60)         # recompresión: distancia pequeña pero no necesariamente 0
    filas = [{"archivo": f"e{i}.jpg", "autor": "x", "licencia": "CC0", "url": "u", "sha256": "", "resultado_duplicados": ""} for i in range(2)]
    det = {}
    verificar_externas(ext, filas, {"c": r}, base=tmp_path, detalle=det)
    im = det["imagenes"]
    assert im[0]["phash_propio"] != im[1]["phash_propio"]
    assert [x["mas_cercana"]["c"]["ruta"] for x in im] == ["refs/r1.jpg", "refs/r3.jpg"]
    assert all(x["mas_cercana"]["c"]["distancia"] <= 6 for x in im)


def test_con_y_sin_cache_da_el_mismo_resultado_y_la_segunda_vez_no_recalcula(mundo, tmp_path, monkeypatch):
    from foliares.data import externas as E
    ext, filas, refs = mundo
    cache = tmp_path / "cache"
    sin, con1, con2 = {}, {}, {}
    r_sin = verificar_externas(ext, filas, refs, base=ext.parent, detalle=sin)
    r1 = verificar_externas(ext, filas, refs, base=ext.parent, cache=cache, detalle=con1)
    # `externas` también hashea las propias (con variantes): solo eso puede llamarse; los conjuntos salen del caché
    monkeypatch.setattr(E, "hashes_uint64", lambda rutas, con_variantes=False: pytest.fail("recalculó") if not con_variantes else hashes_reales(rutas, True))
    r2 = verificar_externas(ext, filas, refs, base=ext.parent, cache=cache, detalle=con2)
    assert r_sin == r1 == r2 and sin["imagenes"] == con1["imagenes"] == con2["imagenes"]
    assert sin["cache"] == {"conjunto": "sin_cache"} and con1["cache"] == {"conjunto": "miss"} and con2["cache"] == {"conjunto": "hit"}


def test_el_cache_se_invalida_si_cambia_un_archivo_de_referencia(mundo, tmp_path):
    ext, filas, refs = mundo
    cache = tmp_path / "cache"
    verificar_externas(ext, filas, refs, cache=cache)
    _foto(refs["conjunto"][1], 2, tam=(200, 150))                    # otro tamaño de archivo: otra clave
    d = {}
    verificar_externas(ext, filas, refs, cache=cache, detalle=d)
    assert d["cache"] == {"conjunto": "miss"}
