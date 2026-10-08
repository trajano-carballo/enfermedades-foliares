"""Tests de no-fuga del candidato de particiones de PlantVillage (sesión 4).

Las pruebas con datos reales se saltean si los datasets no están en DATA_ROOT.
Las pruebas sintéticas corren siempre.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foliares.data import particion_pv as pp
from foliares.data.manifiesto_pv import construir_manifiesto
from foliares.utils.paths import cargar_rutas
from foliares.utils.seeds import semilla_derivada

CLASE_RANGO = "Tomato___Tomato_mosaic_virus"
ZONA = 20  # la que resultó de la validación (ver docs/exploracion_sesion4.md)


# ------------------------------------------------------------------ datos reales
@pytest.fixture(scope="session")
def rutas():
    r = cargar_rutas()
    if not (r.plantvillage / "raw" / "color").is_dir():
        pytest.skip("PlantVillage no está en DATA_ROOT")
    return r


@pytest.fixture(scope="session")
def manifiesto(rutas):
    return construir_manifiesto(rutas)[0]


@pytest.fixture(scope="session")
def candidato(manifiesto):
    return pp.construir_candidato(manifiesto, semilla=42, zona=ZONA)


def test_ninguna_hoja_en_mas_de_una_particion(candidato):
    # Se mira particion_base (antes del tope): una hoja con imágenes en dos tramos es fuga.
    c = candidato[candidato.tiene_grupo & candidato.particion_base.isin(pp.PARTICIONES)]
    n = c.groupby("leaf_id").particion_base.nunique()
    assert (n > 1).sum() == 0, n[n > 1].head().to_dict()


def test_la_hoja_incluye_la_clase(candidato):
    c = candidato[candidato.tiene_grupo]
    assert c.apply(lambda f: f.leaf_id.startswith(f.clase + ":::"), axis=1).all()
    # un mismo número de hoja en dos clases distintas son hojas distintas
    assert c.groupby("leaf_id").clase.nunique().max() == 1


def test_val_y_test_solo_con_grupo_mas_mosaic(candidato):
    vt = candidato[candidato.particion.isin(["val", "test"])]
    sin_grupo = vt[~vt.tiene_grupo]
    assert set(sin_grupo.clase) <= {CLASE_RANGO}
    assert (sin_grupo.regla == f"rango_zona{ZONA}").all()


def test_sin_grupo_va_a_train_salvo_mosaic(candidato):
    s = candidato[~candidato.tiene_grupo & (candidato.clase != CLASE_RANGO)]
    assert (s.particion_base == "train").all()


def test_mosaic_respeta_zona_de_descarte(candidato):
    m = candidato[candidato.clase == CLASE_RANGO]
    mx = lambda p: m[m.particion_base == p].num.max()  # noqa: E731
    mn = lambda p: m[m.particion_base == p].num.min()  # noqa: E731
    assert mn("val") - mx("train") > ZONA and mn("test") - mx("val") > ZONA


def test_tope_de_train_y_particion_base_intacta(candidato):
    train = candidato[candidato.particion == "train"]
    assert train.groupby("clase").size().max() <= 1500
    # el tope solo toca train: val/test no pierden imágenes por tope
    assert not candidato[candidato.motivo_exclusion == "tope_train"].particion_base.ne("train").any()


def test_ninguna_imagen_de_plantdoc_en_el_manifiesto(manifiesto, rutas):
    assert manifiesto.ruta_color.str.startswith("raw/plantvillage/raw/color/").all()
    assert not manifiesto.ruta_color.str.contains("plantdoc", case=False).any()
    # Contenido: igualdad exacta de bytes entre alguna imagen de PlantDoc y alguna del manifiesto.
    # Se hashea solo lo que coincide en tamaño de archivo (barato y sin falsos negativos).
    import hashlib
    tam_pv = {}
    for rel in manifiesto.ruta_color:
        tam_pv.setdefault((rutas.data_root / rel).stat().st_size, []).append(rel)
    if not (rutas.plantdoc / "train").is_dir():
        pytest.skip("PlantDoc no está en DATA_ROOT")
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()  # noqa: E731
    hashes_pv = {}
    comunes = 0
    for split in ("train", "test"):
        for p in (rutas.plantdoc / split).rglob("*"):
            if p.is_file() and p.stat().st_size in tam_pv:
                if not hashes_pv:
                    hashes_pv = {md5(rutas.data_root / r): r for rs in tam_pv.values() for r in rs}
                comunes += md5(p) in hashes_pv
    assert comunes == 0


def test_reproducible_con_la_misma_semilla(manifiesto):
    a = pp.construir_candidato(manifiesto, semilla=42, zona=ZONA)
    b = pp.construir_candidato(manifiesto.sample(frac=1, random_state=1), semilla=42, zona=ZONA)
    a, b = (x.sort_values("ruta_color").reset_index(drop=True) for x in (a, b))
    pd.testing.assert_frame_equal(a, b)  # incluso con las filas del manifiesto en otro orden


def test_otra_semilla_cambia_la_particion(manifiesto):
    a = pp.construir_candidato(manifiesto, semilla=42, zona=ZONA)
    b = pp.construir_candidato(manifiesto, semilla=43, zona=ZONA)
    assert (a.particion_base != b.particion_base).any()


def test_manifiesto_tiene_las_15_clases_y_pares(manifiesto):
    assert manifiesto.clase.nunique() == 15 and len(manifiesto) == 22787
    assert manifiesto.segmented_existe.all()
    assert manifiesto.clase_comun.groupby(manifiesto.clase).first().sum() == 13


# ------------------------------------------------------------------ sintéticos
def _sintetico(n_hojas=60, por_hoja=4, clase="A"):
    filas = []
    for h in range(n_hojas):
        for k in range(por_hoja):
            filas.append({"ruta_color": f"{clase}/h{h}_{k}.jpg", "clase": clase, "cultivo": "X", "clase_comun": True,
                          "sesion": "S1" if h % 2 else "S2", "num": float(h * por_hoja + k),
                          "leaf_id": f"{clase}:::{h}", "tiene_grupo": True})
    return pd.DataFrame(filas)


def test_particionar_hojas_70_15_15_sin_fuga():
    df = _sintetico()
    base = pp.particionar_hojas(df, 7)
    assert base.groupby(df.leaf_id).nunique().max() == 1
    hojas = base.groupby(df.leaf_id).first().value_counts().to_dict()
    assert hojas == {"train": 42, "val": 9, "test": 9}


def test_particion_de_una_clase_no_depende_de_las_otras():
    a, b = _sintetico(clase="A"), _sintetico(clase="B")
    solo_a = pp.particionar_hojas(a, 7)
    ambas = pp.particionar_hojas(pd.concat([a, b], ignore_index=True), 7).iloc[: len(a)]
    assert (solo_a.to_numpy() == ambas.to_numpy()).all()


def test_cortar_por_rangos_deja_zona():
    nums = np.arange(2047, 2421)
    for zona in (10, 15, 20):
        lab = pp.cortar_por_rangos(nums, zona)
        for a, b in ((0, 1), (1, 2)):
            assert nums[lab == b].min() - nums[lab == a].max() > zona
        assert (lab == 0).sum() == int(np.ceil(0.7 * len(nums)))


def test_tope_estratificado_por_sesion():
    df = _sintetico(n_hojas=400)  # 1600 imágenes, 2 sesiones iguales
    base = pd.Series("train", index=df.index)
    ex = pp.aplicar_tope(df, base, 1000, 3)
    quedan = df[~ex]
    assert len(quedan) == 1000 and quedan.groupby("sesion").size().tolist() == [500, 500]
    assert (ex == pp.aplicar_tope(df, base, 1000, 3)).all()


def test_simulacion_detecta_fuga_cuando_la_hoja_abarca_mas_que_la_zona():
    # hojas con imágenes separadas por 30 números: una zona de 10 no alcanza, una de 40 sí
    filas = [{"leaf_id": f"L{h}", "num": float(h + 30 * k)} for h in range(30) for k in range(2)]
    g = pd.DataFrame(filas)
    g = g.assign(tiene_grupo=True)
    chica = pp.simular_fuga_por_cortes(g, 10, 500, 1, "u")
    grande = pp.simular_fuga_por_cortes(g, 40, 500, 1, "u")
    assert chica["prop_con_fuga"] > 0.5 and grande["prop_con_fuga"] == 0


def test_semilla_derivada_estable():
    assert semilla_derivada(42, "a", "b") == semilla_derivada(42, "a", "b")
    assert semilla_derivada(42, "a", "b") != semilla_derivada(42, "b", "a")
