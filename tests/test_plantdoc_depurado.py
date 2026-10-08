"""Tests de la depuración y partición propia de PlantDoc (sesión 5)."""

import pandas as pd
import pytest

from foliares.data import plantdoc_depurado as dp
from foliares.utils.paths import cargar_rutas

CLASES = ["A", "B"]


def _meta():
    filas = []
    for c in CLASES:
        for k in range(40):
            filas.append({"ruta": f"raw/plantdoc/{'test' if k < 5 else 'train'}/{c}/img{k:02d}.jpg", "clase": c,
                          "split_oficial": "test" if k < 5 else "train", "nombre_original": f"img{k:02d}.jpg"})
    filas.append({"ruta": "raw/plantdoc/train/C/x.jpg", "clase": "C", "split_oficial": "train", "nombre_original": "x.jpg"})
    return pd.DataFrame(filas)


def _pares(lista):
    return pd.DataFrame([{"ruta_a": a, "ruta_b": b, "distancia": 0} for a, b in lista], columns=["ruta_a", "ruta_b", "distancia"])


R = lambda c, k: f"raw/plantdoc/{'test' if k < 5 else 'train'}/{c}/img{k:02d}.jpg"  # noqa: E731
PARES = _pares([(R("A", 1), R("B", 1)),            # contradictorio A|B (train y train)
                (R("A", 2), R("B", 2)), (R("A", 2), R("A", 20)),  # grupo de 3 con 2 clases (encadenado)
                (R("A", 30), R("A", 31)),         # misma clase: se conserva una
                (R("B", 3), R("B", 25))])         # misma clase con una copia en test oficial


def test_contradictorias_quitan_todas_las_copias():
    df = dp.depurar(_meta(), PARES, set(), CLASES)
    e = df.set_index("ruta").estado
    for r in (R("A", 1), R("B", 1), R("A", 2), R("B", 2), R("A", 20)):
        assert e[r] == "excluida_contradictoria"


def test_misma_clase_conserva_la_primera_por_ruta():
    df = dp.depurar(_meta(), PARES, set(), CLASES).set_index("ruta")
    assert df.estado[R("A", 30)] == "conservada" and df.estado[R("A", 31)] == "excluida_duplicada"
    # la copia test (ruta 'test/...' < 'train/...') es la que queda
    assert df.estado[R("B", 3)] == "conservada" and df.estado[R("B", 25)] == "excluida_duplicada"
    assert df.copia_conservada[R("B", 25)] == R("B", 3)


def test_fuera_de_alcance():
    df = dp.depurar(_meta(), PARES, set(), CLASES).set_index("ruta")
    assert df.estado["raw/plantdoc/train/C/x.jpg"] == "fuera_de_alcance"
    assert df.particion_propuesta["raw/plantdoc/train/C/x.jpg"] == ""


def test_dev_y_test_disjuntos_70_30_y_sin_grupo_en_ambos_lados():
    df = dp.depurar(_meta(), PARES, set(), CLASES)
    c = df[df.estado == "conservada"]
    assert set(c.particion_propuesta) == {"dev", "test"} and not c.ruta.duplicated().any()
    for clase, g in c.groupby("clase"):
        assert abs((g.particion_propuesta == "test").mean() - 0.30) < 0.04
    g = c[c.grupo_hash != ""].groupby("grupo_hash").particion_propuesta.nunique()
    assert (g <= 1).all()
    assert (df[df.estado != "conservada"].particion_propuesta == "").all()


def test_reproducible_y_depende_de_la_semilla():
    a = dp.depurar(_meta(), PARES, set(), CLASES, 42)
    b = dp.depurar(_meta().sample(frac=1, random_state=3), PARES, set(), CLASES, 42).sort_values("ruta").reset_index(drop=True)
    pd.testing.assert_frame_equal(a.sort_values("ruta").reset_index(drop=True), b)
    assert (dp.depurar(_meta(), PARES, set(), CLASES, 43).particion_propuesta != a.particion_propuesta).any()


def test_revision_solo_si_ambos_descartan_y_vocabulario():
    plan = pd.DataFrame({"id": ["1", "2", "3", "4"], "split": "train",
                         "ruta": [R("A", 10), R("A", 11), R("A", 12), R("A", 13)],
                         "revisor_1": ["descartar", "descartar", "Descartar", "mantener"],
                         "revisor_2": ["descartar", "mantener", "descartar", "mantener"],
                         "motivo": ["2", "1", "2", ""]})
    excl, estado, inv = dp.leer_revision(plan)
    assert excl == {R("A", 10)}                       # solo la fila 1; "Descartar" no se interpreta
    assert list(inv.id) == ["3"] and estado.startswith("PROVISIONAL")


def test_planilla_vacia_es_provisional_y_no_excluye():
    plan = pd.DataFrame({"id": ["1"], "split": "train", "ruta": [R("A", 10)], "revisor_1": "", "revisor_2": "", "motivo": ""})
    excl, estado, inv = dp.leer_revision(plan)
    assert excl == set() and estado.startswith("PROVISIONAL") and len(inv) == 0
    plan.loc[0, ["revisor_1", "revisor_2"]] = ["mantener", "descartar"]
    assert dp.leer_revision(plan)[1] == "COMPLETA"


def test_revision_excluye_antes_del_split():
    df = dp.depurar(_meta(), PARES, {R("A", 10)}, CLASES).set_index("ruta")
    assert df.estado[R("A", 10)] == "excluida_revision" and df.particion_propuesta[R("A", 10)] == ""


# ------------------------------------------------------------------ datos reales
@pytest.fixture(scope="module")
def real():
    r = cargar_rutas()
    cache = r.interim / "hashes_plantdoc_candidatas.npz"
    if not (r.plantdoc / "train").is_dir() or not cache.exists():
        pytest.skip("PlantDoc o la caché de hashes no están disponibles")
    args = (r, r.repo / "docs/bitacora/plantdoc_archivos_renombrados.csv", cache,
            r.repo / "docs/bitacora/revision_etiquetas_planilla.csv")
    return dp.construir_depurado(*args)


def test_real_contradictorias_todas_excluidas(real):
    df, info = real
    g = df[df.clases_en_grupo > 1]
    assert len(g) > 0 and (g.estado == "excluida_contradictoria").all()
    # ninguna imagen conservada pertenece a un grupo con etiquetas contradictorias
    assert not ((df.estado == "conservada") & (df.clases_en_grupo > 1)).any()


def test_real_particion_sin_fuga(real):
    df, _ = real
    c = df[df.estado == "conservada"]
    assert not c.ruta.duplicated().any()
    dev, test = set(c.ruta[c.particion_propuesta == "dev"]), set(c.ruta[c.particion_propuesta == "test"])
    assert not dev & test
    g = c[c.grupo_hash != ""].groupby("grupo_hash").particion_propuesta.nunique()
    assert (g <= 1).all()
    assert set(c.clase) == set(df[df.estado != "fuera_de_alcance"].clase)
