"""Reglas de revisión humana (D1: un revisor para la planilla; D3: pares 7-10) y guards de los archivos versionados.
Todo sobre datos sintéticos en tmp_path: no toca los datasets ni la planilla real."""

import json

import pandas as pd
import pytest

from foliares.data import plantdoc_chequeos as pc
from foliares.data import plantdoc_depurado as dp
from foliares.utils.archivos import escribir_csv

POOL = ["A", "B", "D"]


def R(c, k):
    return f"raw/plantdoc/{'test' if k < 5 else 'train'}/{c}/img{k:02d}.jpg"


def _meta():
    filas = [{"ruta": R(c, k), "clase": c, "split_oficial": "test" if k < 5 else "train", "nombre_original": f"img{k:02d}.jpg"}
             for c in POOL for k in range(40)]
    return pd.DataFrame(filas)


def _P(*tripletas):
    """Pares (ruta_a, ruta_b, distancia) con las columnas que usa la depuración."""
    return pd.DataFrame([{"ruta_a": a, "ruta_b": b, "distancia": d} for a, b, d in tripletas], columns=["ruta_a", "ruta_b", "distancia"])


# pares <= 6 (mecánicos) y 7-10 (para el revisor) del escenario
P6 = [(R("A", 1), R("B", 1), 2), (R("D", 30), R("D", 31), 0)]
P710 = [(R("A", 10), R("B", 10), 8), (R("A", 11), R("A", 12), 9)]


def _planilla(valores, filas_extra=0, **cols):
    """Planilla sobre imágenes de la clase A (img20...). `valores` = revisor_1 por fila."""
    n = len(valores)
    df = pd.DataFrame({"id": [f"TL-{k:03d}" for k in range(n)], "split": "train", "ruta": [R("A", 20 + k) for k in range(n)],
                       "revisor_1": valores, "revisor_2": "", "motivo": ""})
    for c, v in cols.items():
        df[c] = v
    return df


def _listado(decisiones, pares=P710):
    return pd.DataFrame({"ruta_a": [a for a, _, _ in pares], "ruta_b": [b for _, b, _ in pares], "decision_equipo": decisiones})


def _correr(planilla, listado, semilla=42, cfg_rev=None):
    cfg = {"filas_esperadas": len(planilla), **(cfg_rev or {})}
    return dp.depurar_con_revision(_meta(), _P(*P6, *P710), planilla, listado, 6, semilla, 0.30, POOL, cfg)


COMPLETA = ["mantener"] * 4
PARES_OK = ["distinta", "distinta"]


# ------------------------------------------------------------------ planilla (D1)
def test_planilla_vacia_provisional():
    rev = dp.leer_revision(_planilla([""] * 4), cfg={"filas_esperadas": 4})
    assert not rev.completa and rev.excluir == set()
    assert rev.estado == "PROVISIONAL (4 de 4 filas sin completar por algún revisor; 0 filas inválidas)"


def test_planilla_parcial_provisional():
    rev = dp.leer_revision(_planilla(["mantener", "descartar", "", ""]), cfg={"filas_esperadas": 4})
    assert not rev.completa and rev.estado.startswith("PROVISIONAL (2 de 4")
    assert rev.excluir == {R("A", 21)}              # lo que ya está decidido se aplica igual


def test_planilla_completa_con_un_solo_revisor():
    rev = dp.leer_revision(_planilla(["mantener", "descartar", "mantener", "descartar"], motivo=["", "2", "", "3"]),
                           cfg={"filas_esperadas": 4})
    assert rev.completa and rev.estado == "COMPLETA"
    assert rev.excluir == {R("A", 21), R("A", 23)} and rev.motivo_de == {R("A", 21): "2", R("A", 23): "3"}
    assert rev.descartar_sin_motivo == []


def test_planilla_con_mayusculas_no_se_interpreta():
    rev = dp.leer_revision(_planilla(["Mantener", "DESCARTAR", "descartar", "mantener"]), cfg={"filas_esperadas": 4})
    assert not rev.completa and list(rev.invalidas.id) == ["TL-000", "TL-001"]
    assert rev.excluir == {R("A", 22)}               # "DESCARTAR" no cuenta como descartar
    assert "PROVISIONAL" in rev.estado and "2 filas inválidas" in rev.estado


def test_revisor_2_con_valores_avisa_y_se_ignora():
    pl = _planilla(["mantener", "descartar", "mantener", "mantener"])
    pl["revisor_2"] = ["descartar", "mantener", "", "valor raro"]
    rev = dp.leer_revision(pl, cfg={"filas_esperadas": 4})
    assert rev.completa and len(rev.invalidas) == 0                # `revisor_2` ni siquiera se valida
    assert rev.excluir == {R("A", 21)}                              # fila 0: r2 = descartar pero r1 = mantener => no excluye
    assert rev.revisor_ignorado_con_valores == {"revisor_2": 3} and any("revisor_2" in a for a in rev.avisos)


def test_descartar_sin_motivo_se_reporta_pero_excluye():
    rev = dp.leer_revision(_planilla(["descartar", "mantener", "mantener", "mantener"]), cfg={"filas_esperadas": 4})
    assert rev.descartar_sin_motivo == ["TL-000"] and rev.excluir == {R("A", 20)} and rev.completa
    assert any("sin motivo" in a for a in rev.avisos)


def test_motivo_fuera_de_vocabulario_es_fila_invalida():
    rev = dp.leer_revision(_planilla(["descartar", "mantener", "mantener", "mantener"], motivo=["7", "", "", ""]), cfg={"filas_esperadas": 4})
    assert not rev.completa and rev.excluir == set() and len(rev.invalidas) == 1


def test_planilla_con_otra_cantidad_de_filas_no_es_completa():
    rev = dp.leer_revision(_planilla(["mantener"] * 3), cfg={"filas_esperadas": 4})
    assert not rev.completa and "filas 3 de 4" in rev.estado


def test_ruta_inexistente_en_el_manifiesto_es_error_claro():
    pl = _planilla(["mantener"] * 4)
    pl.loc[2, "ruta"] = "raw/plantdoc/train/A/no_existe.jpg"
    with pytest.raises(dp.PlanillaError, match="no existen en el manifiesto"):
        dp.leer_revision(pl, set(_meta().ruta), {"filas_esperadas": 4})
    with pytest.raises(dp.PlanillaError, match="repetidas"):
        dp.leer_revision(_planilla(["mantener"] * 4).assign(ruta=R("A", 20)), set(_meta().ruta), {"filas_esperadas": 4})


def test_planilla_con_punto_y_coma_es_error_claro(tmp_path):
    f = tmp_path / "planilla.csv"
    f.write_text("id;split;ruta;revisor_1;revisor_2;motivo\nTL-000;train;x;mantener;;\n", encoding="utf-8-sig")
    with pytest.raises(dp.PlanillaError, match="punto y coma"):
        dp.leer_planilla(f)


def test_planilla_con_columnas_inesperadas_es_error_claro(tmp_path):
    f = tmp_path / "planilla.csv"
    f.write_text("id,split,ruta,revisor_1,revisor_2,motivo,comentario\nTL-000,train,x,mantener,,,\n", encoding="utf-8")
    with pytest.raises(dp.PlanillaError, match="sobran: \\['comentario'\\]"):
        dp.leer_planilla(f)
    f.write_text("id,split,ruta,revisor_2,motivo\nTL-000,train,x,,\n", encoding="utf-8")
    with pytest.raises(dp.PlanillaError, match="Faltan: \\['revisor_1'\\]"):
        dp.leer_planilla(f)
    f.write_text("", encoding="utf-8")
    with pytest.raises(dp.PlanillaError, match="vacío"):
        dp.leer_planilla(f)


def test_planilla_con_bom_y_crlf_se_lee(tmp_path):
    f = tmp_path / "planilla.csv"
    f.write_bytes("id,split,ruta,revisor_1,revisor_2,motivo\r\nTL-000,train,x,mantener,,\r\n".encode("utf-8-sig"))
    assert list(dp.leer_planilla(f).columns)[0] == "id" and dp.leer_planilla(f).revisor_1[0] == "mantener"


# ------------------------------------------------------------------ pares 7-10 (D3)
def test_pares_vacios_es_provisional_y_no_cambia_nada():
    base, _ = _correr(_planilla(COMPLETA), None)
    df, info = _correr(_planilla(COMPLETA), _listado(["", ""]))
    assert info["par"].estado.startswith("PROVISIONAL (2 de 2 pares sin decisión") and not info["par"].completo
    assert info["modo_revision"] == "PROVISIONAL" and info["pendiente"] == ["pares_7_10"]
    pd.testing.assert_frame_equal(df, base)            # el listado vacío es igual a no tener listado


def test_misma_foto_entre_clases_excluye_todas_las_copias():
    df, info = _correr(_planilla(COMPLETA), _listado(["misma_foto", "distinta"]))
    e = df.set_index("ruta").estado
    assert e[R("A", 10)] == "excluida_contradictoria" and e[R("B", 10)] == "excluida_contradictoria"
    assert e[R("A", 11)] == "conservada" and e[R("A", 12)] == "conservada"        # `distinta` no toca
    assert len(info["par"].misma_foto) == 1


def test_misma_foto_de_la_misma_clase_conserva_la_primera_por_ruta():
    df, _ = _correr(_planilla(COMPLETA), _listado(["distinta", "misma_foto"]))
    d = df.set_index("ruta")
    primera = sorted([R("A", 11), R("A", 12)])[0]
    otra = R("A", 12) if primera == R("A", 11) else R("A", 11)
    assert d.estado[primera] == "conservada" and d.estado[otra] == "excluida_duplicada" and d.copia_conservada[otra] == primera
    assert d.estado[R("A", 10)] == "conservada" and d.estado[R("B", 10)] == "conservada"


def test_misma_foto_se_une_a_un_grupo_existente():
    # A1~B1 (mecánico, contradictorio) y A1~A10 (7-10 misma_foto): el grupo crece y A10 también sale
    pares = [(R("A", 1), R("A", 10), 9)]
    df, _ = dp.depurar_con_revision(_meta(), _P(*P6, *pares), _planilla(COMPLETA), _listado(["misma_foto"], pares), 6, 42, 0.30, POOL,
                                    {"filas_esperadas": 4})
    e = df.set_index("ruta").estado
    assert all(e[R(c, k)] == "excluida_contradictoria" for c, k in (("A", 1), ("B", 1), ("A", 10)))


def test_pares_con_valor_fuera_de_vocabulario_no_unen_y_no_completan():
    df, info = _correr(_planilla(COMPLETA), _listado(["Misma_foto", "distinta"]))
    assert not info["par"].completo and len(info["par"].invalidos) == 1 and len(info["par"].misma_foto) == 0
    assert df.set_index("ruta").estado[R("A", 10)] == "conservada"


def test_listado_de_pares_que_no_coincide_con_los_detectados_no_es_completo():
    df, info = _correr(_planilla(COMPLETA), _listado(["distinta"], P710[:1]))
    assert not info["par"].completo and any("no coincide" in a for a in info["par"].avisos)


def test_pares_con_ruta_inexistente_es_error_claro():
    malo = _listado(["distinta", "distinta"]).assign(ruta_b=["raw/plantdoc/train/A/no_existe.jpg", R("A", 12)])
    with pytest.raises(dp.ParesError, match="no existen en el manifiesto"):
        _correr(_planilla(COMPLETA), malo)


def test_pares_con_punto_y_coma_es_error_claro(tmp_path):
    f = tmp_path / "pares.csv"
    f.write_text("ruta_a;ruta_b;decision_equipo\nx;y;misma_foto\n", encoding="utf-8-sig")
    with pytest.raises(dp.ParesError, match="punto y coma"):
        dp.leer_pares_7_10(f)


def test_sin_listado_de_pares_es_provisional(tmp_path):
    assert dp.leer_pares_7_10(tmp_path / "no_existe.csv").estado.startswith("PROVISIONAL")
    _, info = _correr(_planilla(COMPLETA), None)
    assert info["pendiente"] == ["pares_7_10"]


# ------------------------------------------------------------------ modo COMPLETO
def test_completo_exige_planilla_y_pares_completos():
    _, solo_planilla = _correr(_planilla(COMPLETA), _listado(["", ""]))
    _, solo_pares = _correr(_planilla(["mantener", "", "", ""]), _listado(PARES_OK))
    _, ambos = _correr(_planilla(COMPLETA), _listado(PARES_OK))
    assert solo_planilla["modo_revision"] == "PROVISIONAL" and solo_planilla["pendiente"] == ["pares_7_10"]
    assert solo_pares["modo_revision"] == "PROVISIONAL" and solo_pares["pendiente"] == ["planilla"]
    assert solo_pares["estado_revision"].startswith("PROVISIONAL (3 de 4 filas sin completar")
    assert ambos["modo_revision"] == "COMPLETO" and ambos["estado_revision"] == "COMPLETO" and ambos["pendiente"] == []


# ------------------------------------------------------------------ la exclusión es previa al split y solo mueve lo afectado
def test_exclusion_previa_al_split_y_solo_cambian_las_clases_afectadas():
    base, _ = _correr(_planilla(["mantener"] * 4), _listado(PARES_OK))
    # se descartan 2 imágenes de A (planilla) y se une un par B~B de otra clase vía `misma_foto`
    pares = [(R("B", 20), R("B", 21), 9)]
    cambiado, _ = dp.depurar_con_revision(_meta(), _P(*P6, *pares), _planilla(["descartar", "mantener", "descartar", "mantener"], motivo=["1", "", "2", ""]),
                                          _listado(["misma_foto"], pares), 6, 42, 0.30, POOL, {"filas_esperadas": 4})
    a, b = base.set_index("ruta"), cambiado.set_index("ruta")
    assert (b.estado[[R("A", 20), R("A", 22)]] == "excluida_revision").all() and (b.particion_propuesta[[R("A", 20), R("A", 22)]] == "").all()
    # la clase sin cambios (D) queda idéntica; A y B cambian su partición
    d = [r for r in a.index if "/D/" in r]
    # (el número de `grupo_hash` es cosmético: se numera de forma global y puede correrse si aparecen grupos nuevos)
    pd.testing.assert_frame_equal(a.loc[d].drop(columns="grupo_hash"), b.loc[d].drop(columns="grupo_hash"))
    for clase in ("A", "B"):
        cl = [r for r in a.index if f"/{clase}/" in r]
        assert (a.loc[cl, "particion_propuesta"] != b.loc[cl, "particion_propuesta"]).any(), clase
    # excluidas antes del split: el 70/30 se calcula sobre lo que queda
    for clase, g in b[b.estado == "conservada"].groupby("clase"):
        assert abs((g.particion_propuesta == "test").mean() - 0.30) < 0.04


def test_meta_registra_modo_exclusiones_pares_y_commit(tmp_path):
    pares = [(R("B", 20), R("B", 21), 9)]
    df, info = dp.depurar_con_revision(_meta(), _P(*P6, *pares), _planilla(["descartar", "mantener", "mantener", "mantener"], motivo=["2", "", "", ""]),
                                       _listado(["misma_foto"], pares), 6, 42, 0.30, POOL, {"filas_esperadas": 4})
    meta = dp.meta_depurado(df, info, 42, 0.30, 6, "abc", tmp_path, "nb")           # tmp_path no es un repo git
    json.dumps(meta)                                                                  # serializable
    assert meta["revision_modo"] == "COMPLETO" and meta["revision_estado"] == "COMPLETO" and meta["revision_pendiente"] == []
    assert meta["exclusiones_por_clase"]["A"] == {"excluida_contradictoria": 1, "excluida_revision": 1}
    assert meta["exclusiones_por_clase"]["B"] == {"excluida_contradictoria": 1, "excluida_duplicada": 1} and meta["exclusiones_por_clase"]["D"] == {"excluida_duplicada": 1}
    assert meta["exclusiones_revision_por_motivo"] == {"2": 1}
    assert meta["pares_7_10_aplicados"][0]["estado_a"] in ("conservada", "excluida_duplicada")
    assert meta["commit_repo"] == "(sin git)" and meta["repo_con_cambios_sin_commitear"] is None
    assert meta["revision"]["planilla"]["decisiones"] == {"revisor_1": {"mantener": 3, "descartar": 1}}


# ------------------------------------------------------------------ guards de los archivos versionados
def test_guard_planilla_no_pisa_las_decisiones(tmp_path):
    plan = _planilla([""] * 3).drop(columns=[])
    f = tmp_path / "docs" / "planilla.csv"
    assert pc.guardar_planilla(plan, f) == "creada" and f.is_file()
    f.write_text(f.read_text(encoding="utf-8-sig").replace("TL-000,train,raw/plantdoc/train/A/img20.jpg,,,", "TL-000,train,raw/plantdoc/train/A/img20.jpg,descartar,,2"),
                 encoding="utf-8-sig")
    antes = f.read_bytes()
    assert b"descartar" in antes
    assert pc.guardar_planilla(plan, f) == "existente" and f.read_bytes() == antes
    otra = plan.assign(ruta=[R("A", 30 + k) for k in range(3)])
    assert pc.guardar_planilla(otra, f) == "existente_distinta" and f.read_bytes() == antes


def test_guard_pares_nunca_pierde_decision_equipo(tmp_path):
    out = _P(*P710).assign(categoria="x", distancia="8", decision_equipo="")
    f = tmp_path / "docs" / "pares.csv"
    assert dp.guardar_pares_7_10(out, f) == "creado"
    ex = pd.read_csv(f, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    ex["decision_equipo"] = ["misma_foto", "distinta"]
    escribir_csv(ex, f, bom=True)                                  # el revisor completó el listado
    antes = f.read_bytes()
    assert b"misma_foto" in antes
    assert dp.guardar_pares_7_10(out, f) == "existente" and f.read_bytes() == antes
    assert dp.guardar_pares_7_10(out.iloc[:1], f) == "existente_distinto" and f.read_bytes() == antes
