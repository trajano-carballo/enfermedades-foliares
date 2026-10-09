"""Depuración y partición propia dev/test de PlantDoc (candidato; sesiones 5 y 5d).

Reglas del equipo (2026-10-08, ajustadas el 2026-10-09), aplicadas ANTES del split sobre el pool de las 12 clases:
 (a) grupos de duplicados (phash <= 6) con etiquetas contradictorias: se excluyen TODAS las copias;
 (b) grupos de una sola clase: se conserva una copia (la primera por ruta ordenada);
 (c) pares de distancia 7-10: los revisa UN revisor (D3, 2026-10-09). `decision_equipo` = `misma_foto` los une
     a los grupos de duplicados (y entonces valen (a) y (b)); `distinta` no los toca; vacío => PROVISIONAL;
 (d) revisión humana de `Tomato leaf` y `Bell_pepper leaf` con UN revisor (D1, 2026-10-09): se excluye una
     imagen si `revisor_1` = `descartar`; `revisor_2` se ignora. Fila vacía => PROVISIONAL.
Después: dev/test 70/30 estratificado por clase, semilla fija. No toca data/splits/.
El resultado es COMPLETO solo si la planilla y los pares 7-10 están completos; si no, PROVISIONAL
(y `congelar_splits.py` se niega a congelar).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from foliares.data.plantdoc_chequeos import componentes
from foliares.utils.archivos import escribir_csv
from foliares.utils.git import info_repo
from foliares.utils.seeds import rng

ESTADOS = ("conservada", "excluida_contradictoria", "excluida_duplicada", "excluida_revision", "fuera_de_alcance")
COLUMNAS_PLANILLA = ("id", "split", "ruta", "revisor_1", "revisor_2", "motivo")
COLUMNAS_PARES = ("ruta_a", "ruta_b", "decision_equipo")
CFG_REVISION = {"filas_esperadas": 124, "revisores_que_deciden": ["revisor_1"], "revisores_ignorados": ["revisor_2"],
                "vocabulario": ["mantener", "descartar"], "motivos": ["1", "2", "3"]}
CFG_PARES = {"vocabulario": ["misma_foto", "distinta"]}
COMPLETO = "COMPLETO"
PROVISIONAL = "PROVISIONAL"


class PlanillaError(ValueError):
    """La planilla de revisión no se puede interpretar (formato, columnas o rutas)."""


class ParesError(ValueError):
    """El listado de pares 7-10 no se puede interpretar (formato, columnas o rutas)."""


# ------------------------------------------------------------------ lectura estricta de CSV de personas
def _leer_csv_estricto(ruta: Path, requeridas: tuple[str, ...], exactas: tuple[str, ...] | None, error: type[ValueError],
                       nombre: str) -> pd.DataFrame:
    """Lee un CSV que editan personas (Excel). Falla con un mensaje claro ante `;` como separador o columnas
    inesperadas, en lugar de dejar que pandas o el código rompan más adelante con un AttributeError."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise error(f"{nombre}: no existe {ruta}")
    texto = ruta.read_text(encoding="utf-8-sig")
    if not texto.strip():
        raise error(f"{nombre}: el archivo {ruta.name} está vacío (se esperaba al menos la fila de encabezado)")
    cabecera = texto.splitlines()[0]
    if cabecera.count(";") > cabecera.count(","):
        raise error(f"{nombre}: {ruta.name} está separado por punto y coma (';'), típico de Excel con configuración regional en "
                    f"español. Guardar como 'CSV UTF-8 (delimitado por comas)'. Encabezado leído: {cabecera!r}")
    df = pd.read_csv(io.StringIO(texto), dtype=str, keep_default_na=False)
    cols = tuple(df.columns)
    faltan = [c for c in requeridas if c not in cols]
    sobran = [c for c in cols if exactas is not None and c not in exactas]
    if faltan or sobran:
        raise error(f"{nombre}: columnas inesperadas en {ruta.name}. Faltan: {faltan or 'ninguna'}; sobran: {sobran or 'ninguna'}; "
                    f"leídas: {list(cols)}. Si es una sola columna con todo junto, el separador no es la coma.")
    return df


def leer_planilla(ruta: Path) -> pd.DataFrame:
    """Planilla de revisión (`id, split, ruta, revisor_1, revisor_2, motivo`), todo como texto. Ver `leer_revision`."""
    return _leer_csv_estricto(ruta, COLUMNAS_PLANILLA, COLUMNAS_PLANILLA, PlanillaError, "Planilla de revisión")


def _verificar_rutas(rutas: pd.Series, validas: set[str] | None, error: type[ValueError], nombre: str) -> None:
    if rutas.duplicated().any():
        raise error(f"{nombre}: rutas repetidas, p. ej. {rutas[rutas.duplicated()].iloc[:3].tolist()}")
    if validas is not None:
        ajenas = sorted(set(rutas) - validas)
        if ajenas:
            raise error(f"{nombre}: {len(ajenas)} rutas no existen en el manifiesto de PlantDoc (¿se editó la columna `ruta` "
                        f"o es otra versión del dataset?), p. ej. {ajenas[:3]}")


# ------------------------------------------------------------------ revisión humana de etiquetas (D1)
@dataclass
class Revision:
    """Resultado de leer la planilla. `estado` es COMPLETA o PROVISIONAL (n de m filas sin completar...)."""
    excluir: set[str]
    estado: str
    completa: bool
    invalidas: pd.DataFrame
    filas: int
    conteo: dict
    motivo_de: dict = field(default_factory=dict)             # ruta excluida -> motivo ("" si falta)
    descartar_sin_motivo: list = field(default_factory=list)  # ids con `descartar` y sin motivo 1/2/3
    avisos: list = field(default_factory=list)
    revisor_ignorado_con_valores: dict = field(default_factory=dict)


def leer_revision(planilla: pd.DataFrame, rutas_validas: set[str] | None = None, cfg: dict | None = None) -> Revision:
    """Interpreta la planilla con el vocabulario exacto (tras `strip`, minúsculas): los `revisores_que_deciden`
    (D1: solo `revisor_1`) in {mantener, descartar}; `motivo` in {"", 1, 2, 3}. Se excluye una imagen si TODOS los
    que deciden escribieron `descartar`. Lo que está fuera del vocabulario NO se interpreta: la fila se reporta y
    no cuenta como `descartar`. Los `revisores_ignorados` con valores: aviso, no se interpretan.
    COMPLETA solo si hay `filas_esperadas` filas, todas con los que deciden en el vocabulario y sin filas inválidas;
    si no, PROVISIONAL. `descartar` sin motivo se reporta (no cambia el estado)."""
    cfg = {**CFG_REVISION, **(cfg or {})}
    deciden, ignorados = list(cfg["revisores_que_deciden"]), list(cfg["revisores_ignorados"])
    vocab, motivos = tuple(cfg["vocabulario"]), tuple(cfg["motivos"])
    p = planilla.fillna("").astype(str).apply(lambda c: c.str.strip())
    _verificar_rutas(p.ruta, rutas_validas, PlanillaError, "Planilla de revisión")
    malo = ~p.motivo.isin(("",) + motivos)
    for r in deciden:
        malo |= ~p[r].isin(("",) + vocab)
    invalidas = planilla[malo]
    ok = p[~malo]
    descartan = np.ones(len(ok), dtype=bool)
    for r in deciden:
        descartan &= (ok[r] == "descartar").to_numpy()
    excluir = set(ok.ruta[descartan])
    motivo_de = dict(zip(ok.ruta[descartan], ok.motivo[descartan]))
    sin_motivo = list(ok.id[descartan & (ok.motivo == "").to_numpy()])
    incompletas = int(np.any([(p[r] == "").to_numpy() for r in deciden], axis=0).sum())
    avisos, ign = [], {}
    for r in ignorados:
        n = int((p[r] != "").sum()) if r in p else 0
        if n:
            ign[r] = n
            avisos.append(f"{r} tiene {n} valores: se ignoran (regla de un solo revisor, D1)")
    if sin_motivo:
        avisos.append(f"{len(sin_motivo)} filas con `descartar` sin motivo 1/2/3: {sin_motivo}")
    otras_filas = len(p) != cfg["filas_esperadas"]
    if otras_filas:
        avisos.append(f"la planilla tiene {len(p)} filas y se esperaban {cfg['filas_esperadas']}")
    completa = incompletas == 0 and len(invalidas) == 0 and not otras_filas
    estado = "COMPLETA" if completa else (
        f"{PROVISIONAL} ({incompletas} de {len(p)} filas sin completar por algún revisor; {len(invalidas)} filas inválidas"
        + (f"; filas {len(p)} de {cfg['filas_esperadas']}" if otras_filas else "") + ")")
    conteo = {r: {("(vacío)" if v == "" else v): int(n) for v, n in p[r].value_counts().items()} for r in deciden}
    return Revision(excluir, estado, completa, invalidas, len(p), conteo, motivo_de, sin_motivo, avisos, ign)


# ------------------------------------------------------------------ pares de distancia 7-10 (D3)
@dataclass
class ResultadoPares:
    """Decisiones del equipo sobre los pares 7-10. `misma_foto` (ruta_a, ruta_b) se une a los grupos de duplicados."""
    misma_foto: pd.DataFrame
    estado: str
    completo: bool
    pares: int
    conteo: dict
    invalidos: pd.DataFrame
    avisos: list = field(default_factory=list)


def pares_7_10(P: pd.DataFrame, clase_de: dict[str, str], clases_pool: list[str], umbral: int) -> pd.DataFrame:
    """Pares con distancia > `umbral` (hasta el máximo de `P`) entre imágenes del pool de las 12 clases."""
    x = P[P.distancia > umbral]
    en_pool = x.ruta_a.map(lambda r: clase_de.get(r) in clases_pool) & x.ruta_b.map(lambda r: clase_de.get(r) in clases_pool)
    return x[en_pool].reset_index(drop=True)


def _sin_pares(motivo: str) -> ResultadoPares:
    vacio = pd.DataFrame(columns=["ruta_a", "ruta_b"])
    return ResultadoPares(vacio, f"{PROVISIONAL} ({motivo})", False, 0, {}, vacio)


def leer_pares_7_10(ruta: Path | None, rutas_validas: set[str] | None = None, detectados: pd.DataFrame | None = None,
                    cfg: dict | None = None) -> ResultadoPares:
    """Lee el listado de pares 7-10 (`decision_equipo`) y lo interpreta (ver `interpretar_pares`).
    Archivo inexistente => PROVISIONAL."""
    if ruta is None or not Path(ruta).is_file():
        return _sin_pares("no existe el listado de pares 7-10")
    df = _leer_csv_estricto(Path(ruta), COLUMNAS_PARES, None, ParesError, "Listado de pares 7-10")
    return interpretar_pares(df, rutas_validas, detectados, cfg)


def interpretar_pares(df: pd.DataFrame, rutas_validas: set[str] | None = None, detectados: pd.DataFrame | None = None,
                      cfg: dict | None = None) -> ResultadoPares:
    """`decision_equipo` in {misma_foto, distinta}. COMPLETO si todas las filas tienen una decisión válida y (si se
    pasan `detectados`) el listado cubre exactamente los pares detectados; vacío => PROVISIONAL. Valores fuera del
    vocabulario se reportan y no se interpretan (esa fila no une nada)."""
    vocab = tuple((cfg or CFG_PARES)["vocabulario"])
    d = df.apply(lambda c: c.str.strip())
    for col in ("ruta_a", "ruta_b"):
        _verificar_rutas(d[col], None, ParesError, "Listado de pares 7-10")
    if rutas_validas is not None:
        ajenas = sorted((set(d.ruta_a) | set(d.ruta_b)) - rutas_validas)
        if ajenas:
            raise ParesError(f"Listado de pares 7-10: {len(ajenas)} rutas no existen en el manifiesto de PlantDoc, p. ej. {ajenas[:3]}")
    malo = ~d.decision_equipo.isin(("",) + vocab)
    vacias = int((d.decision_equipo == "").sum())
    avisos = []
    if detectados is not None:
        quiero = {frozenset(t) for t in zip(detectados.ruta_a, detectados.ruta_b)}
        tengo = {frozenset(t) for t in zip(d.ruta_a, d.ruta_b)}
        if quiero != tengo:
            avisos.append(f"el listado no coincide con los pares detectados (faltan {len(quiero - tengo)}, sobran {len(tengo - quiero)})")
    completo = vacias == 0 and not malo.any() and not avisos
    conteo = {("(vacío)" if v == "" else v): int(n) for v, n in d.decision_equipo.value_counts().items()}
    estado = COMPLETO if completo else (
        f"{PROVISIONAL} ({vacias} de {len(d)} pares sin decisión; {int(malo.sum())} inválidos" + "".join(f"; {a}" for a in avisos) + ")")
    return ResultadoPares(d.loc[(d.decision_equipo == "misma_foto") & ~malo, ["ruta_a", "ruta_b"]].reset_index(drop=True),
                          estado, completo, len(d), conteo, df[malo], avisos)


def guardar_pares_7_10(pares: pd.DataFrame, ruta: Path) -> str:
    """Escribe el listado de pares 7-10 SOLO si no existe: el archivo está versionado y trae `decision_equipo`, que
    escriben personas. Devuelve 'creado', 'existente' (mismos pares; no se toca) o 'existente_distinto' (los pares
    detectados ya no coinciden con el archivo: no se pisa nada, el equipo decide cómo resolverlo)."""
    ruta = Path(ruta)
    if not ruta.exists():
        escribir_csv(pares, ruta, bom=True)
        return "creado"
    ex = _leer_csv_estricto(ruta, ("ruta_a", "ruta_b"), None, ParesError, "Listado de pares 7-10")
    igual = {frozenset(t) for t in zip(ex.ruta_a, ex.ruta_b)} == {frozenset(t) for t in zip(pares.ruta_a, pares.ruta_b)}
    return "existente" if igual else "existente_distinto"


# ------------------------------------------------------------------ depuración
def grupos_de_duplicados(meta: pd.DataFrame, pares: pd.DataFrame) -> pd.DataFrame:
    """Componentes de pares <= 6 entre las imágenes de `meta` (pool). `pares` trae `ruta_a`, `ruta_b`.
    Devuelve columnas grupo_hash ('' si está sola), tam_grupo_hash, clases_en_grupo."""
    pos = {r: k for k, r in enumerate(meta.ruta)}
    pr = [(pos[a], pos[b]) for a, b in zip(pares.ruta_a, pares.ruta_b) if a in pos and b in pos]
    comp = componentes(len(meta), pr)
    s = pd.Series(comp)
    tam = s.map(s.value_counts()).to_numpy()
    ncl = np.array([meta.clase.iloc[np.flatnonzero(comp == c)].nunique() for c in comp])
    # id estable: numeración por la menor ruta de cada grupo (no depende del orden de las filas)
    minr = pd.Series(meta.ruta.to_numpy()).groupby(comp).transform("min")
    orden = {r: k + 1 for k, r in enumerate(sorted(set(minr[tam > 1])))}
    return pd.DataFrame({"grupo_hash": [f"H{orden[r]:04d}" if t > 1 else "" for r, t in zip(minr, tam)],
                         "tam_grupo_hash": tam, "clases_en_grupo": ncl}, index=meta.index)


def depurar(meta: pd.DataFrame, pares6: pd.DataFrame, excluir_revision: set[str], clases_pool: list[str],
            semilla: int = 42, frac_test: float = 0.30) -> pd.DataFrame:
    """Estado por imagen y `particion_propuesta` (dev/test). `meta`: ruta, clase, split_oficial,
    nombre_original (todas las carpetas candidatas); `pares6`: pares phash <= 6 (cualquier clase)."""
    df = meta.reset_index(drop=True).copy()
    en_pool = df.clase.isin(clases_pool)
    df["estado"] = np.where(en_pool, "conservada", "fuera_de_alcance")
    pool = df[en_pool]
    g = grupos_de_duplicados(pool, pares6)
    for c in ("grupo_hash", "tam_grupo_hash", "clases_en_grupo"):
        df.loc[pool.index, c] = g[c]
    df["grupo_hash"] = df.grupo_hash.fillna("")
    df["tam_grupo_hash"] = df.tam_grupo_hash.fillna(1).astype(int)
    df["clases_en_grupo"] = df.clases_en_grupo.fillna(1).astype(int)
    df["copia_conservada"] = ""
    # (a) contradictorios: todas las copias
    contra = en_pool & (df.clases_en_grupo > 1)
    df.loc[contra, "estado"] = "excluida_contradictoria"
    # (b) misma clase: primera por ruta ordenada
    for h, sub in df[en_pool & ~contra & (df.tam_grupo_hash > 1)].groupby("grupo_hash"):
        orden = sorted(sub.ruta)
        df.loc[sub.index[sub.ruta != orden[0]], "estado"] = "excluida_duplicada"
        df.loc[sub.index, "copia_conservada"] = orden[0]
    # (d) revisión humana (solo lo que sigue conservado)
    rev = df.ruta.isin(excluir_revision) & (df.estado == "conservada")
    df.loc[rev, "estado"] = "excluida_revision"
    # partición propia 70/30 estratificada por clase
    df["particion_propuesta"] = ""
    for clase in clases_pool:
        idx = df.index[(df.clase == clase) & (df.estado == "conservada")]
        rutas = np.array(sorted(df.ruta[idx]))
        n_test = int(np.floor(frac_test * len(rutas) + 0.5))
        test = set(rutas[rng(semilla, "plantdoc_split", clase).permutation(len(rutas))[:n_test]])
        df.loc[idx, "particion_propuesta"] = ["test" if r in test else "dev" for r in df.ruta[idx]]
    return df


def tabla_por_clase(df: pd.DataFrame, clases_pool: list[str]) -> pd.DataFrame:
    filas = []
    for c in clases_pool:
        g = df[df.clase == c]
        filas.append({"clase": c, "pool_antes": len(g), "pool_despues": int((g.estado == "conservada").sum()),
                      "excl_contradictoria": int((g.estado == "excluida_contradictoria").sum()),
                      "excl_duplicada": int((g.estado == "excluida_duplicada").sum()),
                      "excl_revision": int((g.estado == "excluida_revision").sum()),
                      "dev": int((g.particion_propuesta == "dev").sum()),
                      "test": int((g.particion_propuesta == "test").sum())})
    t = pd.DataFrame(filas)
    return pd.concat([t, t.drop(columns="clase").sum().to_frame().T.assign(clase="TOTAL")], ignore_index=True)


def alertas(tabla: pd.DataFrame, min_test: int = 10, min_dev: int = 30) -> pd.DataFrame:
    """Clases con test < min_test o dev < min_dev. Solo informa; no corrige nada."""
    t = tabla[tabla.clase != "TOTAL"]
    return t[(t.test < min_test) | (t.dev < min_dev)][["clase", "dev", "test"]]


# ------------------------------------------------------------------ matriz de contradictorios
def matriz_contradictorios(df: pd.DataFrame, pares6: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """(1) por combinación de clases del grupo: grupos e imágenes (train/test); (2) matriz simétrica
    clase x clase con el número de PARES (<=6) entre esas dos clases; (3) pares por clase-par."""
    contra = df[df.estado == "excluida_contradictoria"]
    filas = []
    for h, g in contra.groupby("grupo_hash"):
        filas.append({"combinacion": " | ".join(sorted(g.clase.unique())), "grupo": h, "imagenes": len(g),
                      "en_test_oficial": int((g.split_oficial == "test").sum())})
    comb = pd.DataFrame(filas)
    por_comb = (comb.groupby("combinacion").agg(grupos=("grupo", "size"), imagenes=("imagenes", "sum"),
                                                en_test_oficial=("en_test_oficial", "sum"))
                .sort_values(["grupos", "imagenes"], ascending=False).reset_index()) if len(comb) else comb
    cl = df[df.estado != "fuera_de_alcance"]
    clases = sorted(cl.clase.unique())
    mat = pd.DataFrame(0, index=clases, columns=clases)
    cl_de = dict(zip(df.ruta, df.clase))
    filas_p = []
    for r in pares6.itertuples():
        a, b = cl_de.get(r.ruta_a), cl_de.get(r.ruta_b)
        if a is None or b is None or a == b or a not in clases or b not in clases:
            continue
        mat.loc[a, b] += 1
        mat.loc[b, a] += 1
        filas_p.append({"clase_par": " | ".join(sorted((a, b))), "ruta_a": r.ruta_a, "ruta_b": r.ruta_b, "distancia": r.distancia})
    pp = pd.DataFrame(filas_p)
    por_par = (pp.groupby("clase_par").size().rename("pares").sort_values(ascending=False).reset_index()
               if len(pp) else pp)
    return {"por_combinacion": por_comb, "matriz_pares": mat, "por_clase_par": por_par, "pares": pp}


# ------------------------------------------------------------------ pipeline completo
def depurar_con_revision(meta: pd.DataFrame, P: pd.DataFrame, planilla: pd.DataFrame, pares_listado: pd.DataFrame | None,
                         umbral: int = 6, semilla: int = 42, frac_test: float = 0.30, clases_pool: list[str] | None = None,
                         revision: dict | None = None, pares: dict | None = None):
    """Las reglas (a)-(d) sobre `meta` (ruta, clase, split_oficial, nombre_original) y los pares `P` (phash <= 10, con
    `ruta_a`, `ruta_b`, `distancia`), sin tocar el disco. `planilla`: ya leída; `pares_listado`: el listado de pares
    7-10 con `decision_equipo` (None = no existe => PROVISIONAL). Devuelve (df, info); ver `construir_depurado`."""
    if clases_pool is None:
        from foliares.data.taxonomia import CLASES_EVAL_PD
        clases_pool = CLASES_EVAL_PD
    P6 = P[P.distancia <= umbral]
    validas = set(meta.ruta)
    rev = leer_revision(planilla, validas, revision)
    detectados = pares_7_10(P, dict(zip(meta.ruta, meta.clase)), clases_pool, umbral)
    par = _sin_pares("no existe el listado de pares 7-10") if pares_listado is None else interpretar_pares(
        pares_listado, validas, detectados, pares)
    unidos = pd.concat([P6[["ruta_a", "ruta_b"]], par.misma_foto], ignore_index=True)
    df = depurar(meta, unidos, rev.excluir, clases_pool, semilla, frac_test)
    pendiente = [n for n, ok in (("planilla", rev.completa), ("pares_7_10", par.completo)) if not ok]
    # texto del estado: COMPLETO, o el de lo primero que falta (el detalle de cada parte va en el .meta.json)
    estado = COMPLETO if not pendiente else (rev.estado if not rev.completa else par.estado)
    return df, {"meta": meta, "P": P, "P6": P6, "P_unidos": unidos, "rev": rev, "par": par, "detectados_7_10": detectados,
                "modo_revision": PROVISIONAL if pendiente else COMPLETO, "estado_revision": estado, "pendiente": pendiente,
                "revision_invalidas": rev.invalidas, "revision_excluidas": rev.excluir}


def construir_depurado(rutas, renombrados_csv, cache_hashes, planilla_csv, semilla: int = 42,
                       frac_test: float = 0.30, umbral: int = 6, *, pares_csv: Path | None = None,
                       revision: dict | None = None, pares: dict | None = None):
    """De los archivos al manifiesto depurado. Devuelve (df, info) con info = dict(meta, P, P6, P_unidos, rev, par,
    detectados_7_10, modo_revision, estado_revision, pendiente, revision_invalidas, revision_excluidas).
    Usa solo imágenes y phash (sin modelos). `revision` y `pares` son las secciones del YAML de reglas;
    `pares_csv` es el listado de pares 7-10 con `decision_equipo` (los `misma_foto` se unen a los grupos)."""
    from foliares.data import plantdoc_chequeos as pc
    m = pc.listar_plantdoc(rutas, renombrados_csv)
    h8, _ = pc.hashes_con_cache(rutas, m, cache_hashes)
    D, var = pc.distancias(h8)
    P = pc.tabla_pares(m, D, var, 10)
    listado = None
    if pares_csv is not None and Path(pares_csv).is_file():
        listado = _leer_csv_estricto(Path(pares_csv), COLUMNAS_PARES, None, ParesError, "Listado de pares 7-10")
    return depurar_con_revision(m.rename(columns={"split": "split_oficial"}), P, leer_planilla(planilla_csv), listado,
                                umbral, semilla, frac_test, None, revision, pares)


def meta_depurado(df: pd.DataFrame, info: dict, semilla: int, frac_test: float, umbral: int, commit_plantdoc: str,
                  repo: Path, script: str) -> dict:
    """Contenido del `.meta.json` del manifiesto depurado: parámetros, modo de revisión (COMPLETO/PROVISIONAL), qué
    queda pendiente, exclusiones por estado, clase y motivo, pares 7-10 aplicados y el commit del repo."""
    rev, par = info["rev"], info["par"]
    excl = df[df.estado.str.startswith("excluida")]
    por_clase = {c: {e: int(n) for e, n in g.estado.value_counts().items()} for c, g in excl.groupby("clase")}
    motivos = pd.Series([rev.motivo_de[r] or "sin_motivo" for r in df.ruta[df.estado == "excluida_revision"]], dtype=str)
    ref = df.set_index("ruta")
    aplicados = [{"ruta_a": a, "ruta_b": b, "clase_a": ref.clase[a], "clase_b": ref.clase[b],
                  "estado_a": ref.estado[a], "estado_b": ref.estado[b]} for a, b in zip(par.misma_foto.ruta_a, par.misma_foto.ruta_b)]
    git = info_repo(repo)
    return {"semilla": semilla, "frac_test": frac_test, "umbral_duplicado": umbral, "commit_plantdoc": commit_plantdoc,
            "revision_modo": info["modo_revision"], "revision_estado": info["estado_revision"],
            "revision_pendiente": info["pendiente"],
            "revision": {"planilla": {"estado": rev.estado, "completa": rev.completa, "filas": rev.filas,
                                      "regla": "se excluye si revisor_1 = descartar; revisor_2 se ignora (D1, 2026-10-09)",
                                      "decisiones": rev.conteo, "filas_invalidas": int(len(rev.invalidas)),
                                      "descartar_sin_motivo": rev.descartar_sin_motivo,
                                      "revisor_ignorado_con_valores": rev.revisor_ignorado_con_valores, "avisos": rev.avisos},
                         "pares_7_10": {"estado": par.estado, "completo": par.completo, "pares": par.pares,
                                        "decisiones": par.conteo, "avisos": par.avisos,
                                        "regla": "misma_foto => se une a los grupos de duplicados antes del split (D3, 2026-10-09)"}},
            "estados": {k: int(v) for k, v in df.estado.value_counts().items()},
            "exclusiones_por_clase": por_clase,
            "exclusiones_revision_por_motivo": {k: int(v) for k, v in motivos.value_counts().items()},
            "pares_7_10_aplicados": aplicados,
            "commit_repo": git["commit"], "repo_con_cambios_sin_commitear": git["cambios_sin_commitear"],
            "script": script}
