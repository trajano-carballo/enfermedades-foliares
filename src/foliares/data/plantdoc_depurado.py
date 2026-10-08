"""Depuración y partición propia dev/test de PlantDoc (candidato; sesión 5).

Reglas del equipo (2026-10-08), aplicadas ANTES del split sobre el pool de las 12 clases:
 (a) grupos de duplicados (phash <= 6) con etiquetas contradictorias: se excluyen TODAS las copias;
 (b) grupos de una sola clase: se conserva una copia (la primera por ruta ordenada);
 (c) pares de 7-10: solo se listan para revisión, no se excluyen;
 (d) revisión humana de `Tomato leaf` y `Bell_pepper leaf`: se excluye una imagen solo si AMBOS
     revisores escribieron `descartar`. Columnas vacías => nada se excluye y todo es PROVISIONAL.
Después: dev/test 70/30 estratificado por clase, semilla fija. No toca data/splits/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foliares.data.plantdoc_chequeos import componentes
from foliares.utils.seeds import rng

ESTADOS = ("conservada", "excluida_contradictoria", "excluida_duplicada", "excluida_revision", "fuera_de_alcance")
VOCAB_REVISOR = ("mantener", "descartar")
VOCAB_MOTIVO = ("1", "2", "3")


# ------------------------------------------------------------------ revisión humana
def leer_revision(planilla: pd.DataFrame) -> tuple[set[str], str, pd.DataFrame]:
    """Devuelve (rutas a excluir, estado, filas inválidas). Vocabulario exacto (tras `strip`):
    revisor_* in {"", mantener, descartar}; motivo in {"", 1, 2, 3}. Lo que está fuera del vocabulario
    NO se interpreta: la fila se reporta y no cuenta como `descartar`.
    Estado: COMPLETA solo si todas las filas tienen ambos revisores válidos y sin inválidas;
    en otro caso PROVISIONAL (con el detalle de cuántas filas faltan)."""
    p = planilla.fillna("").astype(str).apply(lambda c: c.str.strip())
    malo = (~p.revisor_1.isin(("",) + VOCAB_REVISOR)) | (~p.revisor_2.isin(("",) + VOCAB_REVISOR)) \
        | (~p.motivo.isin(("",) + VOCAB_MOTIVO))
    invalidas = planilla[malo]
    ok = p[~malo]
    excluir = set(ok.ruta[(ok.revisor_1 == "descartar") & (ok.revisor_2 == "descartar")])
    incompletas = int(((p.revisor_1 == "") | (p.revisor_2 == "")).sum())
    if incompletas == 0 and len(invalidas) == 0:
        estado = "COMPLETA"
    else:
        estado = f"PROVISIONAL ({incompletas} de {len(p)} filas sin completar por algún revisor; {len(invalidas)} filas inválidas)"
    return excluir, estado, invalidas


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
def construir_depurado(rutas, renombrados_csv, cache_hashes, planilla_csv, semilla: int = 42,
                       frac_test: float = 0.30, umbral: int = 6):
    """De los archivos al manifiesto depurado. Devuelve (df, info) con info = dict(meta, P, P6, estado_revision,
    revision_invalidas, revision_excluidas). Usa solo imágenes y phash (sin modelos)."""
    from foliares.data import plantdoc_chequeos as pc
    from foliares.data.taxonomia import CLASES_EVAL_PD
    m = pc.listar_plantdoc(rutas, renombrados_csv)
    h8, _ = pc.hashes_con_cache(rutas, m, cache_hashes)
    D, var = pc.distancias(h8)
    P = pc.tabla_pares(m, D, var, 10)
    P6 = P[P.distancia <= umbral]
    plan = pd.read_csv(planilla_csv, dtype=str, keep_default_na=False)
    excl, estado, inv = leer_revision(plan)
    meta = m.rename(columns={"split": "split_oficial"})
    df = depurar(meta, P6, excl, CLASES_EVAL_PD, semilla, frac_test)
    return df, {"meta": meta, "P": P, "P6": P6, "estado_revision": estado, "revision_invalidas": inv,
                "revision_excluidas": excl}
