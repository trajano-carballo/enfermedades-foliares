"""Candidato de particiones de PlantVillage (CANDIDATO: no escribe en data/splits/).

Reglas (decisiones del equipo, sesión 4):
- La hoja es (clase, hoja) = `leaf_id`; se parte por hoja, por clase, 70/15/15, con semilla fija.
- Las imágenes sin grupo van SOLO a train. Excepción: las clases listadas en
  `clases_por_rango` (Tomato mosaic virus, sin ningún grupo) se cortan por rangos contiguos
  de numeración con una zona de descarte entre train, val y test.
- Tope de train por clase con muestreo estratificado por sesión.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foliares.utils.seeds import rng

PARTICIONES = ("train", "val", "test")


# ---------------------------------------------------------------- partición por hoja
def _conteos(n: int, fracs: tuple[float, float, float]) -> tuple[int, int, int]:
    """Hojas para (train, val, test). Val y test redondean; con >=3 hojas ninguna queda vacía."""
    n_val, n_test = int(round(fracs[1] * n)), int(round(fracs[2] * n))
    if n >= 3:
        n_val, n_test = max(n_val, 1), max(n_test, 1)
    return n - n_val - n_test, n_val, n_test


def particionar_hojas(df: pd.DataFrame, semilla: int, fracs=(0.70, 0.15, 0.15)) -> pd.Series:
    """Partición base por hoja (por clase). Devuelve una Serie con train/val/test para las
    filas con grupo y NaN para las demás. Independiente del orden de las filas."""
    out = pd.Series(np.nan, index=df.index, dtype=object)
    for clase, g in df[df.tiene_grupo].groupby("clase", sort=True):
        hojas = np.array(sorted(g.leaf_id.unique()))
        orden = rng(semilla, "hojas", clase).permutation(len(hojas))
        n_tr, n_va, _ = _conteos(len(hojas), fracs)
        asign = {}
        for k, i in enumerate(orden):
            asign[hojas[i]] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
        out.loc[g.index] = g.leaf_id.map(asign)
    return out


# ---------------------------------------------------------------- rangos (mosaic)
def etiquetas_por_cortes(nums: np.ndarray, c1: float, c2: float, zona: float) -> np.ndarray:
    """0=train (num<=c1), -1=descarte (c1,c1+zona], 1=val (c1+zona,c2], -1=descarte
    (c2,c2+zona], 2=test (>c2+zona). La zona se mide en números de archivo."""
    return np.where(nums <= c1, 0, np.where(nums <= c1 + zona, -1,
                    np.where(nums <= c2, 1, np.where(nums <= c2 + zona, -1, 2))))


def cortes_por_proporcion(nums: np.ndarray, fracs=(0.70, 0.15, 0.15)) -> tuple[float, float]:
    """c1, c2 = número de la imagen en el rango 70 % y 85 % de la numeración ordenada."""
    s = np.sort(nums)
    n = len(s)
    return float(s[int(np.ceil(fracs[0] * n)) - 1]), float(s[int(np.ceil((fracs[0] + fracs[1]) * n)) - 1])


def cortar_por_rangos(nums: np.ndarray, zona: int, fracs=(0.70, 0.15, 0.15)) -> np.ndarray:
    """Etiquetas 0/1/2 (train/val/test) y -1 (descarte) con rangos contiguos de numeración.
    La zona de descarte se toma del tramo siguiente a cada corte (train conserva su 70 %)."""
    nums = np.asarray(nums, dtype=float)
    c1, c2 = cortes_por_proporcion(nums, fracs)
    return etiquetas_por_cortes(nums, c1, c2, zona)


def simular_fuga_por_cortes(g: pd.DataFrame, zona: int, n_cortes: int, semilla: int, clave: str,
                            frac_min: float = 0.05) -> dict:
    """Sobre una tanda numerada CON grupo (una sesión de una clase): sortea `n_cortes` pares
    de cortes al azar (cada tramo con al menos `frac_min` de las imágenes), aplica la zona de
    descarte y cuenta en cuántos cortes alguna hoja real queda con imágenes en >=2 tramos."""
    g = g.sort_values("num")
    nums = g.num.to_numpy(dtype=float)
    _, leaf_idx = np.unique(g.leaf_id.to_numpy(), return_inverse=True)
    orden = np.argsort(leaf_idx, kind="stable")
    li = leaf_idx[orden]
    inicios = np.r_[0, np.flatnonzero(np.diff(li)) + 1]
    n = len(nums)
    gen = rng(semilla, "simulacion", clave, zona)
    m = int(np.ceil(frac_min * n))
    con_fuga, hojas_cruzadas = 0, 0
    for _ in range(n_cortes):
        r1 = int(gen.integers(m, n - 2 * m + 1))
        r2 = int(gen.integers(r1 + m, n - m + 1))
        lab = etiquetas_por_cortes(nums, nums[r1 - 1], nums[r2 - 1], zona)[orden]
        mn = np.minimum.reduceat(np.where(lab >= 0, lab, 9), inicios)
        mx = np.maximum.reduceat(np.where(lab >= 0, lab, -1), inicios)
        cruzadas = int(((mx >= 0) & (mn < 9) & (mn != mx)).sum())
        con_fuga += cruzadas > 0
        hojas_cruzadas += cruzadas
    return {"unidad": clave, "zona": zona, "imagenes": n, "hojas": int(len(inicios)), "cortes": n_cortes,
            "cortes_con_fuga": int(con_fuga), "prop_con_fuga": con_fuga / n_cortes,
            "hojas_cruzadas_media": hojas_cruzadas / n_cortes}


def intervalo_wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0, c - h)), float(min(1, c + h))


def validar_zonas(df: pd.DataFrame, zonas=(10, 15, 20), n_cortes: int = 5000, semilla: int = 42,
                  criterio: float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame, int | None]:
    """Valida la zona de descarte sobre todas las tandas (clase, sesión) con grupo y numeración.
    Devuelve (por_unidad, resumen_por_zona, zona_elegida). Regla fijada de antemano: se elige
    la primera zona cuya proporción AGREGADA de cortes con fuga es <= `criterio`; si ninguna,
    None (la clase sin grupo va solo a train)."""
    base = df[df.tiene_grupo & df.num.notna()]
    filas = []
    for (clase, sesion), g in base.groupby(["clase", "sesion"], sort=True):
        if g.leaf_id.nunique() < 2 or len(g) < 20:
            continue
        for z in zonas:
            filas.append({"clase": clase, "sesion": sesion,
                          **simular_fuga_por_cortes(g, z, n_cortes, semilla, f"{clase}|{sesion}")})
    por_unidad = pd.DataFrame(filas)
    res = []
    for z, g in por_unidad.groupby("zona"):
        k, n = int(g.cortes_con_fuga.sum()), int(g.cortes.sum())
        lo, hi = intervalo_wilson(k, n)
        peor = g.loc[g.prop_con_fuga.idxmax()]
        res.append({"zona": z, "unidades": len(g), "cortes": n, "cortes_con_fuga": k, "prop_con_fuga": k / n,
                    "wilson95_bajo": lo, "wilson95_alto": hi, "peor_unidad": peor.unidad,
                    "peor_unidad_prop": peor.prop_con_fuga, "cumple": k / n <= criterio})
    resumen = pd.DataFrame(res).sort_values("zona").reset_index(drop=True)
    ok = resumen[resumen.cumple]
    return por_unidad, resumen, (int(ok.zona.iloc[0]) if len(ok) else None)


# ---------------------------------------------------------------- tope de train
def aplicar_tope(df: pd.DataFrame, particion_base: pd.Series, tope: int, semilla: int) -> pd.Series:
    """Bool por fila: True si la imagen de train se descarta por el tope. Muestreo estratificado
    por sesión (asignación proporcional, resto por mayor fracción), semilla derivada por
    (clase, sesión)."""
    excluida = pd.Series(False, index=df.index)
    for clase, g in df[particion_base == "train"].groupby("clase", sort=True):
        if len(g) <= tope:
            continue
        tam = g.groupby("sesion").size().sort_index()
        exacta = tam / tam.sum() * tope
        cuota = np.floor(exacta).astype(int)
        resto = tope - int(cuota.sum())
        for ses in (exacta - cuota).sort_values(ascending=False, kind="stable").index[:resto]:
            cuota[ses] += 1
        for ses, gs in g.groupby("sesion", sort=True):
            rutas = np.array(sorted(gs.ruta_color))
            quedan = set(rutas[rng(semilla, "tope", clase, ses).permutation(len(rutas))[: int(cuota[ses])]])
            excluida.loc[gs.index] = ~gs.ruta_color.isin(quedan)
    return excluida


# ---------------------------------------------------------------- candidato completo
def construir_candidato(manifiesto: pd.DataFrame, semilla: int = 42, fracs=(0.70, 0.15, 0.15),
                        tope: int = 1500, clases_por_rango: tuple[str, ...] = ("Tomato___Tomato_mosaic_virus",),
                        zona: int | None = 10) -> pd.DataFrame:
    """Candidato de particiones. Columnas: particion_base (antes del tope: train/val/test/descarte),
    particion (final: train/val/test/excluida), motivo_exclusion, regla, semilla.
    `zona=None` => las clases por rango van enteras a train (regla de contingencia)."""
    df = manifiesto.reset_index(drop=True)
    base = particionar_hojas(df, semilla, fracs)
    regla = pd.Series("hoja", index=df.index, dtype=object)
    base[~df.tiene_grupo] = "train"
    regla[~df.tiene_grupo] = "sin_grupo->train"
    if zona is not None:
        for clase in clases_por_rango:
            idx = df.index[(df.clase == clase) & ~df.tiene_grupo]
            lab = cortar_por_rangos(df.loc[idx, "num"].to_numpy(), zona, fracs)
            base.loc[idx] = pd.Series(lab, index=idx).map({0: "train", 1: "val", 2: "test", -1: "descarte"})
            regla.loc[idx] = f"rango_zona{zona}"
    excl_tope = aplicar_tope(df, base, tope, semilla)
    final = base.copy()
    motivo = pd.Series("", index=df.index, dtype=object)
    final[base == "descarte"], motivo[base == "descarte"] = "excluida", "zona_descarte"
    final[excl_tope], motivo[excl_tope] = "excluida", "tope_train"
    out = df[["ruta_color", "clase", "cultivo", "clase_comun", "clase_eval", "sesion", "num", "leaf_id", "tiene_grupo"]].copy()
    out["regla"], out["particion_base"], out["particion"], out["motivo_exclusion"] = regla, base, final, motivo
    out["semilla"] = semilla
    return out


def tabla_resumen(cand: pd.DataFrame) -> pd.DataFrame:
    """Imágenes y hojas por clase y partición final (más las excluidas y el total de la clase)."""
    filas = []
    for clase, g in cand.groupby("clase", sort=True):
        fila = {"clase": clase, "total": len(g), "clase_comun": bool(g.clase_comun.iloc[0]),
                "clase_eval": bool(g.clase_eval.iloc[0])}
        for p in PARTICIONES:
            s = g[g.particion == p]
            fila[f"img_{p}"], fila[f"hojas_{p}"] = len(s), int(s.leaf_id.replace("", np.nan).nunique())
        fila["excl_tope"] = int((g.motivo_exclusion == "tope_train").sum())
        fila["excl_zona"] = int((g.motivo_exclusion == "zona_descarte").sum())
        filas.append(fila)
    return pd.DataFrame(filas)
