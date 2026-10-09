"""Chequeos de PlantDoc sin ningún modelo (sesión 4): duplicados por phash entre las 13
carpetas candidatas (train + test juntos), acción propuesta (NO se ejecuta) y material para
revisar etiquetas. Solo describe y propone; no mueve ni borra nada."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from foliares.data import duplicados as dup
from foliares.data.taxonomia import CLASES_PLANTDOC_CANDIDATAS
from foliares.utils.archivos import escribir_csv
from foliares.utils.paths import Rutas
from foliares.utils.seeds import rng

CATEGORIAS = ("misma clase train↔test", "misma clase train↔train", "misma clase test↔test", "clases distintas")


def clases_candidatas_pd() -> list[str]:
    return [c for v in CLASES_PLANTDOC_CANDIDATAS.values() for c in v]


def listar_plantdoc(rutas: Rutas, renombrados_csv: Path) -> pd.DataFrame:
    """Una fila por imagen de las 13 carpetas candidatas (train y test). `ruta` es relativa a
    DATA_ROOT; `nombre_original` recupera el nombre oficial de los archivos renombrados por NTFS."""
    ren = pd.read_csv(renombrados_csv)
    orig = {(r.split, r.clase, r.nombre_local): r.nombre_original for r in ren.itertuples()}
    filas = []
    for split in ("train", "test"):
        for clase in clases_candidatas_pd():
            d = rutas.plantdoc / split / clase
            for p in sorted(q for q in d.iterdir() if q.is_file()) if d.is_dir() else []:
                filas.append({"split": split, "clase": clase, "ruta": rutas.relativa(p), "nombre_local": p.name,
                              "nombre_original": orig.get((split, clase, p.name), p.name)})
    return pd.DataFrame(filas)


def hashes_con_cache(rutas: Rutas, meta: pd.DataFrame, cache: Path) -> tuple[np.ndarray, dict]:
    """phash (64 bits) de las 8 variantes (4 rotaciones x espejado) de cada imagen. Devuelve
    (array (n, 8) uint64, info con tiempo). Cachea en `cache` (.npz) y valida por lista de rutas."""
    t0 = time.time()
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["rutas"]) == meta.ruta.tolist():
            return z["h8"], {"desde_cache": True, "segundos": 0.0}
    ok, h8 = dup.hashes_uint64([rutas.absoluta(r) for r in meta.ruta], con_variantes=True)
    assert len(ok) == len(meta), "hay imágenes ilegibles en PlantDoc"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, rutas=np.array(meta.ruta.tolist()), h8=h8)
    return h8, {"desde_cache": False, "segundos": round(time.time() - t0, 1)}


def distancias(h8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Matriz (n, n) de distancia mínima de Hamming entre la identidad de una imagen y las 8
    variantes de la otra, simetrizada (min en ambos sentidos), y la variante que da el mínimo."""
    h0 = h8[:, 0]
    best = np.full((len(h0), len(h0)), 255, dtype=np.uint8)
    var = np.zeros_like(best)
    for k in range(8):
        d = dup.hamming_matriz(h8[:, k], h0)
        m = d < best
        best[m], var[m] = d[m], k
    mejor_T = best.T
    usar_T = mejor_T < best
    var = np.where(usar_T, var.T, var)
    return np.minimum(best, mejor_T), var


def categoria(a: pd.Series, b: pd.Series) -> str:
    if a.clase != b.clase:
        return CATEGORIAS[3]
    par = {a.split, b.split}
    return CATEGORIAS[0] if par == {"train", "test"} else (CATEGORIAS[1] if par == {"train"} else CATEGORIAS[2])


def tabla_pares(meta: pd.DataFrame, D: np.ndarray, var: np.ndarray, umbral: int = 10) -> pd.DataFrame:
    """Todos los pares (i<j) con distancia <= umbral, de cualquier categoría."""
    i, j = np.where(np.triu(D <= umbral, k=1))
    filas = []
    for a, b in zip(i, j):
        ra, rb = meta.iloc[a], meta.iloc[b]
        filas.append({"i": int(a), "j": int(b), "categoria": categoria(ra, rb), "distancia": int(D[a, b]),
                      "variante": int(var[a, b]),
                      "split_a": ra.split, "clase_a": ra.clase, "ruta_a": ra.ruta, "nombre_original_a": ra.nombre_original,
                      "split_b": rb.split, "clase_b": rb.clase, "ruta_b": rb.ruta, "nombre_original_b": rb.nombre_original})
    p = pd.DataFrame(filas)
    p["umbral"] = np.where(p.distancia <= 6, "estricto (<=6)", "exploratorio (7-10)")
    return p.sort_values(["categoria", "distancia", "ruta_a"]).reset_index(drop=True)


def componentes(n: int, pares: list[tuple[int, int]]) -> np.ndarray:
    """Union-find: id de componente (0..k-1, por orden de aparición) para cada imagen."""
    padre = list(range(n))

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for a, b in pares:
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            padre[max(ra, rb)] = min(ra, rb)
    return np.array([raiz(x) for x in range(n)])


def sacar_de_dev(meta: pd.DataFrame, pares: pd.DataFrame, incluir_clases_distintas: bool = False) -> set[int]:
    """Índices de imágenes de train (dev) que la regla propuesta sacaría: nunca de test.
    Se arman componentes de duplicados con los `pares` dados (el caller filtra por umbral). Por
    componente: si hay una copia en test, se sacan todas las de train; si no, se conserva una
    (la primera por ruta) y se sacan las demás; si el componente mezcla clases (etiquetas
    contradictorias) no se elige una etiqueta: se sacan todas las copias de train.
    `incluir_clases_distintas=False` ignora los pares entre clases distintas."""
    if not incluir_clases_distintas:
        pares = pares[pares.categoria != CATEGORIAS[3]]
    if not len(pares):
        return set()
    comp = componentes(len(meta), list(zip(pares.i, pares.j)))
    sacar: set[int] = set()
    for c in np.unique(comp[np.unique(np.r_[pares.i, pares.j])]):
        idx = np.where(comp == c)[0]
        tr = [k for k in idx if meta.split.iloc[k] == "train"]
        hay_test = any(meta.split.iloc[k] == "test" for k in idx)
        if hay_test or meta.clase.iloc[idx].nunique() > 1:
            sacar |= set(tr)
        else:
            sacar |= set(sorted(tr, key=lambda k: meta.ruta.iloc[k])[1:])
    return sacar


def conteos_dev_test(meta: pd.DataFrame, sacar: set[int]) -> pd.DataFrame:
    """Imágenes por clase en dev (= train oficial) y test, antes y después de sacar `sacar`."""
    m = meta.assign(sacada=[k in sacar for k in range(len(meta))])
    t = m.groupby("clase").apply(lambda g: pd.Series({
        "dev_antes": int((g.split == "train").sum()),
        "dev_despues": int(((g.split == "train") & ~g.sacada).sum()),
        "test": int((g.split == "test").sum())}), include_groups=False)
    t["dev_sacadas"] = t.dev_antes - t.dev_despues
    return t.reset_index()


def accion_propuesta(fila: pd.Series, sacar: set[int]) -> str:
    """Texto de la acción propuesta por par (NO se ejecuta nada). `sacar` = resultado de
    sacar_de_dev(..., incluir_clases_distintas=True) con los pares estrictos."""
    if fila.umbral != "estricto (<=6)":
        return "Sin acción propuesta: rango exploratorio, revisar a ojo antes de cualquier acción."
    quita = [fila[f"ruta_{x}"] for x, i in (("a", fila.i), ("b", fila.j))
             if fila[f"split_{x}"] == "train" and i in sacar]
    if fila.categoria == CATEGORIAS[2]:
        return "Mantener ambas (regla: nunca sacar de test); declarar que el test tiene una copia interna."
    if fila.categoria == CATEGORIAS[3]:
        base = "Etiquetas contradictorias: la etiqueta la decide el equipo. "
        if fila.split_a == "test" or fila.split_b == "test":
            return base + "Regla: sacar de dev la copia de train; la copia de test no se toca (su etiqueta queda en revisión)."
        return base + "Regla: sacar de dev ambas copias (no se elige etiqueta); si el equipo decide la etiqueta, se conserva una."
    if quita:
        return "Sacar de dev: " + " | ".join(quita)
    return "Conservar esta copia de dev (la otra copia del grupo es la que se saca)."


# ------------------------------------------------------------------ material visual
def _fuente(tam: int):
    try:
        return ImageFont.load_default(size=tam)
    except TypeError:  # Pillow viejo
        return ImageFont.load_default()


def hoja_de_contacto(items: list[tuple[str, Path]], salida: Path, titulo: str, ncols: int = 5,
                     lado: int = 230) -> Path:
    """Hoja de contacto: miniatura cuadrada (letterbox) con el id anónimo. `items` = [(id, ruta)]."""
    nrows = -(-len(items) // ncols)
    alto_tit, alto_id = 46, 34
    W, H = ncols * (lado + 10) + 10, alto_tit + nrows * (lado + alto_id + 10) + 10
    hoja = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(hoja)
    d.text((10, 10), titulo, fill="black", font=_fuente(24))
    for k, (id_, ruta) in enumerate(items):
        x, y = 10 + (k % ncols) * (lado + 10), alto_tit + (k // ncols) * (lado + alto_id + 10)
        try:
            with Image.open(ruta) as im:
                im = im.convert("RGB")
                im.thumbnail((lado, lado))
                hoja.paste(im, (x + (lado - im.width) // 2, y + alto_id + (lado - im.height) // 2))
        except Exception:  # noqa: BLE001
            d.text((x + 10, y + alto_id + 10), "ilegible", fill="red", font=_fuente(16))
        d.rectangle([x, y + alto_id, x + lado, y + alto_id + lado], outline="#bbbbbb")
        d.text((x + 4, y + 2), id_, fill="#003399", font=_fuente(26))
    salida.parent.mkdir(parents=True, exist_ok=True)
    hoja.save(salida)
    return salida


def material_revision(rutas: Rutas, meta: pd.DataFrame, clases: dict[str, str], salida_dir: Path,
                      semilla: int, por_hoja: int = 20) -> pd.DataFrame:
    """Planilla (id, split, ruta, revisor_1, revisor_2, motivo) + hojas de contacto con id anónimo.
    `clases` = {prefijo de id: carpeta de PlantDoc}. Los ids se asignan en orden aleatorio
    (semilla fija) dentro de cada clase: no revelan split ni orden de archivo."""
    filas = []
    for pref, clase in clases.items():
        sub = meta[meta.clase == clase].reset_index(drop=True)
        orden = rng(semilla, "revision", clase).permutation(len(sub))
        for n, k in enumerate(orden, 1):
            filas.append({"id": f"{pref}-{n:03d}", "split": sub.split[k], "ruta": sub.ruta[k],
                          "nombre_original": sub.nombre_original[k], "clase": clase,
                          "revisor_1": "", "revisor_2": "", "motivo": ""})
    plan = pd.DataFrame(filas)
    for pref, clase in clases.items():
        g = plan[plan.clase == clase].reset_index(drop=True)
        for pag in range(-(-len(g) // por_hoja)):
            parte = g.iloc[pag * por_hoja:(pag + 1) * por_hoja]
            hoja_de_contacto([(r.id, rutas.absoluta(r.ruta)) for r in parte.itertuples()],
                             salida_dir / f"contacto_{pref}_{pag + 1}.png",
                             f"Carpeta de PlantDoc: {clase} - {parte.id.iloc[0]} a {parte.id.iloc[-1]}")
    return plan[["id", "split", "ruta", "revisor_1", "revisor_2", "motivo"]]


def guardar_planilla(plan: pd.DataFrame, ruta: Path) -> str:
    """Escribe la planilla de revisión SOLO si no existe: el archivo está versionado y los revisores escriben en él;
    regenerarlo borraría sus decisiones. Devuelve 'creada', 'existente' (mismos `id` y `ruta`; no se toca) o
    'existente_distinta' (el material regenerado ya no coincide con el archivo: no se pisa nada)."""
    ruta = Path(ruta)
    if not ruta.exists():
        escribir_csv(plan, ruta, bom=True)
        return "creada"
    ex = pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if not {"id", "ruta"} <= set(ex.columns):
        return "existente_distinta"
    return "existente" if set(zip(ex.id, ex.ruta)) == set(zip(plan.id, plan.ruta)) else "existente_distinta"


def grilla_pares(pares: pd.DataFrame, rutas: Rutas, salida: Path, titulo: str, pares_por_fila: int = 3):
    """Grilla con los pares uno al lado del otro (A, B, A, B, ...). Rótulo: split, clase,
    distancia y los primeros caracteres del nombre original."""
    from foliares.data import visual
    items = []
    for r in pares.itertuples():
        for x in ("a", "b"):
            rot = (f"{getattr(r, 'split_' + x)} | {getattr(r, 'clase_' + x)[:26]}\n"
                   f"{getattr(r, 'nombre_original_' + x)[:30]}" + (f"\nd={r.distancia}" if x == "b" else ""))
            items.append((rutas.absoluta(getattr(r, "ruta_" + x)), rot))
    fig = visual.grilla(items, ncols=2 * pares_por_fila, lado=2.0, titulo=titulo, guardar=salida)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return salida
