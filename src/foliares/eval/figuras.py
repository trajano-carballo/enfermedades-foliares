"""Figuras de la sesión 9: brecha con IC, matrices de confusión normalizadas por fila de los dos dominios y grilla de CAM.

Las dos primeras salen de `tests.json` (solo números: ninguna imagen de un test). La grilla muestra únicamente PlantDoc-dev y
PlantVillage-val (`foliares.explain.muestra.exigir_particion_permitida`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from foliares.explain import cam as C  # noqa: E402
from foliares.explain.muestra import exigir_particion_permitida  # noqa: E402

_CORTOS = {"Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato YLCV", "Tomato___Tomato_mosaic_virus": "Tomato mosaic"}


def nombre_corto(clase: str) -> str:
    return _CORTOS.get(clase) or clase.replace("Pepper,_bell___", "Pepper ").replace("___", " ").replace("_", " ")


def figura_brecha(tests: dict, ruta: str | Path) -> Path:
    """Macro-F1 con IC de PV-test (con y sin mosaic) y PD-test, el trivial de cada uno, y la brecha con su IC."""
    r = tests["resultados"]
    puntos = [("PV-test", r["PV-test"]["base_macro_f1_ic"]), ("PV-test sin mosaic", r["PV-test"]["base_macro_f1_sin_mosaic_ic"]),
              ("PD-test", r["PD-test"]["base_macro_f1_ic"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    y = np.arange(len(puntos))[::-1]
    for yi, (nombre, v) in zip(y, puntos):
        a1.errorbar(v["estimacion"], yi, xerr=[[v["estimacion"] - v["ic_bajo"]], [v["ic_alto"] - v["estimacion"]]], fmt="o", capsize=4,
                    color="#1f77b4" if nombre != "PD-test" else "#d62728")
        a1.text(max(v["estimacion"], 0.2), yi + 0.18, f"{v['estimacion']:.3f} [{v['ic_bajo']:.3f}; {v['ic_alto']:.3f}]", ha="center", fontsize=8)
    a1.axvline(r["PV-test"]["trivial_macro_f1_ic"]["estimacion"], color="gray", ls=":", label="trivial PV-test")
    a1.axvline(r["PD-test"]["trivial_macro_f1_ic"]["estimacion"], color="gray", ls="--", label="trivial PD-test")
    a1.set_yticks(y, [p[0] for p in puntos])
    a1.set_xlim(0, 1)
    a1.set_ylim(-0.6, len(puntos) - 0.3)
    a1.set_xlabel("macro-F1 (IC 95 % bootstrap, 2.000 réplicas estratificadas)")
    a1.set_title("Modelo base: macro-F1 por dominio")
    a1.legend(fontsize=7, loc="center")
    b = tests["brecha_macro_f1_pv_test_menos_pd_test"]
    a2.errorbar([0], [b["estimacion"]], yerr=[[b["estimacion"] - b["ic_bajo"]], [b["ic_alto"] - b["estimacion"]]], fmt="s", capsize=5,
                color="#444")
    a2.text(0, b["ic_alto"] + 0.02, f"{b['estimacion']:.3f}\n[{b['ic_bajo']:.3f}; {b['ic_alto']:.3f}]", ha="center", fontsize=8)
    a2.axhline(0, color="k", lw=0.6)
    a2.set_xlim(-1, 1)
    a2.set_ylim(0, 1)
    a2.set_xticks([])
    a2.set_ylabel("macro-F1(PV-test) − macro-F1(PD-test)")
    a2.set_title("Brecha ID → OOD", fontsize=10)
    a2.set_xlabel("remuestreo independiente de cada test", fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return Path(ruta)


def leyenda_matrices(ic_json: str) -> str:
    return f"Descriptiva, sin IC; IC por clase en {ic_json}"


def figura_matrices(tests: dict, ruta: str | Path, *, ic_json: str) -> Path:
    """Las matrices de confusión normalizadas por fila (recall en la diagonal) de PV-test y PD-test, mismo orden de clases.
    `ic_json` es la ruta (relativa al repo) del JSON con los IC por clase, que se cita en la leyenda de la figura."""
    nombres = [nombre_corto(c) for c in tests["orden_clases"]]
    fig, ejes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, test in zip(ejes, ("PV-test", "PD-test")):
        b = tests["resultados"][test]["base"]
        m = np.asarray(b["matriz_normalizada_por_fila"])
        soporte = [b["soporte_por_clase"][c] for c in tests["orden_clases"]]
        im = ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j] >= 0.005:
                    ax.text(j, i, f"{m[i, j]:.2f}".lstrip("0") if m[i, j] < 1 else "1", ha="center", va="center", fontsize=6.5,
                            color="white" if m[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(nombres)), nombres, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(nombres)), [f"{n} (n={s})" for n, s in zip(nombres, soporte)], fontsize=8)
        ax.set_xlabel("clase predicha")
        ax.set_title(f"{test} (n={tests['resultados'][test]['n']}, macro-F1 {tests['resultados'][test]['base_macro_f1_ic']['estimacion']:.3f})")
    ejes[0].set_ylabel("clase verdadera")
    fig.colorbar(im, ax=ejes, fraction=0.02, pad=0.02, label="proporción de la fila")
    fig.text(0.5, 0.005, leyenda_matrices(ic_json), ha="center", fontsize=9, style="italic")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(ruta)


def grilla_cam(celdas: list[dict], nombres: list[str], ruta: str | Path, titulo: str) -> Path:
    """Una columna por clase y una fila por tipo de imagen. Cada celda trae `imagen` (H, W, 3 en [0, 1]), `cam01` (H, W),
    `real`, `pred`, `prob`, `origen` y `fila`. Se superpone el CAM de la clase predicha."""
    filas = sorted({c["fila_grilla"] for c in celdas}, key=lambda f: f[0])
    columnas = sorted({c["columna"] for c in celdas})
    fig, ejes = plt.subplots(len(filas), len(columnas), figsize=(2.6 * len(columnas), 3.5 * len(filas)), squeeze=False)
    for ax in ejes.ravel():
        ax.axis("off")
    for c in celdas:
        ax = ejes[[f[0] for f in filas].index(c["fila_grilla"][0]), columnas.index(c["columna"])]
        ax.imshow(C.superponer(c["imagen"], c["cam01"]))
        ok = c["real"] == c["pred"]
        ax.set_title(f"real: {nombre_corto(nombres[c['real']])}\npred: {nombre_corto(nombres[c['pred']])} (p={c['prob']:.2f})",
                     fontsize=7, color="#1a7f37" if ok else "#c0262d")
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#1a7f37" if ok else "#c0262d")
            s.set_linewidth(2)
    for i, f in enumerate(filas):
        ejes[i, 0].set_ylabel(f[1], fontsize=8)
    fig.suptitle(titulo + " — CAM de la clase predicha (escala relativa por imagen); borde verde = acierto, rojo = error", fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.02, top=0.9, hspace=0.32, wspace=0.08)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return Path(ruta)


def construir_celdas(modelo, dataset, seleccion, origen: str, rotulos_slot: dict[int, str]) -> list[dict]:
    """CAM de la clase predicha para las filas `seleccion.fila` de `dataset` (solo PD-dev o PV-val). `seleccion.slot` dice en
    qué fila de la grilla va cada imagen y `rotulos_slot` cómo se rotula esa fila. Un forward del modelo completo, sin backward."""
    import torch

    from foliares.data.preproceso import MEDIA_IMAGENET, STD_IMAGENET

    exigir_particion_permitida(dataset)
    xs = torch.stack([dataset[int(f)][0] for f in seleccion.fila])
    out = C.cam_de_modelo(modelo, xs)
    cam01 = C.reescalar(out["cam"], xs.shape[-1]).numpy()
    imgs = C.desnormalizar(xs, MEDIA_IMAGENET, STD_IMAGENET)
    celdas = []
    for i, (_, s) in enumerate(seleccion.iterrows()):
        f = dataset.filas.iloc[int(s.fila)]
        celdas.append({"imagen": imgs[i], "cam01": cam01[i], "real": int(f.etiqueta), "pred": int(out["pred"][i]),
                       "prob": float(out["probs"][i].max()), "origen": origen, "ruta": f.ruta, "columna": int(f.etiqueta),
                       "fila_grilla": (int(s.slot), rotulos_slot[int(s.slot)]), "tipo": s.get("tipo")})
    return celdas
