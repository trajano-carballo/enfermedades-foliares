"""Tablas (markdown) y figura de latencia del documento final, solo desde los JSON versionados de results/.

No calcula ninguna métrica: copia y formatea lo que ya está medido. Lo que un JSON no trae se escribe "no medido".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NO_MEDIDO = "no medido"


def num(x: float | None, d: int = 3) -> str:
    if x is None:
        return NO_MEDIDO
    return miles(round(x)) if d == 0 else f"{x:.{d}f}".replace(".", ",")


def ic(v: dict | None, d: int = 3) -> str:
    """'0,937 [0,922; 0,950]'; si la estimación no está definida, la nota del JSON (§g.7)."""
    if v is None:
        return NO_MEDIDO
    if v.get("estimacion") is None:
        return v.get("nota", "no definido")
    return f"{num(v['estimacion'], d)} [{num(v['ic_bajo'], d)}; {num(v['ic_alto'], d)}]"


def tabla_brecha(base: dict, variante: dict, comparacion: dict) -> str:
    """Macro-F1 base vs. variante en los dos tests, con IC, la brecha y la diferencia pareada."""
    rb, rv, rc = base["resultados"], variante["resultados"], comparacion["resultados"]
    filas = [
        (f"PV-test (n = {miles(rb['PV-test']['n'])})",rb["PV-test"]["base_macro_f1_ic"], rv["PV-test"]["base_macro_f1_ic"],
         ic(rc["PV-test"]["diferencia_variante_menos_base"])),
        ("PV-test sin mosaic", rb["PV-test"]["base_macro_f1_sin_mosaic_ic"], rv["PV-test"]["base_macro_f1_sin_mosaic_ic"], NO_MEDIDO),
        (f"PD-test (n = {rb['PD-test']['n']})", rb["PD-test"]["base_macro_f1_ic"], rv["PD-test"]["base_macro_f1_ic"],
         ic(rc["PD-test"]["diferencia_variante_menos_base"])),
        ("Brecha (PV-test − PD-test)", base["brecha_macro_f1_pv_test_menos_pd_test"], variante["brecha_macro_f1_pv_test_menos_pd_test"], NO_MEDIDO),
        ("Trivial PD-test (siempre Tomato late blight)", rb["PD-test"]["trivial_macro_f1_ic"], rv["PD-test"]["trivial_macro_f1_ic"], "—"),
    ]
    out = ["| | Base | Variante | Diferencia pareada (variante − base) |", "|---|---|---|---|"]
    out += [f"| {n} | {ic(b)} | {ic(v)} | {d} |" for n, b, v, d in filas]
    return "\n".join(out)


def tabla_calibracion(pv_val: dict, pd_test: dict) -> str:
    """PV-val (informativo, sesión 12) contra PD-test (fila 5, sesión 11), con T = 1 y con la T calibrada; τ fijo."""
    T, tau = pd_test["temperatura"], pd_test["tau"]
    pv1, pvT, pd1, pdT = pv_val["referencia_sin_calibrar_T1"], pv_val["resultados"], pd_test["antes_T_1"], pd_test["despues_T_final"]
    metricas = [("ECE (15 bins)", "ece"), ("Cobertura con τ", "cobertura"), ("Respondidas", "n_respondidas"),
                ("Acierto sobre lo respondido", "acierto_respondidas"), ("Errores de alta confianza (fracción del total)", "errores_alta_confianza"),
                ("Exactitud (no cambia con T)", "exactitud")]
    pv1 = {**pv1, "exactitud": pvT["exactitud"]}     # la exactitud no depende de T: misma medición en las dos columnas de PV-val
    t = num(T, 2)
    out = [f"| | PV-val, T = 1 | PV-val, T = {t} | PD-test, T = 1 | PD-test, T = {t} |", "|---|---|---|---|---|"]
    for nombre, k in metricas:
        d = 0 if k == "n_respondidas" else 3
        out.append(f"| {nombre} | {ic(pv1.get(k), d)} | {ic(pvT.get(k), d)} | {ic(pd1.get(k), d)} | {ic(pdT.get(k), d)} |")
    out += ["", f"τ = {num(tau, 1)}. n: PV-val {miles(pv_val['n_pv_val'])}, PD-test {pd_test['n_pd_test']}. "
            "PV-val no es un test (informativo); PD-test es la pasada única de la fila 5."]
    return "\n".join(out)


def miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _p(v: float) -> str:
    return num(v, 1)


def tabla_latencia(lat: dict) -> str:
    """p50 y p95 por conjunto (sin CAM → con CAM); del total, solo el p50 (decisión del 2026-11-10)."""
    r = lat["resultados_ms"]
    out = ["| Conjunto | p50 sin CAM | p95 sin CAM | p50 con CAM | p95 con CAM |", "|---|---|---|---|---|"]
    for c in ("PV-val", "PD-dev", "Commons"):
        s, k = r["sin_cam"][c], r["con_cam"][c]
        out.append(f"| {c} ({s['n']} llamadas) | {_p(s['p50_ms'])} | {_p(s['p95_ms'])} | {_p(k['p50_ms'])} | {_p(k['p95_ms'])} |")
    s, k = r["sin_cam"]["todas"], r["con_cam"]["todas"]
    out.append(f"| Total ({s['n']} llamadas) | {_p(s['p50_ms'])} | — | {_p(k['p50_ms'])} | — |")
    e, p = lat["entorno"], lat["pesos"]
    out += ["", f"ms, CPU {e['modelo_cpu']}, {e['hilos_torch']} hilos de torch, torch {e['torch']}; sin la red de Gradio. "
            f"Pesos: {miles(p['bytes'])} bytes ({num(p['MB'], 2)} MB). Carga del modelo: {num(lat['carga_del_modelo_s'], 2)} s. "
            f"Sobrecosto del CAM (p50 total): {_p(lat['sobrecosto_cam_p50_ms'])} ms. "
            f"Memoria pico del proceso: {num(lat['memoria_pico_cpu_mb'], 0)} MB."]
    return "\n".join(out)


def figura_latencia(lat: dict, ruta: str | Path) -> Path:
    """Barras de p50 por conjunto, sin y con CAM, con el p95 marcado; el total solo con p50."""
    r = lat["resultados_ms"]
    conjuntos = ["PV-val", "PD-dev", "Commons", "todas"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ancho = 0.38
    for k, (modo, color, nombre) in enumerate([("sin_cam", "#7f7f7f", "sin CAM"), ("con_cam", "#1f77b4", "con CAM")]):
        xs = [i + (k - 0.5) * ancho for i in range(len(conjuntos))]
        p50 = [r[modo][c]["p50_ms"] for c in conjuntos]
        ax.bar(xs, p50, ancho, color=color, label=f"p50 {nombre}")
        for x, c, v in zip(xs, conjuntos, p50):
            ax.text(x, v / 2, _p(v), ha="center", va="center", fontsize=7, color="white")   # adentro: no choca con la marca del p95
            if c != "todas":                     # p95 global no se cita (decisión del 2026-11-10)
                ax.plot(x, r[modo][c]["p95_ms"], "_", color="#d62728", ms=14, mew=2, label="p95" if (k, c) == (0, "PV-val") else None)
    ax.set_xticks(range(len(conjuntos)), ["PV-val", "PD-dev", "Commons", "Total (solo p50)"])
    ax.set_ylabel("ms por imagen (de punta a punta, CPU)")
    p = lat["pesos"]
    ax.set_title(f"Latencia de la app en CPU ({lat['entorno']['modelo_cpu']}, {lat['entorno']['hilos_torch']} hilos); "
                 f"pesos {num(p['MB'], 2)} MB; p95 global no se cita", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return Path(ruta)
