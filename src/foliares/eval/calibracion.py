"""Calibración por temperatura y abstención (etapa 5, sesión 11; §g de `docs/protocolo_evaluacion.md`).

- Regla 1: se calibra el modelo de mayor macro-F1 (con mosaic) en PV-val, leído de los JSON versionados; mayor estricto,
  empate → el base. No usa PlantDoc ni ningún test.
- Regla 2: un único T que minimiza la NLL en PD-dev, por grilla; si el óptimo cae en un borde, `TemperaturaEnBorde`.
- Regla 3: ECE de 15 bins de igual ancho sobre la confianza máxima; en PD-dev, fuera de partición (5 particiones
  estratificadas: cada imagen se calibra con una T ajustada SIN ella). La T final se ajusta con todo PD-dev.
- Reglas 4–5: τ fijo; responde si p ≥ τ. Métricas del sistema con IC bootstrap (`foliares.eval.bootstrap`).
- Regla 6: PD-test una sola vez, detrás de la guarda de accesos. PV-test no se toca en esta etapa.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from foliares.data.dataset import DatasetParticion
from foliares.eval import bootstrap as B
from foliares.eval.evaluacion import _copiar_al_repo, _nombres, _relativa, cargar_modelo_base, exigir_fuera_del_repo, predecir
from foliares.eval.guarda_accesos import exigir_fila_accesos
from foliares.utils.sesion import Medidor, escribir_json, registro_base


class TemperaturaEnBorde(RuntimeError):
    """El mínimo de la NLL cae en un extremo de la grilla: la regla 2 manda frenar y avisar."""


def softmax_t(z, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / T
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def nll(z, y, T: float) -> float:
    z = np.asarray(z, dtype=np.float64) / T
    m = z.max(1)
    lse = m + np.log(np.exp(z - m[:, None]).sum(1))
    return float(np.mean(lse - z[np.arange(len(z)), np.asarray(y)]))


def grilla_temperatura(g: dict) -> np.ndarray:
    return np.round(np.arange(g["desde"], g["hasta"] + g["paso"] / 2, g["paso"]), 10)


def ajustar_temperatura(z, y, grilla) -> float:
    i = int(np.argmin([nll(z, y, T) for T in grilla]))
    if i in (0, len(grilla) - 1):
        raise TemperaturaEnBorde(f"El óptimo de la NLL cae en el borde de la grilla (T = {grilla[i]}); la regla 2 manda frenar")
    return float(grilla[i])


def bins_fiabilidad(probs, y, n_bins: int) -> list[dict]:
    """Bins de igual ancho (k/n, (k+1)/n] sobre la confianza máxima: n, confianza media y acierto de cada uno."""
    conf, ok = probs.max(1), probs.argmax(1) == np.asarray(y)
    b = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    return [{"desde": k / n_bins, "hasta": (k + 1) / n_bins, "n": int((b == k).sum()),
             "confianza": float(conf[b == k].mean()) if (b == k).any() else None,
             "acierto": float(ok[b == k].mean()) if (b == k).any() else None} for k in range(n_bins)]


def ece(probs, y, n_bins: int) -> float:
    n = len(y)
    return float(sum(c["n"] / n * abs(c["acierto"] - c["confianza"]) for c in bins_fiabilidad(probs, y, n_bins) if c["n"]))


def metricas_sistema(y, probs, tau: float, n_bins: int) -> dict[str, float]:
    """ECE y abstención con umbral τ sobre la probabilidad máxima. Errores de alta confianza: respondidas y mal clasificadas,
    como fracción del total. Acierto sobre lo respondido: NaN si no se responde nada."""
    conf, ok = probs.max(1), probs.argmax(1) == np.asarray(y)
    resp = conf >= tau
    return {"ece": ece(probs, y, n_bins), "cobertura": float(resp.mean()), "abstencion": float(1 - resp.mean()),
            "n_respondidas": int(resp.sum()),
            "acierto_respondidas": float(ok[resp].mean()) if resp.any() else float("nan"),
            "errores_alta_confianza": float((resp & ~ok).mean()), "exactitud": float(ok.mean())}


def curva_cobertura(probs, y, taus) -> list[dict]:
    """Descriptiva (la regla 4 fija τ; no se elige nada sobre esta curva)."""
    conf, ok = probs.max(1), probs.argmax(1) == np.asarray(y)
    return [{"tau": float(t), "cobertura": float((conf >= t).mean()),
             "acierto_respondidas": float(ok[conf >= t].mean()) if (conf >= t).any() else None} for t in taus]


def particiones_estratificadas(y, k: int, semilla: int) -> np.ndarray:
    """Número de partición (0..k-1) de cada imagen, estratificado por clase."""
    f = np.empty(len(y), dtype=int)
    for i, (_, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=semilla).split(np.zeros(len(y)), y)):
        f[te] = i
    return f


def calibrar_fuera_de_particion(z, y, particiones, grilla) -> tuple[np.ndarray, list[float]]:
    """Probabilidades de cada imagen calibradas con la T ajustada en las OTRAS particiones; y la T de cada partición."""
    z, y = np.asarray(z), np.asarray(y)
    probs, ts = np.empty(z.shape), []
    for k in np.unique(particiones):
        m = particiones == k
        T = ajustar_temperatura(z[~m], y[~m], grilla)
        probs[m] = softmax_t(z[m], T)
        ts.append(T)
    return probs, ts


def regla_modelo(repo: str | Path, candidatos: dict) -> dict:
    """Regla 1: macro-F1 con mosaic en PV-val (`pv_val.macro_f1` del `resultados.json` versionado); mayor estricto, empate → base."""
    v = {k: json.loads((Path(repo) / c["resultados"]).read_text(encoding="utf-8"))["pv_val"]["macro_f1"] for k, c in candidatos.items()}
    elegido = "variante" if v["variante"] > v["base"] else "base"
    return {"macro_f1_pv_val": v, "json": {k: c["resultados"] for k, c in candidatos.items()}, "elegido": elegido,
            "run_id": candidatos[elegido]["run_id"]}


def _exigir_modelo_de_la_regla(repo: Path, cfg: dict) -> dict:
    regla = regla_modelo(repo, cfg["candidatos"])
    if regla["run_id"] != cfg["modelo_base"]["run_id"]:
        raise ValueError(f"La config calibra {cfg['modelo_base']['run_id']}, pero la regla 1 elige {regla['run_id']}")
    return regla


def _ic(y, probs, cfg: dict) -> dict:
    bs = cfg["bootstrap"]
    """IC de las métricas del sistema; acierto sobre lo respondido según §g.7 (réplicas sin respondidas afuera, contadas)."""
    f = lambda yy, pp: metricas_sistema(yy, pp, cfg["tau"], cfg["ece_bins"])  # noqa: E731
    r = B.bootstrap_multiple(y, probs, f, bs["replicas"], bs["semilla"], bs["nivel"], excluir_nan=("acierto_respondidas",))
    if r["n_respondidas"]["estimacion"] == 0:
        r["acierto_respondidas"] = {**r["acierto_respondidas"], "estimacion": None, "nota": "no definido (0 respondidas)"}
    return r


def _forward(cfg, cfg_modelo, splits, data_root, salida, particion: str, dir_corrida: Path) -> tuple[dict, str, dict]:
    """Logits del modelo elegido sobre PlantDoc `particion` (dev o test); los guarda fuera del repo."""
    nombres, nombres_pd, _ = _nombres(cfg_modelo)
    modelo = cargar_modelo_base(salida, cfg, len(nombres))
    ds = DatasetParticion(splits, data_root, "plantdoc", particion, "evaluacion", nombres, nombres_pd,
                          geometria=cfg["geometria"], tamano=cfg["tamano"])
    with Medidor() as m:
        out = predecir(modelo, ds, cfg["batch"], torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    df = ds.filas[["ruta", "clase", "etiqueta"]].copy()
    df[[f"z_{i}" for i in range(len(nombres))]] = out["logits"]
    ruta = dir_corrida / f"logits_pd_{particion}.csv"
    df.to_csv(ruta, index=False, lineterminator="\n")
    return out, _relativa(ruta, salida), m.resumen()


def calibrar_dev(cfg: dict, cfg_modelo: dict, repo: Path, splits: Path, data_root: Path, salida: Path, run_id: str) -> dict:
    """Fase dev: regla 1, T por grilla en PD-dev, ECE y métricas del sistema fuera de partición, con IC. No toca ningún test."""
    registro = registro_base(repo, splits, {"calibracion": cfg, "modelo": cfg_modelo},
                             {"bootstrap": cfg["bootstrap"]["semilla"], "particiones": cfg["particiones"]["semilla"]})
    salida = exigir_fuera_del_repo(salida, repo)
    regla = _exigir_modelo_de_la_regla(repo, cfg)
    dir_corrida = salida / cfg["experimento"] / run_id
    dir_corrida.mkdir(parents=True, exist_ok=True)
    with Medidor() as med:
        out, archivo, t_fwd = _forward(cfg, cfg_modelo, splits, data_root, salida, "dev", dir_corrida)
        z, y = out["logits"], out["y"]
        grilla = grilla_temperatura(cfg["temperatura_grilla"])
        T = ajustar_temperatura(z, y, grilla)
        assert (softmax_t(z, T).argmax(1) == z.argmax(1)).all()   # T no cambia el argmax: el macro-F1 no se mueve
        part = particiones_estratificadas(y, cfg["particiones"]["k"], cfg["particiones"]["semilla"])
        p_oof, ts = calibrar_fuera_de_particion(z, y, part, grilla)
        p1 = softmax_t(z, 1.0)
        antes, despues = _ic(y, p1, cfg), _ic(y, p_oof, cfg)
    taus = np.round(np.arange(0, 1.0001, 0.01), 2)
    datos = {**registro, "fase": "dev", "run_id": run_id, "regla_modelo": regla, "modelo_base": cfg["modelo_base"],
             "geometria": cfg["geometria"], "n_pd_dev": int(len(y)),
             "temperatura_final": T, "nll_pd_dev": {"T_1": nll(z, y, 1.0), "T_final": nll(z, y, T)},
             "temperatura_por_particion": ts, "imagenes_por_particion": np.bincount(part).tolist(), "tau": cfg["tau"],
             "antes_T_1": antes, "despues_fuera_de_particion": despues,
             "fiabilidad": {"antes_T_1": bins_fiabilidad(p1, y, cfg["ece_bins"]),
                            "despues_fuera_de_particion": bins_fiabilidad(p_oof, y, cfg["ece_bins"])},
             "curva_cobertura": {"antes_T_1": curva_cobertura(p1, y, taus), "despues_fuera_de_particion": curva_cobertura(p_oof, y, taus)},
             "logits": archivo, "tiempo_forward": t_fwd, **med.resumen(), "rutas_relativas_a": "la carpeta --salida",
             "nota_memoria": "memoria_pico_cpu_mb es el máximo del proceso entero"}
    escribir_json(dir_corrida / "calibracion_dev.json", datos)
    _copiar_al_repo(repo, cfg, run_id, dir_corrida / "calibracion_dev.json")
    return datos


def calibrar_test(cfg: dict, cfg_modelo: dict, config_rel: str, repo: Path, splits: Path, data_root: Path, salida: Path,
                  run_id: str, fila: int) -> dict:
    """Fase test: una pasada por PD-test con modelo, T y τ ya fijos en la config. La guarda corre ANTES de leer nada."""
    accesos = {"PD-test": exigir_fila_accesos(repo, fila, "PD-test", config_rel)}
    if cfg.get("temperatura") is None or not cfg.get("run_dev"):
        raise ValueError("Falta fijar `temperatura` y `run_dev` en la config (fase dev) antes de evaluar PD-test")
    dev = json.loads((Path(repo) / "results" / cfg["experimento"] / cfg["run_dev"] / "calibracion_dev.json").read_text(encoding="utf-8"))
    if dev["temperatura_final"] != cfg["temperatura"] or dev["modelo_base"]["run_id"] != cfg["modelo_base"]["run_id"] or dev["tau"] != cfg["tau"]:
        raise ValueError("La T, el modelo o τ de la config no son los de la fase dev registrada")
    registro = registro_base(repo, splits, {"calibracion": cfg, "modelo": cfg_modelo}, {"bootstrap": cfg["bootstrap"]["semilla"]})
    salida = exigir_fuera_del_repo(salida, repo)
    dir_corrida = salida / cfg["experimento"] / run_id
    dir_corrida.mkdir(parents=True, exist_ok=True)
    T = float(cfg["temperatura"])
    with Medidor() as med:
        out, archivo, t_fwd = _forward(cfg, cfg_modelo, splits, data_root, salida, "test", dir_corrida)
        z, y = out["logits"], out["y"]
        p1, pT = softmax_t(z, 1.0), softmax_t(z, T)
        antes, despues = _ic(y, p1, cfg), _ic(y, pT, cfg)
    datos = {**registro, "fase": "test", "run_id": run_id, "accesos": accesos, "modelo_base": cfg["modelo_base"],
             "geometria": cfg["geometria"], "n_pd_test": int(len(y)), "temperatura": T, "tau": cfg["tau"], "run_dev": cfg["run_dev"],
             "antes_T_1": antes, "despues_T_final": despues,
             "fiabilidad": {"antes_T_1": bins_fiabilidad(p1, y, cfg["ece_bins"]), "despues_T_final": bins_fiabilidad(pT, y, cfg["ece_bins"])},
             "logits": archivo, "tiempo_forward": t_fwd, **med.resumen(), "rutas_relativas_a": "la carpeta --salida",
             "nota_memoria": "memoria_pico_cpu_mb es el máximo del proceso entero"}
    escribir_json(dir_corrida / "calibracion_test.json", datos)
    _copiar_al_repo(repo, cfg, run_id, dir_corrida / "calibracion_test.json")
    return datos
