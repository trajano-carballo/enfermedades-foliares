"""Medición informativa en PV-val (sesión 12, fase 1; decisión 3 del 2026-11-10): con el modelo y la T de la config de
calibración, fracción de imágenes con probabilidad máxima calibrada ≥ τ y acierto de las que superan τ, con IC bootstrap;
más la referencia sin calibrar (T = 1) con τ = 0,5.
PV-val no es un test (rol `validacion`); ninguna regla depende de este número. No lee PV-test ni PD-test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from foliares.data.dataset import DatasetParticion
from foliares.eval.calibracion import _ic, softmax_t
from foliares.eval.evaluacion import _copiar_al_repo, _nombres, cargar_modelo_base, exigir_fuera_del_repo, predecir
from foliares.utils.sesion import Medidor, escribir_json, registro_base

CLAVES = ("cobertura", "abstencion", "n_respondidas", "acierto_respondidas", "exactitud")
REFERENCIA = ("cobertura", "n_respondidas", "acierto_respondidas")   # sin calibrar (T = 1), solo informativa


def medir_pv_val(cfg: dict, cfg_cal: dict, cfg_modelo: dict, repo: Path, splits: Path, data_root: Path, salida: Path, run_id: str) -> dict:
    if cfg_cal.get("temperatura") is None:
        raise ValueError("La config de calibración no tiene `temperatura` fijada")
    registro = registro_base(repo, splits, {"app": cfg, "calibracion": cfg_cal, "modelo": cfg_modelo},
                             {"bootstrap": cfg_cal["bootstrap"]["semilla"]})
    salida = exigir_fuera_del_repo(salida, repo)
    nombres, _, _ = _nombres(cfg_modelo)
    dir_corrida = salida / cfg["experimento"] / run_id
    dir_corrida.mkdir(parents=True, exist_ok=True)
    with Medidor() as med:
        modelo = cargar_modelo_base(salida, cfg_cal, len(nombres))
        ds = DatasetParticion(splits, data_root, "plantvillage", "val", "validacion", nombres, geometria=cfg_cal["geometria"],
                              tamano=cfg_cal["tamano"])
        out = predecir(modelo, ds, cfg_cal["batch"], torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        y, p = out["y"], softmax_t(out["logits"], cfg_cal["temperatura"])
        ic, ic1 = _ic(y, p, cfg_cal), _ic(y, softmax_t(out["logits"], 1.0), cfg_cal)
    pmax = p.max(1)
    datos = {**registro, "fase": "pv_val_informativo", "run_id": run_id, "modelo_base": cfg_cal["modelo_base"],
             "geometria": cfg_cal["geometria"], "temperatura": cfg_cal["temperatura"], "tau": cfg_cal["tau"], "n_pv_val": int(len(y)),
             "resultados": {k: ic[k] for k in CLAVES},
             "referencia_sin_calibrar_T1": {k: ic1[k] for k in REFERENCIA},
             "confianza_maxima_calibrada": {"min": float(pmax.min()), "p50": float(np.median(pmax)), "max": float(pmax.max())},
             "solo_informativo": "PV-val no es un test; ninguna regla depende de este número (decisión del 2026-11-10)",
             **med.resumen(), "nota_memoria": "memoria_pico_cpu_mb es el máximo del proceso entero"}
    escribir_json(dir_corrida / "pv_val_informativo.json", datos)
    _copiar_al_repo(repo, cfg, run_id, dir_corrida / "pv_val_informativo.json")
    return datos
