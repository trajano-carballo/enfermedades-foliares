"""Evaluación del modelo base: selección de geometría en PlantDoc-dev y evaluación única de los tests (sesión 9).

Protocolo: `docs/protocolo_evaluacion.md`. PlantDoc-dev elige la geometría (regla a, IC pareado); los tests (PV-test y PD-test)
se evalúan una sola vez con la geometría elegida, la misma para los dos (invariante 3) y SOLO si la guarda de accesos lo
permite (decisión e). Toda métrica OOD lleva IC bootstrap (invariante 6). Las predicciones se guardan fuera del repo.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from foliares.data.congelar import sha256_archivo
from foliares.data.dataset import DatasetParticion, filas_particion
from foliares.eval import bootstrap as B
from foliares.eval import metricas
from foliares.eval.guarda_accesos import exigir_fila_accesos
from foliares.models.mobilenet import MobileNetV3SmallGAP
from foliares.training.checkpoint import cargar_checkpoint
from foliares.utils.sesion import Medidor, escribir_json, registro_base


def exigir_fuera_del_repo(ruta: str | Path, repo: str | Path) -> Path:
    ruta, repo = Path(ruta).resolve(), Path(repo).resolve()
    if ruta == repo or repo in ruta.parents:
        raise ValueError(f"La salida tiene que estar fuera del repo: {ruta}")
    return ruta


def _relativa(ruta: str | Path, base: Path) -> str:
    return Path(os.path.relpath(Path(ruta).resolve(), base)).as_posix()


def cargar_modelo_base(salida: Path, cfg: dict, n_clases: int) -> MobileNetV3SmallGAP:
    """Backbone + cabezal del checkpoint de la sesión 8, verificando su sha256. Modo eval."""
    ruta = Path(salida) / cfg["modelo_base"]["checkpoint"]
    if sha256_archivo(ruta) != cfg["modelo_base"]["sha256"]:
        raise ValueError(f"El checkpoint {ruta} no tiene el sha256 de la config")
    modelo = MobileNetV3SmallGAP(n_clases, pesos_imagenet=False)
    cargar_checkpoint(ruta, modelo, restaurar_generadores=False)
    return modelo.eval()


@torch.inference_mode()
def predecir(modelo, dataset, batch: int, dispositivo: torch.device) -> dict:
    """Logits, probabilidades (softmax), predicción y etiqueta de cada imagen del dataset, en su orden."""
    modelo.eval().to(dispositivo)
    logits = []
    for x, _ in DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0):
        logits.append(modelo(x.to(dispositivo)).float().cpu())
    z = torch.cat(logits)
    p = torch.softmax(z, 1).numpy()
    return {"logits": z.numpy(), "probs": p, "pred": p.argmax(1), "y": dataset.etiquetas.numpy()}


def tabla_predicciones(dataset, out: dict, nombres: list[str]) -> pd.DataFrame:
    df = dataset.filas[["ruta", "clase", "etiqueta"]].copy()
    df["pred"], df["clase_pred"] = out["pred"], [nombres[i] for i in out["pred"]]
    df["prob_pred"] = out["probs"].max(1)
    for i in range(len(nombres)):
        df[f"p_{i}"] = out["probs"][:, i]
    return df


def regla_geometria(y, preds: dict, base: str, n_clases: int, bs: dict) -> dict:
    """Regla (a) con la aclaración del 2026-11-04: una geometría g califica solo si el límite inferior del IC pareado de
    macro-F1(g) − macro-F1(base) es > 0 (una peor, con el IC entero debajo de 0, o una con el IC que incluye el 0, no califica);
    entre las que califican gana la de mayor diferencia media (promedio de las réplicas); si ninguna califica, queda `base`."""
    mf1 = lambda yy, pp: metricas.macro_f1(yy, pp, n_clases)  # noqa: E731
    por_geo = {g: B.bootstrap(y, p, mf1, bs["replicas"], bs["semilla"], bs["nivel"]) for g, p in preds.items()}
    comparaciones = {g: B.bootstrap_pareado(y, p, preds[base], mf1, bs["replicas"], bs["semilla"], bs["nivel"])
                     for g, p in preds.items() if g != base}
    califican = [g for g, c in comparaciones.items() if c["ic_bajo"] > 0]
    elegida = max(califican, key=lambda g: comparaciones[g]["media_replicas"]) if califican else base
    return {"macro_f1_por_geometria": por_geo, "diferencia_contra_" + base: comparaciones, "califican": califican,
            "geometria_elegida": elegida}


def _nombres(cfg_modelo: dict) -> tuple[list[str], list[str], int]:
    pv = [c["pv"] for c in cfg_modelo["clases"]]
    return pv, [c["plantdoc"] for c in cfg_modelo["clases"]], pv.index(cfg_modelo["clase_mosaic"])


def evaluar_dev(cfg: dict, cfg_modelo: dict, repo: Path, splits: Path, data_root: Path, salida: Path, run_id: str) -> dict:
    """Fase B: el modelo base en PlantDoc-dev con cada geometría candidata y la regla (a)."""
    registro = registro_base(repo, splits, {"evaluacion": cfg, "modelo": cfg_modelo}, {"bootstrap": cfg["bootstrap"]["semilla"]})
    salida = exigir_fuera_del_repo(salida, repo)
    nombres, nombres_pd, _ = _nombres(cfg_modelo)
    modelo = cargar_modelo_base(salida, cfg, len(nombres))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_corrida = salida / cfg["experimento"] / run_id
    dir_corrida.mkdir(parents=True, exist_ok=True)
    preds, tiempos, archivos = {}, {}, {}
    with Medidor() as med:
        for g in cfg["geometrias_candidatas"]:
            ds = DatasetParticion(splits, data_root, "plantdoc", "dev", "evaluacion", nombres, nombres_pd, geometria=g,
                                  tamano=cfg["tamano"])
            with Medidor() as m:
                out = predecir(modelo, ds, cfg["batch"], dev)
            tiempos[g] = m.resumen()["tiempo_s"]
            preds[g] = out["pred"]
            y = out["y"]
            ruta = dir_corrida / f"predicciones_pd_dev_{g}.csv"
            tabla_predicciones(ds, out, nombres).to_csv(ruta, index=False, lineterminator="\n")
            archivos[g] = _relativa(ruta, salida)
    regla = regla_geometria(y, preds, cfg["geometria_base"], len(nombres), cfg["bootstrap"])
    detalle = {g: metricas.resumen(y, p, nombres) for g, p in preds.items()}
    datos = {**registro, "fase": "dev", "run_id": run_id, "modelo_base": cfg["modelo_base"], "n_pd_dev": int(len(y)),
             "regla_a": regla, "detalle_por_geometria": detalle, "tiempo_prediccion_s": tiempos, **med.resumen(),
             "predicciones": archivos, "rutas_relativas_a": "la carpeta --salida",
             "nota_memoria": "memoria_pico_cpu_mb es el máximo del proceso entero"}
    escribir_json(dir_corrida / "dev_geometria.json", datos)
    _copiar_al_repo(repo, cfg, run_id, dir_corrida / "dev_geometria.json")
    return datos


def _copiar_al_repo(repo: Path, cfg: dict, run_id: str, *archivos: Path) -> None:
    destino = Path(repo) / "results" / cfg["experimento"] / run_id
    destino.mkdir(parents=True, exist_ok=True)
    for a in archivos:
        shutil.copyfile(a, destino / Path(a).name)


def matriz_normalizada(m: list[list[int]]) -> list[list[float]]:
    m = np.asarray(m, dtype=float)
    return np.divide(m, m.sum(1, keepdims=True), out=np.zeros_like(m), where=m.sum(1, keepdims=True) > 0).tolist()


def analisis_errores(m: list[list[int]], nombres: list[str]) -> dict:
    """Errores (fuera de la diagonal) entre enfermedades de la misma especie, entre especies, y la clase predicha que más
    errores concentra. La especie es el prefijo del nombre de PlantVillage (antes de `___`)."""
    m = np.asarray(m)
    especie = [n.split("___")[0] for n in nombres]
    errores = m.sum() - np.trace(m)
    misma = sum(m[i, j] for i in range(len(m)) for j in range(len(m)) if i != j and especie[i] == especie[j])
    por_pred = m.sum(0) - np.diag(m)
    k = int(por_pred.argmax())
    return {"errores": int(errores), "misma_especie": int(misma), "entre_especies": int(errores - misma),
            "proporcion_misma_especie": float(misma / errores) if errores else 0.0,
            "clase_que_mas_errores_atrae": nombres[k], "errores_hacia_esa_clase": int(por_pred[k]),
            "proporcion_hacia_esa_clase": float(por_pred[k] / errores) if errores else 0.0}


def evaluar_tests(cfg: dict, cfg_modelo: dict, config_rel: str, repo: Path, splits: Path, data_root: Path, salida: Path,
                  run_id: str, filas: dict[str, int]) -> dict:
    """Fase C: una pasada por PV-test y PD-test con la geometría elegida. La guarda corre ANTES de leer cualquier imagen."""
    if not cfg.get("geometria"):
        raise ValueError("Falta fijar `geometria` en la config (regla a en PD-dev) antes de evaluar los tests")
    accesos = {t: exigir_fila_accesos(repo, filas[t], t, config_rel) for t in ("PV-test", "PD-test")}   # guarda (e)
    registro = registro_base(repo, splits, {"evaluacion": cfg, "modelo": cfg_modelo}, {"bootstrap": cfg["bootstrap"]["semilla"]})
    salida = exigir_fuera_del_repo(salida, repo)
    nombres, nombres_pd, i_mosaic = _nombres(cfg_modelo)
    modo_sin, desempate = cfg_modelo["metricas"]["modo_sin_mosaic"], cfg_modelo["metricas"]["desempate_trivial"]
    bs = cfg["bootstrap"]
    mf1 = lambda yy, pp: metricas.macro_f1(yy, pp, len(nombres))  # noqa: E731
    mf1_sin = lambda yy, pp: metricas.macro_f1_sin_clase(yy, pp, len(nombres), i_mosaic, modo_sin)  # noqa: E731
    y_train = filas_particion(splits, "plantvillage", "train", "entrenamiento", nombres).etiqueta.to_numpy()
    trivial = metricas.clase_mayoritaria(y_train, len(nombres), desempate)
    modelo = cargar_modelo_base(salida, cfg, len(nombres))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_corrida = salida / cfg["experimento"] / run_id
    dir_corrida.mkdir(parents=True, exist_ok=True)
    geo, res, outs = cfg["geometria"], {}, {}
    with Medidor() as med:
        for test, (dataset, particion, extra) in {"PV-test": ("plantvillage", "test", None),
                                                  "PD-test": ("plantdoc", "test", nombres_pd)}.items():
            ds = DatasetParticion(splits, data_root, dataset, particion, "evaluacion", nombres, extra, geometria=geo,
                                  tamano=cfg["tamano"])
            with Medidor() as m:
                out = predecir(modelo, ds, cfg["batch"], dev)
            outs[test] = out
            ruta = dir_corrida / f"predicciones_{test.replace('-', '_').lower()}.csv"
            tabla_predicciones(ds, out, nombres).to_csv(ruta, index=False, lineterminator="\n")
            y, p, pt = out["y"], out["pred"], np.full_like(out["y"], trivial)
            r = {"n": int(len(y)), "tiempo_prediccion_s": m.resumen()["tiempo_s"], "predicciones": _relativa(ruta, salida),
                 "base": metricas.resumen(y, p, nombres, i_mosaic, modo_sin),
                 "base_macro_f1_ic": B.bootstrap(y, p, mf1, bs["replicas"], bs["semilla"], bs["nivel"]),
                 "trivial": metricas.resumen(y, pt, nombres, i_mosaic, modo_sin),
                 "trivial_macro_f1_ic": B.bootstrap(y, pt, mf1, bs["replicas"], bs["semilla"], bs["nivel"])}
            r["base"]["matriz_normalizada_por_fila"] = matriz_normalizada(r["base"]["matriz_confusion"]["filas_verdadera_columnas_predicha"])
            r["base"]["analisis_errores"] = analisis_errores(r["base"]["matriz_confusion"]["filas_verdadera_columnas_predicha"], nombres)
            if test == "PV-test":
                r["base_macro_f1_sin_mosaic_ic"] = B.bootstrap(y, p, mf1_sin, bs["replicas"], bs["semilla"], bs["nivel"])
            res[test] = r
    pv, pd_ = outs["PV-test"], outs["PD-test"]
    brecha = B.bootstrap_diferencia_independiente(pv["y"], pv["pred"], pd_["y"], pd_["pred"], mf1, bs["replicas"],
                                                  bs["semilla"], bs["nivel"])
    datos = {**registro, "fase": "tests", "run_id": run_id, "modelo_base": cfg["modelo_base"], "geometria": geo,
             "accesos": accesos, "trivial_clase": nombres[trivial], "resultados": res,
             "brecha_macro_f1_pv_test_menos_pd_test": brecha, **med.resumen(), "orden_clases": nombres,
             "rutas_relativas_a": "la carpeta --salida", "nota_memoria": "memoria_pico_cpu_mb es el máximo del proceso entero"}
    escribir_json(dir_corrida / "tests.json", datos)
    _copiar_al_repo(repo, cfg, run_id, dir_corrida / "tests.json")
    return datos


def comparar_con_base(y, p_variante, p_base, nombres: list[str], bs: dict) -> dict:
    """Macro-F1 de la variante y del base con IC, y la diferencia pareada variante − base (mismas réplicas de imágenes).
    Solo informa: no elige nada (§f del protocolo)."""
    mf1 = lambda yy, pp: metricas.macro_f1(yy, pp, len(nombres))  # noqa: E731
    return {"variante": B.bootstrap(y, p_variante, mf1, bs["replicas"], bs["semilla"], bs["nivel"]),
            "base": B.bootstrap(y, p_base, mf1, bs["replicas"], bs["semilla"], bs["nivel"]),
            "diferencia_variante_menos_base": B.bootstrap_pareado(y, p_variante, p_base, mf1, bs["replicas"], bs["semilla"], bs["nivel"])}


def _predicciones_base(salida: Path, cfg: dict, run: str, archivo: str, ds) -> np.ndarray:
    """Predicciones guardadas del modelo base para las mismas imágenes que `ds` (mismo orden y etiquetas, verificado)."""
    pred = pd.read_csv(Path(salida) / "03_brecha" / run / archivo)
    if list(pred.ruta) != list(ds.filas.ruta) or not (pred.etiqueta.to_numpy() == ds.filas.etiqueta.to_numpy()).all():
        raise ValueError(f"Las predicciones del base ({archivo}) no corresponden a las mismas imágenes")
    return pred.pred.to_numpy()


def evaluar_dev_informativo(cfg: dict, cfg_modelo: dict, repo: Path, splits: Path, data_root: Path, salida: Path, run_id: str) -> dict:
    """Etapa 4: la variante en PlantDoc-dev con la geometría fija de la config, macro-F1 con IC y diferencia pareada contra las
    predicciones guardadas del base. No toca ningún test y no elige nada."""
    registro = registro_base(repo, splits, {"evaluacion": cfg, "modelo": cfg_modelo}, {"bootstrap": cfg["bootstrap"]["semilla"]})
    salida = exigir_fuera_del_repo(salida, repo)
    nombres, nombres_pd, _ = _nombres(cfg_modelo)
    geo = cfg["geometria"]
    modelo = cargar_modelo_base(salida, cfg, len(nombres))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DatasetParticion(splits, data_root, "plantdoc", "dev", "evaluacion", nombres, nombres_pd, geometria=geo, tamano=cfg["tamano"])
    p_base = _predicciones_base(salida, cfg, cfg["referencia_base"]["run_dev"], f"predicciones_pd_dev_{geo}.csv", ds)
    dir_corrida = salida / cfg["experimento"] / run_id
    dir_corrida.mkdir(parents=True, exist_ok=True)
    with Medidor() as med:
        out = predecir(modelo, ds, cfg["batch"], dev)
    ruta = dir_corrida / f"predicciones_pd_dev_{geo}.csv"
    tabla_predicciones(ds, out, nombres).to_csv(ruta, index=False, lineterminator="\n")
    comp = comparar_con_base(out["y"], out["pred"], p_base, nombres, cfg["bootstrap"])
    datos = {**registro, "fase": "dev_informativo", "run_id": run_id, "modelo_evaluado": cfg["modelo_base"],
             "referencia_base": cfg["referencia_base"], "geometria": geo, "n_pd_dev": int(len(out["y"])),
             "macro_f1_por_geometria": {geo: comp["variante"]}, "comparacion_con_base": comp,
             "variante": metricas.resumen(out["y"], out["pred"], nombres), "predicciones": _relativa(ruta, salida),
             "solo_informativo": "PD-dev no elige nada en la etapa 4 (§f del protocolo)", **med.resumen(),
             "rutas_relativas_a": "la carpeta --salida", "nota_memoria": "memoria_pico_cpu_mb es el máximo del proceso entero"}
    escribir_json(dir_corrida / "dev_informativo.json", datos)
    _copiar_al_repo(repo, cfg, run_id, dir_corrida / "dev_informativo.json")
    return datos
