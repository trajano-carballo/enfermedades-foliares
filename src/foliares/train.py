"""Modelo base (etapa 2): entrena solo el `Linear(576, 12)` sobre features GAP precalculadas, con early stopping por PV-val.

Decisiones del equipo de la sesión 8 (a–h, en `docs/bitacora/decisiones.md`): sin aumentación ni ponderación de clases,
Adam LR 1e-3, batch 256, hasta 100 épocas, paciencia 10, checkpoint por el mejor macro-F1 en PV-val con mosaic, semilla 42
determinista. Solo PlantVillage-train (entrenamiento) y PlantVillage-val (validación): PV-test y PlantDoc no se tocan acá.
"""

from __future__ import annotations

import copy
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from foliares.data.congelar import sha256_archivo
from foliares.data.dataset import DatasetParticion
from foliares.eval import metricas
from foliares.models.mobilenet import MobileNetV3SmallGAP
from foliares.training.checkpoint import guardar_checkpoint
from foliares.training.features import clave_features, congelar_backbone, features_con_cache
from foliares.utils.seeds import fijar_semillas
from foliares.utils.sesion import Medidor, escribir_json, registro_base


class DesbalanceExcedido(RuntimeError):
    """PV-train supera el umbral max/min de la decisión (b): se frena antes de entrenar."""


def exigir_fuera_del_repo(ruta: str | Path, repo: str | Path, nombre: str) -> Path:
    ruta, repo = Path(ruta).resolve(), Path(repo).resolve()
    if ruta == repo or repo in ruta.parents:
        raise ValueError(f"{nombre} tiene que estar fuera del repo (decisión g): {ruta}")
    return ruta


def conteo_por_clase(etiquetas: torch.Tensor, nombres: list[str]) -> dict[str, int]:
    cuenta = torch.bincount(etiquetas, minlength=len(nombres)).tolist()
    return dict(zip(nombres, cuenta))


def control_desbalance(conteo: dict[str, int], umbral: float) -> dict:
    vals = list(conteo.values())
    cociente = max(vals) / min(vals) if min(vals) else float("inf")
    return {"max": max(vals), "min": min(vals), "cociente_max_min": round(cociente, 4), "umbral": umbral,
            "excede": cociente > umbral}


def entrenar_cabezal(fc: nn.Linear, Xtr: torch.Tensor, ytr: torch.Tensor, Xva: torch.Tensor, yva: torch.Tensor, ent: dict,
                     semilla: int, determinista: bool, n_clases: int, al_mejorar=None, clase_excluible: int | None = None,
                     modo_sin: str | None = None) -> dict:
    """Entrena `fc` sobre features. Reinicia `fc` desde la semilla (mismo resultado con la misma semilla). Early stopping
    por macro-F1 en val con mosaic (estrictamente mayor; ante empate queda la primera época). Deja en `fc` los pesos de la
    mejor época. `al_mejorar(epoca, optimizador)` se llama cada vez que mejora (para guardar el checkpoint). Con `clase_excluible` y
    `modo_sin`, la curva trae también el macro-F1 sin esa clase (solo se reporta; no elige la época)."""
    fijar_semillas(semilla, determinista=determinista)
    fc.reset_parameters()
    if ent["optimizador"] != "adam":
        raise ValueError(f"Optimizador no previsto: {ent['optimizador']!r}")
    opt = torch.optim.Adam(fc.parameters(), lr=ent["lr"])
    gen = torch.Generator().manual_seed(semilla)
    mejor_f1, mejor_epoca, mejor_estado, sin_mejora, curva = -1.0, 0, None, 0, []
    detenido_por = "max_epocas"
    for epoca in range(1, ent["max_epocas"] + 1):
        t0 = time.perf_counter()
        fc.train()
        perm, total = torch.randperm(len(Xtr), generator=gen), 0.0
        for i in range(0, len(Xtr), ent["batch"]):
            idx = perm[i:i + ent["batch"]]
            opt.zero_grad(set_to_none=True)
            perdida = F.cross_entropy(fc(Xtr[idx]), ytr[idx])
            perdida.backward()
            opt.step()
            total += perdida.item() * len(idx)
        fc.eval()
        with torch.no_grad():
            logits = fc(Xva)
            perdida_val = F.cross_entropy(logits, yva).item()
            pred = logits.argmax(1)
        f1 = metricas.macro_f1(yva, pred, n_clases)
        curva.append({"epoca": epoca, "perdida_train": total / len(Xtr), "perdida_val": perdida_val,
                      "macro_f1_val": f1, "exactitud_val": float((pred == yva).float().mean()),
                      **({"macro_f1_val_sin_clase": metricas.macro_f1_sin_clase(yva, pred, n_clases, clase_excluible, modo_sin)}
                         if modo_sin else {}),
                      "tiempo_s": round(time.perf_counter() - t0, 4)})
        if f1 > mejor_f1:
            mejor_f1, mejor_epoca, mejor_estado, sin_mejora = f1, epoca, copy.deepcopy(fc.state_dict()), 0
            if al_mejorar is not None:
                al_mejorar(epoca, opt)
        else:
            sin_mejora += 1
            if sin_mejora >= ent["paciencia"]:
                detenido_por = "paciencia"
                break
    fc.load_state_dict(mejor_estado)
    return {"epoca_elegida": mejor_epoca, "macro_f1_val_elegida": mejor_f1, "epocas_corridas": len(curva),
            "detenido_por": detenido_por, "curva": curva}


def _exigir_implementado(cfg: dict) -> None:
    """La config no puede pedir algo que el código no hace (M3, review de la sesión 8): fallaría en silencio."""
    pedidos = {"entrenamiento.criterio": (cfg["entrenamiento"].get("criterio"), "macro_f1_val_con_mosaic"),
               "aumentacion": (cfg.get("aumentacion"), "ninguna"), "ponderacion_clases": (cfg.get("ponderacion_clases"), False)}
    malos = {k: v for k, (v, ok) in pedidos.items() if v != ok}
    if malos:
        raise ValueError(f"La config pide algo que no está implementado: {malos} (implementado: "
                         f"{ {k: ok for k, (_, ok) in pedidos.items()} })")


def _relativa(ruta: str | Path, base: Path) -> str:
    """Ruta relativa a la carpeta de salida, con `/`: el JSON no lleva rutas de una máquina en particular."""
    return Path(os.path.relpath(Path(ruta).resolve(), base)).as_posix()


def _exigir_definiciones(cfg: dict) -> tuple[str, str]:
    """Las dos definiciones que el equipo tiene que fijar ANTES de calcular los números (ver `foliares.eval.metricas`)."""
    m = cfg.get("metricas") or {}
    modo, desempate = m.get("modo_sin_mosaic"), m.get("desempate_trivial")
    if modo not in metricas.MODOS_SIN_CLASE or desempate not in metricas.DESEMPATES:
        raise ValueError("Falta fijar en la config `metricas.modo_sin_mosaic` "
                         f"({metricas.MODOS_SIN_CLASE}) y `metricas.desempate_trivial` ({metricas.DESEMPATES}): "
                         "son decisiones del equipo y se registran antes de calcular los números")
    return modo, desempate


def ejecutar(cfg: dict, repo: Path, splits_dir: Path, data_root: Path, salida: Path, cache_dir: Path, run_id: str,
             limite: int | None = None) -> dict:
    """Corrida completa. Escribe el JSON en `salida/<experimento>/<run_id>/` (fuera del repo) y, si no es una prueba corta,
    copia los JSON chicos a `results/<experimento>/<run_id>/` del repo."""
    registro = registro_base(repo, splits_dir, cfg, {})          # estado de git antes de escribir nada
    salida = exigir_fuera_del_repo(salida, repo, "--salida")
    cache_dir = exigir_fuera_del_repo(cache_dir, repo, "el caché de features")
    _exigir_implementado(cfg)
    nombres = [c["pv"] for c in cfg["clases"]]
    i_mosaic = nombres.index(cfg["clase_mosaic"])
    geo, tam = cfg["preproceso"]["geometria"], cfg["preproceso"]["tamano"]
    ds = {p: DatasetParticion(splits_dir, data_root, "plantvillage", p, rol, nombres, geometria=geo, tamano=tam)
          for p, rol in (("train", "entrenamiento"), ("val", "validacion"))}
    conteo = {p: conteo_por_clase(d.etiquetas, nombres) for p, d in ds.items()}
    desbalance = control_desbalance(conteo["train"], cfg["umbral_desbalance"])
    desbalance["aceptado_por_el_equipo"] = bool(cfg.get("desbalance_aceptado_por_el_equipo", False))
    if desbalance["excede"] and not desbalance["aceptado_por_el_equipo"]:
        raise DesbalanceExcedido(f"PV-train: max/min = {desbalance['max']}/{desbalance['min']} = "
                                 f"{desbalance['cociente_max_min']} > {desbalance['umbral']} (decisión b): se frena antes de "
                                 f"entrenar. Conteo: {conteo['train']}")
    modo_sin, desempate = _exigir_definiciones(cfg)
    if limite:                                                   # prueba corta: las primeras `limite` filas de cada partición
        for d in ds.values():
            d.filas = d.filas.iloc[:limite].reset_index(drop=True)
    registro["semillas"] = fijar_semillas(cfg["semilla"], determinista=cfg["determinista"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = congelar_backbone(MobileNetV3SmallGAP(len(nombres), pesos_imagenet=cfg["modelo"]["pesos_imagenet"]))
    sha_csv = sha256_archivo(Path(splits_dir) / cfg["splits"]["plantvillage"])
    feats = {}
    with Medidor() as med_ext:
        for p, d in ds.items():
            clave = clave_features("plantvillage", p, d.rutas, d.filas.etiqueta.tolist(), sha_csv, geo, tam,
                                   cfg["modelo"]["pesos_imagenet"], dev.type)
            feats[p] = features_con_cache(modelo, d, clave, cache_dir, cfg["features"]["batch"], dev)
    dir_corrida = salida / cfg["experimento"] / run_id
    ruta_ckpt = dir_corrida / "mejor_checkpoint.pt"
    ckpt = {}

    def al_mejorar(epoca, opt):
        ckpt.update(guardar_checkpoint(ruta_ckpt, modelo, opt, epoca, extra={"run_id": run_id, "epoca": epoca}))

    modelo.to("cpu")
    with Medidor() as med_ent:
        ent = entrenar_cabezal(modelo.fc, feats["train"]["X"], feats["train"]["y"], feats["val"]["X"], feats["val"]["y"],
                               cfg["entrenamiento"], cfg["semilla"], cfg["determinista"], len(nombres), al_mejorar,
                               i_mosaic, modo_sin)
    with torch.no_grad():
        pred_val = modelo.fc(feats["val"]["X"]).argmax(1)
    yva = feats["val"]["y"]
    val = metricas.resumen(yva, pred_val, nombres, i_mosaic, modo_sin)
    trivial_clase = metricas.clase_mayoritaria(feats["train"]["y"], len(nombres), desempate)
    trivial = metricas.resumen(yva, torch.full_like(yva, trivial_clase), nombres, i_mosaic, modo_sin)
    trivial["clase"] = nombres[trivial_clase]
    datos = {**registro, "run_id": run_id, "prueba_corta": bool(limite), "dispositivo_features": dev.type,
             "conteo_por_clase": conteo, "control_desbalance": desbalance,
             "features": {p: {**{k: v for k, v in f.items() if k not in ("X", "y")}, "archivo": _relativa(f["archivo"], salida)}
                          for p, f in feats.items()},
             "tiempo_extraccion_s": med_ext.resumen()["tiempo_s"], "memoria_extraccion": med_ext.resumen(),
             "tiempo_entrenamiento_s": med_ent.resumen()["tiempo_s"], "memoria_entrenamiento": med_ent.resumen(),
             "entrenamiento": ent, "checkpoint": {**ckpt, "ruta": _relativa(ckpt["ruta"], salida)}, "pv_val": val,
             "trivial_pv_val": trivial, "rutas_relativas_a": "la carpeta --salida",
             "nota_memoria": "memoria_pico_cpu_mb es el máximo del proceso entero hasta ese momento (en entrenamiento incluye la "
                             "extracción); memoria_pico_gpu_mb se reinicia en cada bloque",
             "orden_clases": cfg["clases"]}
    escribir_json(dir_corrida / "resultados.json", datos)
    if not limite:
        destino = Path(repo) / "results" / cfg["experimento"] / run_id
        destino.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dir_corrida / "resultados.json", destino / "resultados.json")
        for f in feats.values():
            shutil.copyfile(Path(f["archivo"]).with_suffix(".meta.json"), destino / Path(f["archivo"]).with_suffix(".meta.json").name)
    return datos

