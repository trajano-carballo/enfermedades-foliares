"""Fine-tuning parcial con aumentación dirigida (etapa 4, sesión 10; decisiones a–h en `docs/bitacora/decisiones.md` y §f del
protocolo). Forward completo (no features precalculadas): las imágenes de PV-train se aumentan en cada época.

- Parte del modelo base (checkpoint de la sesión 8, verificado por sha256). Entrenables: `features[i]` de la config y `fc`; el
  resto congelado y en modo eval (BatchNorm de ImageNet fija).
- Adam con un solo LR; selección por el mejor macro-F1 en PV-val con mosaic (estrictamente mayor), paciencia y máximo de épocas
  de la config. Solo PV-train (entrenamiento) y PV-val (validación): PlantDoc no interviene (invariante 1).
- Checkpoint al final de cada época (`ultimo.pt`) y del mejor (`mejor.pt`), con los generadores; si el directorio ya tiene
  `ultimo.pt`, se retoma desde la época siguiente a la última completa. El azar de cada época (orden de los lotes y aumentación)
  sale de semillas derivadas de (semilla, época), así que retomar no cambia lo que vería una corrida continua en CPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from foliares.data.aumentacion import transformacion_train
from foliares.data.congelar import sha256_archivo
from foliares.data.dataset import DatasetParticion
from foliares.eval import metricas
from foliares.models.mobilenet import MobileNetV3SmallGAP, descongelar_parcial, modo_entrenamiento_parcial
from foliares.training.checkpoint import cargar_checkpoint, guardar_checkpoint
from foliares.utils.seeds import fijar_semillas, semilla_derivada
from foliares.utils.sesion import Medidor, escribir_json, registro_base


def huella_config(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()


def exigir_implementado(cfg: dict) -> None:
    ft = cfg["finetune"]
    malos = {k: v for k, v in {"optimizador": ft["optimizador"], "criterio": ft["criterio"]}.items()
             if v not in ("adam", "macro_f1_val_con_mosaic")}
    if malos:
        raise ValueError(f"La config pide algo que no está implementado: {malos}")


def cargar_base(ruta: str | Path, sha256: str, n_clases: int) -> MobileNetV3SmallGAP:
    if sha256_archivo(ruta) != sha256:
        raise ValueError(f"El checkpoint del modelo base {ruta} no tiene el sha256 de la config")
    modelo = MobileNetV3SmallGAP(n_clases, pesos_imagenet=False)
    cargar_checkpoint(ruta, modelo, restaurar_generadores=False)
    return modelo


@torch.inference_mode()
def predecir_val(modelo, cargador, dispositivo) -> tuple[torch.Tensor, torch.Tensor, float]:
    modelo.eval()
    preds, ys, perdida, n = [], [], 0.0, 0
    for x, y in cargador:
        logits = modelo(x.to(dispositivo))
        perdida += F.cross_entropy(logits, y.to(dispositivo), reduction="sum").item()
        n += len(y)
        preds.append(logits.argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys), torch.cat(preds), perdida / n


def entrenar(cfg: dict, cfg_modelo: dict, repo: Path, splits: Path, data_root: Path, salida: Path, ckpt_dir: Path,
             checkpoint_base: Path, run_id: str, limite: int | None = None) -> dict:
    """Corrida completa (o prueba corta con `limite` imágenes por partición). Devuelve y escribe `resultados.json` en
    `salida/<experimento>/<run_id>/`; los checkpoints van a `ckpt_dir/<experimento>/<run_id>/`."""
    registro = registro_base(repo, splits, {"finetune": cfg, "modelo": cfg_modelo}, {})   # git antes de escribir nada
    exigir_implementado(cfg)
    ft, nombres = cfg["finetune"], [c["pv"] for c in cfg_modelo["clases"]]
    k, i_mosaic = len(nombres), nombres.index(cfg_modelo["clase_mosaic"])
    modo_sin = cfg_modelo["metricas"]["modo_sin_mosaic"]
    geo, tam = cfg["preproceso"]["geometria"], cfg["preproceso"]["tamano"]
    ds_tr = DatasetParticion(splits, data_root, "plantvillage", "train", "entrenamiento", nombres,
                             preproceso=transformacion_train(cfg["aumentacion"], geo, tam))
    ds_va = DatasetParticion(splits, data_root, "plantvillage", "val", "validacion", nombres, geometria=geo, tamano=tam)
    if limite:
        for d in (ds_tr, ds_va):
            d.filas = d.filas.groupby("etiqueta", group_keys=False).head(max(1, limite // k)).reset_index(drop=True)
    registro["semillas"] = fijar_semillas(cfg["semilla"], determinista=cfg["determinista"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = descongelar_parcial(cargar_base(checkpoint_base, cfg["modelo_base"]["sha256"], k), ft["bloques_entrenables"]).to(dev)
    entrenables = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.Adam(entrenables, lr=ft["lr"])
    dir_ck = Path(ckpt_dir) / cfg["experimento"] / run_id
    dir_res = Path(salida) / cfg["experimento"] / run_id
    huella = huella_config({"finetune": cfg, "limite": limite})
    estado = {"epoca": 0, "mejor_f1": -1.0, "mejor_epoca": 0, "sin_mejora": 0, "curva": [], "reanudaciones": []}
    if (dir_ck / "ultimo.pt").is_file():                                    # retomar desde la última época completa
        info = cargar_checkpoint(dir_ck / "ultimo.pt", modelo, opt)
        if info["extra"].get("huella_config") != huella:
            raise ValueError(f"{dir_ck / 'ultimo.pt'} es de otra config: no se retoma")
        estado = json.loads(info["extra"]["estado"])
        estado["reanudaciones"].append(estado["epoca"])
    va = DataLoader(ds_va, batch_size=ft["batch"], shuffle=False, num_workers=ft["workers"])
    with Medidor() as med:
        for epoca in range(estado["epoca"] + 1, ft["max_epocas"] + 1):
            if estado["sin_mejora"] >= ft["paciencia"]:
                break
            t0 = time.perf_counter()
            torch.manual_seed(semilla_derivada(cfg["semilla"], "aumentacion", epoca))   # azar de la aumentación de esta época
            gen = torch.Generator().manual_seed(semilla_derivada(cfg["semilla"], "orden", epoca))
            tr = DataLoader(ds_tr, batch_size=ft["batch"], shuffle=True, generator=gen, num_workers=ft["workers"])
            modo_entrenamiento_parcial(modelo, ft["bloques_entrenables"])
            total, n = 0.0, 0
            for x, y in tr:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(set_to_none=True)
                perdida = F.cross_entropy(modelo(x), y)
                perdida.backward()
                opt.step()
                total += perdida.item() * len(y)
                n += len(y)
            yv, pv, perdida_val = predecir_val(modelo, va, dev)
            f1 = metricas.macro_f1(yv, pv, k)
            estado["curva"].append({"epoca": epoca, "perdida_train": total / n, "perdida_val": perdida_val, "macro_f1_val": f1,
                                    "macro_f1_val_sin_mosaic": metricas.macro_f1_sin_clase(yv, pv, k, i_mosaic, modo_sin),
                                    "exactitud_val": float((pv == yv).float().mean()), "tiempo_s": round(time.perf_counter() - t0, 2)})
            if f1 > estado["mejor_f1"]:
                estado.update(mejor_f1=f1, mejor_epoca=epoca, sin_mejora=0)
                guardar_checkpoint(dir_ck / "mejor.pt", modelo, opt, epoca, extra={"run_id": run_id, "epoca": epoca,
                                                                                    "huella_config": huella})
            else:
                estado["sin_mejora"] += 1
            estado["epoca"] = epoca
            guardar_checkpoint(dir_ck / "ultimo.pt", modelo, opt, epoca,
                               extra={"run_id": run_id, "huella_config": huella, "estado": json.dumps(estado)})
            print(f"época {epoca}: pérdida train {total / n:.4f} | val macro-F1 {f1:.4f} | {estado['curva'][-1]['tiempo_s']} s", flush=True)
    detenido_por = "paciencia" if estado["sin_mejora"] >= ft["paciencia"] else "max_epocas"
    mejor = cargar_base_entrenado(dir_ck / "mejor.pt", k)
    yv, pv, _ = predecir_val(mejor.to(dev), va, dev)
    datos = {**registro, "run_id": run_id, "prueba_corta": bool(limite), "dispositivo": dev.type,
             "modelo_base": cfg["modelo_base"], "n_train": len(ds_tr), "n_val": len(ds_va),
             "parametros_entrenables": sum(p.numel() for p in entrenables), "parametros_totales": sum(p.numel() for p in modelo.parameters()),
             "epoca_elegida": estado["mejor_epoca"], "epocas_corridas": estado["epoca"], "detenido_por": detenido_por,
             "reanudaciones_desde_epoca": estado["reanudaciones"], "curva": estado["curva"],
             "pv_val": metricas.resumen(yv, pv, nombres, i_mosaic, modo_sin),
             "checkpoint_mejor": {"ruta": f"{cfg['experimento']}/{run_id}/mejor.pt", "sha256": sha256_archivo(dir_ck / "mejor.pt"),
                                  "relativa_a": "--ckpt-dir"},
             **med.resumen(), "nota_memoria": "memoria_pico_cpu_mb es el máximo del proceso entero; el tiempo es el de esta ejecución "
                                              "(si hubo reanudaciones, no incluye las ejecuciones anteriores)",
             "workers": ft["workers"], "hilos_cpu_torch": torch.get_num_threads(), "cpu_count": os.cpu_count()}
    escribir_json(dir_res / "resultados.json", datos)
    return datos


def cargar_base_entrenado(ruta: str | Path, n_clases: int) -> MobileNetV3SmallGAP:
    """Modelo completo desde un checkpoint de esta etapa (`mejor.pt`), en modo eval."""
    modelo = MobileNetV3SmallGAP(n_clases, pesos_imagenet=False)
    cargar_checkpoint(ruta, modelo, restaurar_generadores=False)
    return modelo.eval()
