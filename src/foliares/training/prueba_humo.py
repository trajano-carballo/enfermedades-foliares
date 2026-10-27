"""Prueba de humo de Colab en dos fases (cada una en un proceso nuevo = reinicio real del estado).

Fase `entrenar`: MobileNetV3-Small (cabezal GAP->Linear) con pesos de ImageNet, un lote de PlantVillage-TRAIN, unos pasos de
SGD, checkpoint en `ckpt_dir` (Drive en Colab) y unos pasos más (la corrida continua, como control). Fase `reanudar`: proceso
nuevo, modelo SIN pesos de ImageNet y semilla distinta a propósito (todo debe salir del checkpoint), carga el checkpoint y
repite los pasos finales: las pérdidas y los pesos tienen que coincidir con la corrida continua. Cada fase escribe su JSON
(`registro_base` + resultados + tiempo y memoria pico) en `salida/<experimento>/<run_id>/`.

No toca PlantDoc ni ningún test (ver `foliares.data.lote_humo`). No es una corrida de entrenamiento: no mide desempeño.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from foliares.data.congelar import sha256_archivo
from foliares.data.lote_humo import cargar_lote, clases_eval
from foliares.models.mobilenet import MobileNetV3SmallGAP
from foliares.training.checkpoint import cargar_checkpoint, guardar_checkpoint
from foliares.utils.seeds import fijar_semillas
from foliares.utils.sesion import Medidor, escribir_json, registro_base

NOMBRE_CKPT = "ckpt_humo.pt"
JSON_FASE1, JSON_FASE2 = "fase1_entrenar.json", "fase2_reanudar.json"


def hash_pesos(modelo: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in modelo.state_dict().items():
        h.update(k.encode("utf-8"))
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _dispositivo() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _armar(cfg: dict, n_clases: int, pesos_imagenet: bool, dev: torch.device):
    modelo = MobileNetV3SmallGAP(n_clases, pesos_imagenet=pesos_imagenet).to(dev).train()
    return modelo, torch.optim.SGD(modelo.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])


def _paso(modelo, opt, x, y) -> float:
    opt.zero_grad(set_to_none=True)
    perdida = F.cross_entropy(modelo(x), y)
    perdida.backward()
    opt.step()
    return perdida.item()


def _dir_corrida(salida: str | Path, cfg: dict, run_id: str) -> Path:
    return Path(salida) / cfg["experimento"] / run_id


def sonda_determinismo(cfg: dict, x: torch.Tensor, y: torch.Tensor, n_clases: int, dev: torch.device) -> dict:
    """Corre dos veces los mismos pasos desde la misma semilla (sin pesos de ImageNet: no depende de la red) y compara.
    Con GPU muestra empíricamente lo que `docs/reproducibilidad.md` dice que no se puede garantizar."""
    resultado = {}
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    modo = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    for determinista in (False, True):
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            corridas = []
            for _ in range(2):
                fijar_semillas(cfg["semilla"], determinista=determinista)
                modelo, opt = _armar(cfg, n_clases, False, dev)
                perdidas = [_paso(modelo, opt, x, y) for _ in range(cfg["pasos_sonda"])]
                corridas.append((perdidas, [p.detach().clone() for p in modelo.parameters()]))
        (p1, w1), (p2, w2) = corridas
        dif = max((a - b).abs().max().item() for a, b in zip(w1, w2))
        resultado[f"determinista_{determinista}"] = {
            "igual_bit_a_bit": dif == 0.0 and p1 == p2, "max_abs_dif_pesos": dif,
            "max_abs_dif_perdidas": max(abs(a - b) for a, b in zip(p1, p2)),
            "avisos_de_torch": sorted({str(a.message).split("\n")[0][:160] for a in avisos})[:3]}
    torch.use_deterministic_algorithms(modo[0], warn_only=modo[1])  # la sonda deja el modo global como lo encontró
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn
    return resultado


def fase_entrenar(cfg: dict, repo: Path, splits_dir: Path, data_root: Path, salida: Path, ckpt_dir: Path, run_id: str) -> dict:
    registro = registro_base(repo, splits_dir, cfg, {})  # estado de git capturado antes de escribir nada
    registro["semillas"] = fijar_semillas(cfg["semilla"], cfg["determinista"])
    dev = _dispositivo()
    x, y, rutas = cargar_lote(splits_dir, data_root, cfg["n_imagenes"], cfg["semilla"], cfg["tamano"])
    x, y = x.to(dev), y.to(dev)
    n_clases = len(clases_eval(splits_dir))
    modelo, opt = _armar(cfg, n_clases, cfg["pesos_imagenet"], dev)
    with Medidor() as med:
        antes = [_paso(modelo, opt, x, y) for _ in range(cfg["pasos_antes"])]
        ck = guardar_checkpoint(Path(ckpt_dir) / NOMBRE_CKPT, modelo, opt, cfg["pasos_antes"], extra={"run_id": run_id})
        hash_en_ck = hash_pesos(modelo)
        continuas = [_paso(modelo, opt, x, y) for _ in range(cfg["pasos_despues"])]
        hash_final = hash_pesos(modelo)
    datos = {**registro, "fase": "entrenar", "run_id": run_id, "uso_de_datos": "plantvillage_particion.csv, particion == train",
             "n_imagenes": len(rutas), "rutas_lote": rutas, "perdidas_antes": antes, "checkpoint": ck,
             "hash_pesos_en_checkpoint": hash_en_ck, "perdidas_continuas": continuas, "hash_pesos_final": hash_final,
             **med.resumen()}
    if cfg.get("sonda_determinismo"):
        datos["sonda_determinismo"] = sonda_determinismo(cfg, x, y, n_clases, dev)
    escribir_json(_dir_corrida(salida, cfg, run_id) / JSON_FASE1, datos)
    return datos


def fase_reanudar(cfg: dict, repo: Path, splits_dir: Path, data_root: Path, salida: Path, ckpt_dir: Path, run_id: str) -> dict:
    registro = registro_base(repo, splits_dir, cfg, {})
    dir_corrida = _dir_corrida(salida, cfg, run_id)
    previo = json.loads((dir_corrida / JSON_FASE1).read_text(encoding="utf-8"))
    ruta_ck = Path(ckpt_dir) / NOMBRE_CKPT
    sha_ck = sha256_archivo(ruta_ck)
    registro["semillas"] = fijar_semillas(cfg["semilla"] + 1, cfg["determinista"])  # distinta a propósito
    dev = _dispositivo()
    x, y, rutas = cargar_lote(splits_dir, data_root, cfg["n_imagenes"], cfg["semilla"], cfg["tamano"])
    x, y = x.to(dev), y.to(dev)
    modelo, opt = _armar(cfg, len(clases_eval(splits_dir)), False, dev)  # sin pesos de ImageNet: todo sale del checkpoint
    with Medidor() as med:
        info = cargar_checkpoint(ruta_ck, modelo, opt)  # se lee a CPU; load_state_dict copia a la GPU
        hash_cargado = hash_pesos(modelo)
        reanudadas = [_paso(modelo, opt, x, y) for _ in range(cfg["pasos_despues"])]
        hash_final = hash_pesos(modelo)
    dif = max(abs(a - b) for a, b in zip(reanudadas, previo["perdidas_continuas"]))
    comprobaciones = {
        "checkpoint_sha256_igual_al_de_la_fase_1": sha_ck == previo["checkpoint"]["sha256"],
        "paso_del_checkpoint_correcto": info["paso"] == cfg["pasos_antes"],
        "pesos_cargados_iguales_a_los_guardados": hash_cargado == previo["hash_pesos_en_checkpoint"],
        "mismo_lote_que_la_fase_1": rutas == previo["rutas_lote"],
        "perdidas_dentro_de_tolerancia": dif <= cfg["tolerancia_perdida"]}
    datos = {**registro, "fase": "reanudar", "run_id": run_id, "perdidas_reanudadas": reanudadas,
             "perdidas_continuas_fase_1": previo["perdidas_continuas"], "max_abs_dif_perdidas": dif,
             "igual_bit_a_bit": reanudadas == previo["perdidas_continuas"] and hash_final == previo["hash_pesos_final"],
             "hash_pesos_final": hash_final, "comprobaciones": comprobaciones, "reanudacion_ok": all(comprobaciones.values()),
             **med.resumen()}
    escribir_json(dir_corrida / JSON_FASE2, datos)
    return datos
