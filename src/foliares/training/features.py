"""Features GAP (576) del backbone congelado, calculadas una vez y cacheadas fuera del repo (decisión c de la sesión 8).

La clave del caché es el sha256 de todo lo que cambia las features: dataset, partición, rutas y etiquetas (que salen del CSV
congelado), sha256 del CSV de `data/splits/`, geometría y tamaño del preproceso, pesos del backbone, versión de torch,
torchvision y PIL, sha256 del código de `preproceso.py` y `mobilenet.py`, y tipo de dispositivo. Se guarda `features_<dataset>_<particion>_<clave>.pt` con su `.meta.json` (clave
completa, sha256 del archivo, tiempo de extracción). Se carga con `weights_only=True` y se verifica el sha256.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import PIL
import torch
import torchvision
from torch.utils.data import DataLoader

from foliares.data import preproceso as _preproceso
from foliares.models import mobilenet as _mobilenet

from foliares.data.congelar import sha256_archivo
from foliares.models.mobilenet import MobileNetV3SmallGAP
from foliares.utils.sesion import escribir_json


def congelar_backbone(modelo: MobileNetV3SmallGAP) -> MobileNetV3SmallGAP:
    """Backbone sin gradiente: solo el `Linear` del cabezal queda entrenable."""
    for p in modelo.features.parameters():
        p.requires_grad_(False)
    return modelo


@torch.inference_mode()
def extraer_features(modelo: MobileNetV3SmallGAP, dataset, batch: int, dispositivo: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(X, y): X (N, 576) float32 en CPU = GAP del último mapa, y (N,) int64. Modo eval y sin gradiente."""
    modelo.eval().to(dispositivo)
    xs, ys = [], []
    for x, y in DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0):
        xs.append(torch.flatten(modelo.gap(modelo.mapas(x.to(dispositivo))), 1).float().cpu())
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def clave_features(dataset_nombre: str, particion: str, rutas: list[str], etiquetas: list[int], sha256_csv: str,
                   geometria: str, tamano: int, pesos_imagenet: bool, dispositivo: str) -> dict:
    contenido = hashlib.sha256(json.dumps([rutas, etiquetas], ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"dataset": dataset_nombre, "particion": particion, "n": len(rutas), "sha256_rutas_y_etiquetas": contenido,
            "sha256_csv_splits": sha256_csv, "geometria": geometria, "tamano": tamano,
            "pesos": "IMAGENET1K_V1" if pesos_imagenet else "aleatorios", "torch": torch.__version__,
            "torchvision": torchvision.__version__, "pil": PIL.__version__, "dispositivo": dispositivo,
            "sha256_preproceso_py": sha256_archivo(Path(_preproceso.__file__)),
            "sha256_mobilenet_py": sha256_archivo(Path(_mobilenet.__file__))}


def _hash_clave(clave: dict) -> str:
    return hashlib.sha256(json.dumps(clave, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def features_con_cache(modelo, dataset, clave: dict, cache_dir: str | Path, batch: int, dispositivo: torch.device) -> dict:
    """Devuelve {X, y, desde_cache, tiempo_s, archivo, meta}. Si el caché existe y su clave y sha256 coinciden, lo usa."""
    cache_dir = Path(cache_dir)
    base = cache_dir / f"features_{clave['dataset']}_{clave['particion']}_{_hash_clave(clave)}"
    pt, meta_p = base.with_suffix(".pt"), base.with_suffix(".meta.json")
    if pt.is_file() and meta_p.is_file():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta["clave"] == clave and sha256_archivo(pt) == meta["sha256"]:
            d = torch.load(pt, map_location="cpu", weights_only=True)
            return {"X": d["X"], "y": d["y"], "desde_cache": True, "tiempo_s": 0.0, "archivo": str(pt), "meta": meta}
    t0 = time.perf_counter()
    X, y = extraer_features(modelo, dataset, batch, dispositivo)
    tiempo = round(time.perf_counter() - t0, 3)
    if not torch.equal(y, dataset.etiquetas):
        raise RuntimeError("Las etiquetas extraídas no coinciden con las del dataset")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = pt.with_name(pt.name + ".tmp")
    torch.save({"X": X, "y": y}, tmp)
    tmp.replace(pt)
    meta = {"clave": clave, "sha256": sha256_archivo(pt), "tiempo_extraccion_s": tiempo, "forma": list(X.shape),
            "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}
    escribir_json(meta_p, meta)
    return {"X": X, "y": y, "desde_cache": False, "tiempo_s": tiempo, "archivo": str(pt), "meta": meta}
