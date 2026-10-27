"""Registro de corrida/sesión: commit, splits, semillas, entorno, tiempo y memoria pico.

CLAUDE.md: "cada resultado en `results/` debe poder rastrearse a una config y un commit"; sin este registro una corrida
no es un resultado. El estado de git se captura UNA vez al empezar la corrida (antes de escribir salidas): las salidas
sin seguimiento dentro del repo contarían como "cambios sin commitear" (mismo problema que tuvo `congelar.py`).
Por eso las salidas se escriben fuera del repo (en Colab, en Drive) o se captura el estado antes de escribir.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

from foliares.data.congelar import sha256_archivo
from foliares.utils.archivos import escribir_texto
from foliares.utils.git import info_repo


def nuevo_run_id(prefijo: str) -> str:
    return f"{prefijo}_{time.strftime('%Y%m%d_%H%M%S')}"


def sha256_splits(splits_dir: str | Path) -> dict[str, str]:
    """sha256 de cada CSV de `data/splits/` (las particiones congeladas con las que corrió la corrida)."""
    return {p.name: sha256_archivo(p) for p in sorted(Path(splits_dir).glob("*.csv"))}


def info_entorno() -> dict:
    """Versión de torch/CUDA, dispositivo, GPU e hilos de CPU (ver `docs/reproducibilidad.md`)."""
    import torch

    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "dispositivo": "cuda" if cuda else "cpu",
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "hilos_cpu": torch.get_num_threads(),
    }


def memoria_pico_cpu_mb() -> float | None:
    """Pico de memoria residente del proceso en MiB (None si la plataforma no lo informa)."""
    if sys.platform.startswith("linux"):
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # en Linux ru_maxrss está en KiB
    try:
        import psutil

        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2**20
    except Exception:  # noqa: BLE001
        return None


class Medidor:
    """Mide tiempo y memoria pico (GPU y CPU) de un bloque: `with Medidor() as m: ...; m.resumen()`."""

    def __enter__(self) -> "Medidor":
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        self.tiempo_s = 0.0
        return self

    def __exit__(self, *exc) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.tiempo_s = time.perf_counter() - self._t0

    def resumen(self) -> dict:
        import torch

        gpu = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        return {"tiempo_s": round(self.tiempo_s, 3),
                "memoria_pico_gpu_mb": None if gpu is None else round(gpu, 1),
                "memoria_pico_cpu_mb": None if (cpu := memoria_pico_cpu_mb()) is None else round(cpu, 1)}


def registro_base(repo: str | Path, splits_dir: str | Path, config: dict, semillas: dict) -> dict:
    """Lo que toda corrida registra: fecha, commit, cambios sin commitear, config usada, sha256 de los splits,
    semillas y entorno. Llamar ANTES de escribir cualquier salida dentro del repo."""
    git = info_repo(repo)
    return {"fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": git["commit"],
            "cambios_sin_commitear": git["cambios_sin_commitear"], "config": config,
            "sha256_splits": sha256_splits(splits_dir), "semillas": semillas, **info_entorno()}


def escribir_json(ruta: str | Path, datos: dict) -> Path:
    """JSON con LF y UTF-8 (igual en Windows, Linux y Colab)."""
    return escribir_texto(ruta, json.dumps(datos, indent=2, ensure_ascii=False) + "\n")
