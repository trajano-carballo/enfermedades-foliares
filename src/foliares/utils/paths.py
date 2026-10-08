"""Resolución de rutas desde configs/paths.yaml. Ninguna ruta absoluta en el código."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ENV_DATA_ROOT = "DATA_ROOT"


def raiz_repo(inicio: Path | None = None) -> Path:
    """Sube desde `inicio` (o desde este archivo) hasta encontrar configs/paths.yaml."""
    for base in ([Path(inicio)] if inicio else []) + [Path.cwd(), Path(__file__).resolve()]:
        for p in [base, *base.parents]:
            if (p / "configs" / "paths.yaml").is_file():
                return p
    raise FileNotFoundError("No se encontró configs/paths.yaml subiendo desde el directorio actual.")


@dataclass(frozen=True)
class Rutas:
    repo: Path
    data_root: Path
    raw: Path
    interim: Path
    processed: Path
    plantvillage: Path
    plantdoc: Path
    splits: Path

    def relativa(self, ruta: Path) -> str:
        """Ruta relativa a DATA_ROOT en formato posix: es lo que se guarda en los
        manifiestos para que sirvan en cualquier máquina."""
        return Path(ruta).resolve().relative_to(self.data_root.resolve()).as_posix()

    def absoluta(self, relativa: str) -> Path:
        return self.data_root / relativa


def cargar_rutas(repo: Path | None = None, entorno: dict | None = None) -> Rutas:
    repo = raiz_repo(repo)
    cfg = yaml.safe_load((repo / "configs" / "paths.yaml").read_text(encoding="utf-8"))
    entorno = os.environ if entorno is None else entorno

    def resolver(valor: str | Path, base: Path) -> Path:
        p = Path(valor).expanduser()
        return p if p.is_absolute() else base / p

    data_root = resolver(entorno.get(ENV_DATA_ROOT) or cfg["DATA_ROOT"], repo)
    return Rutas(
        repo=repo,
        data_root=data_root,
        raw=data_root / cfg["raw"],
        interim=data_root / cfg["interim"],
        processed=data_root / cfg["processed"],
        plantvillage=data_root / cfg["plantvillage"],
        plantdoc=data_root / cfg["plantdoc"],
        splits=resolver(cfg["SPLITS_DIR"], repo),
    )
