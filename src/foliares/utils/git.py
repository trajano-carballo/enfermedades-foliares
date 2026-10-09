"""Estado del repositorio del proyecto, para dejarlo registrado junto a cada resultado."""

from __future__ import annotations

import subprocess
from pathlib import Path

SIN_GIT = "(sin git)"


def git_salida(repo: str | Path, *args: str) -> str:
    """Salida de `git -C repo <args>`; `SIN_GIT` si no hay git o `repo` no es un repositorio."""
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return SIN_GIT


def info_repo(repo: str | Path) -> dict:
    """`commit` (HEAD) y `cambios_sin_commitear` (True/False; None si no hay git).
    Los datos (`data/`) están en .gitignore, así que no cuentan como cambios."""
    commit = git_salida(repo, "rev-parse", "HEAD")
    estado = git_salida(repo, "status", "--porcelain")
    return {"commit": commit, "cambios_sin_commitear": None if SIN_GIT in (commit, estado) else estado != ""}
