"""Guarda de acceso a los tests (decisión e de la sesión 9, `docs/protocolo_evaluacion.md`).

Antes de evaluar sobre un test, la fila de `docs/bitacora/accesos_test.md` tiene que existir y estar COMMITEADA. Se exige:
- árbol limpio (lo que se evalúa es exactamente `HEAD`);
- la fila `N` en la versión de `HEAD` del archivo, con el test pedido (`PV-test` o `PD-test`) en la columna Test;
- la config que se usa mencionada en la columna "Modelo / config";
- el commit de la fila es un ancestro de `HEAD` y desde él no cambiaron la config, `src/` ni `scripts/` (la fila se escribe
  con el commit del código; el commit que agrega la fila viene después y solo puede tocar documentación).
Si algo no se cumple, levanta `AccesoNoRegistrado` y no se evalúa nada.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ARCHIVO = "docs/bitacora/accesos_test.md"
TESTS = ("PV-test", "PD-test")


class AccesoNoRegistrado(RuntimeError):
    """La evaluación sobre un test no tiene su fila de acceso commiteada y coherente."""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", check=check)


def filas_de_la_tabla(texto: str) -> dict[int, dict]:
    """Filas de la tabla `| # | Fecha | Test | Modelo / config | Commit | Motivo |` (las que empiezan con un número)."""
    filas = {}
    for linea in texto.splitlines():
        celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
        if linea.lstrip().startswith("|") and len(celdas) >= 6 and celdas[0].isdigit():
            filas[int(celdas[0])] = {"fecha": celdas[1], "test": celdas[2], "modelo_config": celdas[3],
                                     "commit": celdas[4].strip("`").strip(), "motivo": " | ".join(celdas[5:])}
    return filas


def exigir_fila_accesos(repo: str | Path, fila: int, test: str, config_rel: str) -> dict:
    """Verifica la fila `fila` para evaluar `test` con `config_rel` (ruta relativa al repo). Devuelve la fila y `HEAD`."""
    repo = Path(repo)
    if test not in TESTS:
        raise AccesoNoRegistrado(f"Test desconocido: {test!r}; opciones: {TESTS}")
    if _git(repo, "rev-parse", "--is-inside-work-tree", check=False).returncode:
        raise AccesoNoRegistrado(f"{repo} no es un repositorio git: no se puede verificar la fila de acceso")
    sucio = _git(repo, "status", "--porcelain").stdout.strip()
    if sucio:
        raise AccesoNoRegistrado(f"El árbol tiene cambios sin commitear; se evalúa solo desde un HEAD limpio:\n{sucio}")
    en_head = _git(repo, "show", f"HEAD:{ARCHIVO}", check=False)
    if en_head.returncode:
        raise AccesoNoRegistrado(f"{ARCHIVO} no está en HEAD")
    filas = filas_de_la_tabla(en_head.stdout)
    if fila not in filas:
        raise AccesoNoRegistrado(f"La fila {fila} no está en {ARCHIVO} de HEAD (filas: {sorted(filas)}); agregarla y commitearla antes")
    f = filas[fila]
    if f["test"] != test:
        raise AccesoNoRegistrado(f"La fila {fila} es de {f['test']!r}, no de {test!r}")
    if config_rel not in f["modelo_config"]:
        raise AccesoNoRegistrado(f"La fila {fila} no menciona la config {config_rel!r} (dice {f['modelo_config']!r})")
    if not re.fullmatch(r"[0-9a-f]{7,40}", f["commit"]):
        raise AccesoNoRegistrado(f"La fila {fila} no tiene un commit válido: {f['commit']!r}")
    if _git(repo, "rev-parse", "--verify", "--quiet", f"{f['commit']}^{{commit}}", check=False).returncode:
        raise AccesoNoRegistrado(f"El commit {f['commit']} de la fila {fila} no existe en el repo")
    if _git(repo, "merge-base", "--is-ancestor", f["commit"], "HEAD", check=False).returncode:
        raise AccesoNoRegistrado(f"El commit {f['commit']} de la fila {fila} no es un ancestro de HEAD")
    cambios = _git(repo, "diff", "--name-only", f["commit"], "HEAD", "--", config_rel, "src", "scripts").stdout.split()
    if cambios:
        raise AccesoNoRegistrado(f"Desde el commit {f['commit']} de la fila {fila} cambiaron la config o el código: {cambios}")
    return {**f, "fila": fila, "head": _git(repo, "rev-parse", "HEAD").stdout.strip()}
