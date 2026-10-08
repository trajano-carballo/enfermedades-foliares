"""Congelamiento de particiones: copia los candidatos de DATA_ROOT/interim a SPLITS_DIR con sha256
y un README de versión. Lo ejecuta el EQUIPO (scripts/congelar_splits.py) al cerrar la revisión humana.
Invariante: data/splits/ no se regenera tras la etapa 1 => nunca sobreescribe sin --force."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def sha256_archivo(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "(sin git)"


def planificar(cfg: dict, interim: Path, splits: Path, force: bool, permitir_provisional: bool) -> dict:
    """Valida todo ANTES de escribir. Devuelve el plan o levanta RuntimeError con el motivo."""
    problemas, plan, metas = [], [], {}
    for nombre, f in cfg["fuentes"].items():
        origen, destino = interim / f["archivo"], splits / f["destino"]
        if not origen.is_file():
            problemas.append(f"falta el candidato: {origen}")
            continue
        if destino.exists() and not force:
            problemas.append(f"ya existe {destino} (las particiones congeladas no se regeneran; --force solo con decisión del equipo)")
        meta = {}
        if f.get("meta"):
            mp = interim / f["meta"]
            if not mp.is_file():
                problemas.append(f"falta el meta del candidato: {mp}")
            else:
                meta = json.loads(mp.read_text(encoding="utf-8"))
        estado = str(meta.get("revision_estado", "COMPLETA"))
        if estado.startswith("PROVISIONAL") and not permitir_provisional:
            problemas.append(f"{nombre}: la revisión humana está {estado}; cerrarla o usar --permitir-provisional")
        metas[nombre] = meta
        plan.append({"nombre": nombre, "origen": origen, "destino": destino})
    if problemas:
        raise RuntimeError("No se congela:\n - " + "\n - ".join(problemas))
    return {"archivos": plan, "metas": metas}


def congelar(cfg: dict, repo: Path, interim: Path, splits: Path, force: bool = False,
             permitir_provisional: bool = False, dry_run: bool = False, hoy: str | None = None) -> dict:
    plan = planificar(cfg, interim, splits, force, permitir_provisional)
    sumas, resumen = {}, {}
    for a in plan["archivos"]:
        sumas[a["destino"].name] = sha256_archivo(a["origen"])
        df = pd.read_csv(a["origen"], low_memory=False)
        col = "particion" if "particion" in df else "particion_propuesta"
        resumen[a["nombre"]] = {"filas": len(df), "por_particion": df[col].fillna("").value_counts().to_dict()}
    if dry_run:
        return {"dry_run": True, "sha256": sumas, "resumen": resumen}
    splits.mkdir(parents=True, exist_ok=True)
    for a in plan["archivos"]:
        shutil.copyfile(a["origen"], a["destino"])
        assert sha256_archivo(a["destino"]) == sumas[a["destino"].name], "la copia no coincide con el origen"
    (splits / "SHA256SUMS.txt").write_text("".join(f"{h}  {n}\n" for n, h in sorted(sumas.items())), encoding="utf-8")
    estados = {k: v.get("revision_estado", "n/a") for k, v in plan["metas"].items()}
    sucio = _git(repo, "status", "--porcelain") not in ("", "(sin git)")
    lineas = [f"# Particiones congeladas — versión {cfg['version']}", "",
              f"- Fecha: {hoy or dt.date.today().isoformat()}",
              f"- Commit del repo al congelar: `{_git(repo, 'rev-parse', 'HEAD')}` (cambios sin commitear: {'sí' if sucio else 'no'})",
              "- Datasets (commits fijados): " + ", ".join(f"{k} `{v}`" for k, v in cfg["commits_datasets"].items()),
              f"- Estado de la revisión humana de etiquetas (PlantDoc): {estados.get('plantdoc', 'n/a')}", "",
              "## Archivos y sha256", ""]
    lineas += [f"- `{n}`: `{h}`" for n, h in sorted(sumas.items())]
    lineas += ["", "## Conteos por partición", ""]
    for k, v in resumen.items():
        lineas.append(f"- {k}: {v['filas']} filas — " + ", ".join(f"{p or '(sin partición)'}: {n}" for p, n in sorted(v["por_particion"].items())))
    lineas += ["", "Metadatos de generación (semilla, zona, tope, fracciones):", "", "```json",
               json.dumps(plan["metas"], indent=2, ensure_ascii=False), "```", "",
               "Estos archivos NO se regeneran tras la etapa 1. Un cambio requiere decisión del equipo y nueva versión.", ""]
    (splits / "README_VERSION.md").write_text("\n".join(lineas), encoding="utf-8")
    return {"dry_run": False, "sha256": sumas, "resumen": resumen}
