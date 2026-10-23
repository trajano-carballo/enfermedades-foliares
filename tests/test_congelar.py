"""Pruebas de la lógica de congelamiento sobre directorios temporales (nunca toca data/splits/)."""

import json
import shutil
import subprocess

import pandas as pd
import pytest

from foliares.data.congelar import congelar, sha256_archivo

CFG = {"version": "vtest", "commits_datasets": {"plantvillage": "a", "plantdoc": "b"},
       "fuentes": {"plantvillage": {"archivo": "pv.csv", "meta": "pv.json", "destino": "pv_final.csv", "requiere_revision": False},
                   "plantdoc": {"archivo": "pd.csv", "meta": "pd.json", "destino": "pd_final.csv"}}}


@pytest.fixture
def dirs(tmp_path):
    i, s = tmp_path / "interim", tmp_path / "splits"
    i.mkdir()
    pd.DataFrame({"particion": ["train", "val", "test"]}).to_csv(i / "pv.csv", index=False)
    pd.DataFrame({"particion_propuesta": ["dev", "test", ""]}).to_csv(i / "pd.csv", index=False)
    (i / "pv.json").write_text(json.dumps({"semilla": 42}))
    (i / "pd.json").write_text(json.dumps({"revision_estado": "COMPLETO"}))
    return tmp_path, i, s


def test_congela_con_sha256_y_readme(dirs):
    repo, i, s = dirs
    r = congelar(CFG, repo, i, s)
    assert sha256_archivo(s / "pv_final.csv") == r["sha256"]["pv_final.csv"] == sha256_archivo(i / "pv.csv")
    assert "pv_final.csv" in (s / "SHA256SUMS.txt").read_text()
    assert "versión vtest" in (s / "README_VERSION.md").read_text(encoding="utf-8")


def test_dry_run_no_escribe(dirs):
    repo, i, s = dirs
    assert congelar(CFG, repo, i, s, dry_run=True)["dry_run"] and not s.exists()


def test_no_sobreescribe_ni_congela_provisional(dirs):
    repo, i, s = dirs
    congelar(CFG, repo, i, s)
    with pytest.raises(RuntimeError, match="ya existe"):
        congelar(CFG, repo, i, s)
    (i / "pd.json").write_text(json.dumps({"revision_estado": "PROVISIONAL (124 filas)"}))
    with pytest.raises(RuntimeError, match="PROVISIONAL"):
        congelar(CFG, repo, i, s, force=True)
    congelar(CFG, repo, i, s, force=True, permitir_provisional=True)


def test_meta_sin_estado_de_revision_se_trata_como_provisional(dirs):
    repo, i, s = dirs
    (i / "pd.json").write_text(json.dumps({"semilla": 42}))                 # el .meta.json no trae `revision_estado`
    with pytest.raises(RuntimeError, match="PROVISIONAL.*no trae"):
        congelar(CFG, repo, i, s)
    assert not s.exists()                                                   # validó antes de escribir
    congelar(CFG, repo, i, s, permitir_provisional=True)


def test_solo_COMPLETO_habilita_el_congelamiento(dirs):
    repo, i, s = dirs
    for estado in ("COMPLETA", "completo", "PROVISIONAL (1 de 124 filas...)", ""):
        (i / "pd.json").write_text(json.dumps({"revision_estado": estado}))
        with pytest.raises(RuntimeError, match="la revisión humana está"):
            congelar(CFG, repo, i, s, dry_run=True)


def test_fuente_sin_revision_no_la_exige(dirs):
    # PlantVillage (`requiere_revision: False`) congela sin `revision_estado`; si no se declara, se exige por defecto
    repo, i, s = dirs
    assert congelar(CFG, repo, i, s, dry_run=True)["dry_run"]
    cfg = {**CFG, "fuentes": {**CFG["fuentes"], "plantvillage": {k: v for k, v in CFG["fuentes"]["plantvillage"].items() if k != "requiere_revision"}}}
    with pytest.raises(RuntimeError, match="plantvillage: la revisión humana"):
        congelar(cfg, repo, i, s, dry_run=True)


def test_archivos_que_escribe_congelar_salen_con_lf(dirs):
    repo, i, s = dirs
    congelar(CFG, repo, i, s)
    for nombre in ("SHA256SUMS.txt", "README_VERSION.md"):
        assert b"\r" not in (s / nombre).read_bytes(), nombre


# ------------------------------------------------------------------ flag "cambios sin commitear" del README
def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True, capture_output=True)


@pytest.fixture
def repo_git(tmp_path):
    """Repo git temporal con la misma disposición que el real: `interim/` ignorado y `splits/` versionable."""
    if shutil.which("git") is None:
        pytest.skip("git no está disponible")
    i, s = tmp_path / "interim", tmp_path / "splits"
    i.mkdir()
    pd.DataFrame({"particion": ["train", "val", "test"]}).to_csv(i / "pv.csv", index=False)
    pd.DataFrame({"particion_propuesta": ["dev", "test", ""]}).to_csv(i / "pd.csv", index=False)
    (i / "pv.json").write_text(json.dumps({"semilla": 42}))
    (i / "pd.json").write_text(json.dumps({"revision_estado": "COMPLETO"}))
    (tmp_path / ".gitignore").write_text("interim/\n")
    (tmp_path / "codigo.py").write_text("x = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".gitignore", "codigo.py")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path, i, s


def test_readme_arbol_limpio_dice_no_aunque_el_script_escriba_en_splits(repo_git):
    # Los archivos que escribe el propio script (CSV y SHA256SUMS.txt, sin seguimiento en git) no son "cambios".
    repo, i, s = repo_git
    congelar(CFG, repo, i, s)
    assert "(cambios sin commitear: no)" in (s / "README_VERSION.md").read_text(encoding="utf-8")


def test_readme_con_cambio_real_en_archivo_versionado_dice_si(repo_git):
    repo, i, s = repo_git
    (repo / "codigo.py").write_text("x = 2\n")
    congelar(CFG, repo, i, s)
    assert "(cambios sin commitear: sí)" in (s / "README_VERSION.md").read_text(encoding="utf-8")
