"""Pruebas de la lógica de congelamiento sobre directorios temporales (nunca toca data/splits/)."""

import json

import pandas as pd
import pytest

from foliares.data.congelar import congelar, sha256_archivo

CFG = {"version": "vtest", "commits_datasets": {"plantvillage": "a", "plantdoc": "b"},
       "fuentes": {"plantvillage": {"archivo": "pv.csv", "meta": "pv.json", "destino": "pv_final.csv"},
                   "plantdoc": {"archivo": "pd.csv", "meta": "pd.json", "destino": "pd_final.csv"}}}


@pytest.fixture
def dirs(tmp_path):
    i, s = tmp_path / "interim", tmp_path / "splits"
    i.mkdir()
    pd.DataFrame({"particion": ["train", "val", "test"]}).to_csv(i / "pv.csv", index=False)
    pd.DataFrame({"particion_propuesta": ["dev", "test", ""]}).to_csv(i / "pd.csv", index=False)
    (i / "pv.json").write_text(json.dumps({"semilla": 42}))
    (i / "pd.json").write_text(json.dumps({"revision_estado": "COMPLETA"}))
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
