"""Registro de corrida: lo que exige CLAUDE.md para que una corrida cuente como resultado."""

import json

from foliares.data.congelar import sha256_archivo
from foliares.utils.sesion import Medidor, escribir_json, registro_base, sha256_splits

CAMPOS = {"fecha", "commit", "cambios_sin_commitear", "config", "sha256_splits", "semillas", "torch", "dispositivo",
          "python", "hilos_cpu"}


def test_registro_base_trae_config_commit_splits_semillas_y_entorno(pv_sintetico, tmp_path):
    splits, _ = pv_sintetico
    r = registro_base(tmp_path, splits, {"lr": 0.1}, {"semilla": 42})
    assert CAMPOS <= set(r) and r["config"] == {"lr": 0.1} and r["semillas"] == {"semilla": 42}
    assert r["sha256_splits"] == {"plantvillage_particion.csv": sha256_archivo(splits / "plantvillage_particion.csv")}
    assert r["dispositivo"] in ("cpu", "cuda")


def test_sha256_splits_solo_csv(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    (tmp_path / "README_VERSION.md").write_text("y")
    assert list(sha256_splits(tmp_path)) == ["a.csv"]


def test_medidor_informa_tiempo_y_memoria():
    with Medidor() as m:
        sum(range(1000))
    r = m.resumen()
    assert r["tiempo_s"] >= 0 and "memoria_pico_gpu_mb" in r and "memoria_pico_cpu_mb" in r


def test_json_con_lf_y_utf8(tmp_path):
    escribir_json(tmp_path / "x" / "r.json", {"motivo": "revisión"})
    b = (tmp_path / "x" / "r.json").read_bytes()
    assert b"\r" not in b and json.loads(b.decode("utf-8")) == {"motivo": "revisión"}
