"""Sesión 11: calibración por temperatura, ECE, particiones fuera de muestra, abstención y guardas (§g del protocolo)."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from foliares.eval import calibracion as C
from foliares.eval.guarda_accesos import AccesoNoRegistrado
from foliares.utils.paths import cargar_rutas

REPO = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO / "configs" / "05_calibracion.yaml").read_text(encoding="utf-8"))
CFG_MODELO = yaml.safe_load((REPO / "configs" / "02_modelo_base.yaml").read_text(encoding="utf-8"))


def _datos(n_por_clase=20, k=12, semilla=0):
    """Logits sobreconfiados: margen grande y la mitad de las predicciones mal."""
    rng = np.random.default_rng(semilla)
    y = np.repeat(np.arange(k), n_por_clase)
    pred = np.where(rng.random(len(y)) < 0.5, y, (y + 1 + rng.integers(0, k - 1, len(y))) % k)
    z = rng.normal(0, 1, (len(y), k))
    z[np.arange(len(y)), pred] += 8
    return z.astype(np.float32), y


def test_T_1_deja_los_logits_igual():
    z, y = _datos()
    np.testing.assert_allclose(C.softmax_t(z, 1.0), torch.softmax(torch.tensor(z, dtype=torch.float64), 1).numpy(), atol=1e-12)
    ce = torch.nn.functional.cross_entropy(torch.tensor(z, dtype=torch.float64), torch.tensor(y)).item()
    assert C.nll(z, y, 1.0) == pytest.approx(ce, abs=1e-10)


@pytest.mark.parametrize("T", [0.5, 0.73, 2.0, 9.95])
def test_el_argmax_no_cambia_con_T(T):
    z, _ = _datos()
    assert (C.softmax_t(z, T).argmax(1) == z.argmax(1)).all()


def test_ece_de_un_caso_armado_a_mano():
    p = np.array([[0.95, 0.05], [0.95, 0.05], [0.55, 0.45], [0.55, 0.45]])
    y = np.array([0, 1, 0, 0])
    # bin (0,9333; 1]: confianza 0,95, acierto 0,5 → 0,45·2/4; bin (0,5333; 0,6]: 0,55 contra 1 → 0,45·2/4
    assert C.ece(p, y, 15) == pytest.approx(0.45)
    assert C.ece(np.array([[1.0, 0.0], [1.0, 0.0]]), np.array([0, 0]), 15) == 0


def test_metricas_del_sistema_a_mano():
    p = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.55, 0.45]])
    y = np.array([0, 1, 0, 0, 1])
    # p máx: 0,9 ok, 0,8 mal, 0,6 ok, 0,6 mal (pred 1), 0,55 mal → todas ≥ 0,5
    m = C.metricas_sistema(y, p, 0.65, 15)
    assert m["cobertura"] == pytest.approx(0.4) and m["abstencion"] == pytest.approx(0.6)
    assert m["acierto_respondidas"] == pytest.approx(0.5) and m["errores_alta_confianza"] == pytest.approx(0.2)
    assert np.isnan(C.metricas_sistema(y, p, 0.99, 15)["acierto_respondidas"])


def test_la_temperatura_corrige_la_sobreconfianza_y_frena_en_el_borde():
    z, y = _datos()
    g = C.grilla_temperatura(CFG["temperatura_grilla"])
    assert g[0] == 0.5 and g[-1] == 10.0 and len(g) == 191
    T = C.ajustar_temperatura(z, y, g)
    assert 1 < T < 10 and C.nll(z, y, T) < C.nll(z, y, 1.0)
    with pytest.raises(C.TemperaturaEnBorde):
        C.ajustar_temperatura(z, y, g[g <= 1.0])     # el óptimo verdadero está fuera: cae en el borde superior


def test_las_particiones_son_disjuntas_cubren_todo_y_estan_estratificadas():
    _, y = _datos(n_por_clase=37)
    f = C.particiones_estratificadas(y, 5, 42)
    assert set(f) == set(range(5)) and len(f) == len(y)      # cada imagen en exactamente una
    for k in range(5):
        cuenta = np.bincount(y[f == k], minlength=12)
        assert cuenta.min() >= 7 and cuenta.max() <= 8
    assert (f == C.particiones_estratificadas(y, 5, 42)).all()


def test_ninguna_imagen_se_calibra_con_una_T_ajustada_con_ella(monkeypatch):
    z, y = _datos()
    z[:, 0] = np.arange(len(z))                               # identifica cada imagen por su fila
    vistas = []
    real = C.ajustar_temperatura
    monkeypatch.setattr(C, "ajustar_temperatura", lambda zz, yy, g: vistas.append(set(zz[:, 0].astype(int))) or 1.5)
    f = C.particiones_estratificadas(y, 5, 42)
    p, ts = C.calibrar_fuera_de_particion(z, y, f, C.grilla_temperatura(CFG["temperatura_grilla"]))
    assert len(vistas) == 5 and ts == [1.5] * 5
    for k, ajuste in enumerate(vistas):
        assert ajuste.isdisjoint(np.flatnonzero(f == k)) and ajuste | set(np.flatnonzero(f == k)) == set(range(len(y)))
    monkeypatch.setattr(C, "ajustar_temperatura", real)
    np.testing.assert_allclose(p, C.softmax_t(z, 1.5))


def test_regla_1_en_el_repo_elige_la_variante_y_empate_da_el_base(tmp_path):
    r = C.regla_modelo(REPO, CFG["candidatos"])
    assert r["elegido"] == "variante" and r["run_id"] == CFG["modelo_base"]["run_id"]
    assert r["macro_f1_pv_val"]["base"] == pytest.approx(0.95589, abs=1e-5) and r["macro_f1_pv_val"]["variante"] == pytest.approx(0.96584, abs=1e-5)
    for k in ("a", "b"):
        (tmp_path / f"{k}.json").write_text(json.dumps({"pv_val": {"macro_f1": 0.9}}))
    empate = {"base": {"run_id": "B", "resultados": "a.json"}, "variante": {"run_id": "V", "resultados": "b.json"}}
    assert C.regla_modelo(tmp_path, empate)["elegido"] == "base"


class _DSFalso:
    leidos: list = []

    def __init__(self, splits, data_root, dataset, particion, *a, **k):
        _DSFalso.leidos.append((dataset, particion))
        z, y = _datos()
        self.filas = pd.DataFrame({"ruta": [f"i{i}.jpg" for i in range(len(y))], "clase": "c", "etiqueta": y})
        self.z, self.y = z, y


def _falsos(monkeypatch):
    _DSFalso.leidos = []
    monkeypatch.setattr(C, "DatasetParticion", _DSFalso)
    monkeypatch.setattr(C, "cargar_modelo_base", lambda *a: None)
    monkeypatch.setattr(C, "predecir", lambda m, ds, b, d: {"logits": ds.z, "y": ds.y})
    monkeypatch.setattr(C, "_copiar_al_repo", lambda *a: None)


def test_la_fase_dev_solo_lee_plantdoc_dev_y_registra_todo(monkeypatch, tmp_path):
    _falsos(monkeypatch)
    cfg = {**CFG, "bootstrap": {**CFG["bootstrap"], "replicas": 50}}
    R = cargar_rutas()
    d = C.calibrar_dev(cfg, CFG_MODELO, REPO, R.splits, R.data_root, tmp_path, "r")
    assert _DSFalso.leidos == [("plantdoc", "dev")]
    assert 1 < d["temperatura_final"] < 10 and len(d["temperatura_por_particion"]) == 5
    assert d["despues_fuera_de_particion"]["ece"]["estimacion"] < d["antes_T_1"]["ece"]["estimacion"]
    for clave in ("commit", "cambios_sin_commitear", "config", "sha256_splits", "semillas", "torch", "dispositivo", "tiempo_s",
                  "memoria_pico_cpu_mb", "regla_modelo"):
        assert clave in d
    assert (tmp_path / "05_calibracion" / "r" / "calibracion_dev.json").is_file()
    assert (tmp_path / "05_calibracion" / "r" / "logits_pd_dev.csv").is_file()


def test_la_fase_dev_frena_si_la_config_no_calibra_el_modelo_de_la_regla(monkeypatch, tmp_path):
    _falsos(monkeypatch)
    cfg = {**CFG, "modelo_base": {**CFG["modelo_base"], "run_id": CFG["candidatos"]["base"]["run_id"]}}
    R = cargar_rutas()
    with pytest.raises(ValueError, match="regla 1"):
        C.calibrar_dev(cfg, CFG_MODELO, REPO, R.splits, R.data_root, tmp_path, "r")
    assert _DSFalso.leidos == []


def _repo_git(tmp_path, dev_json=None):
    if shutil.which("git") is None:
        pytest.skip("git no está disponible")
    g = lambda *a: subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)  # noqa: E731
    (tmp_path / "docs" / "bitacora").mkdir(parents=True)
    (tmp_path / "docs" / "bitacora" / "accesos_test.md").write_text("| # | Fecha | Test | Modelo / config | Commit | Motivo |\n|---|---|---|---|---|---|\n")
    if dev_json:
        (tmp_path / "results" / "05_calibracion" / "rd").mkdir(parents=True)
        (tmp_path / "results" / "05_calibracion" / "rd" / "calibracion_dev.json").write_text(json.dumps(dev_json))
    g("init", "-q")
    g("add", ".")
    g("commit", "-q", "-m", "c1")
    return tmp_path


def test_la_fase_test_sin_fila_valida_no_evalua(monkeypatch, tmp_path):
    _falsos(monkeypatch)
    repo = _repo_git(tmp_path / "repo")
    R = cargar_rutas()
    with pytest.raises(AccesoNoRegistrado):
        C.calibrar_test({**CFG, "temperatura": 2.0, "run_dev": "rd"}, CFG_MODELO, "configs/05_calibracion.yaml", repo, R.splits,
                        R.data_root, tmp_path / "salida", "r", 5)
    assert _DSFalso.leidos == []


def test_la_fase_test_solo_lee_plantdoc_test_con_la_T_de_la_config(monkeypatch, tmp_path):
    _falsos(monkeypatch)
    repo = _repo_git(tmp_path / "repo", {"temperatura_final": 2.0, "modelo_base": CFG["modelo_base"], "tau": CFG["tau"]})
    monkeypatch.setattr(C, "exigir_fila_accesos", lambda *a: {"fila": 5})
    R = cargar_rutas()
    cfg = {**CFG, "temperatura": 2.0, "run_dev": "rd", "bootstrap": {**CFG["bootstrap"], "replicas": 50}}
    d = C.calibrar_test(cfg, CFG_MODELO, "configs/05_calibracion.yaml", repo, R.splits, R.data_root, tmp_path / "salida", "r", 5)
    assert _DSFalso.leidos == [("plantdoc", "test")] and d["accesos"]["PD-test"]["fila"] == 5 and d["temperatura"] == 2.0
    with pytest.raises(ValueError, match="fase dev"):
        C.calibrar_test({**cfg, "temperatura": 2.05}, CFG_MODELO, "configs/05_calibracion.yaml", repo, R.splits, R.data_root,
                        tmp_path / "salida", "r2", 5)


def test_nada_de_la_etapa_5_lee_plantvillage():
    for f in (REPO / "src" / "foliares" / "eval" / "calibracion.py", REPO / "scripts" / "calibrar.py"):
        assert "plantvillage" not in f.read_text(encoding="utf-8").lower()


def test_el_script_exige_la_fila_en_la_fase_test(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "calibrar.py"), "--config", str(REPO / "configs" / "05_calibracion.yaml"),
                        "--salida", str(tmp_path), "--fase", "test"], capture_output=True, text=True, cwd=REPO,
                       env={**os.environ, "DATA_ROOT": str(tmp_path)})
    assert r.returncode == 2 and "--fila-accesos" in r.stderr


def test_las_figuras_de_la_sesion_11_se_generan(monkeypatch, tmp_path):
    from foliares.eval import figuras as FIG
    _falsos(monkeypatch)
    R = cargar_rutas()
    d = C.calibrar_dev({**CFG, "bootstrap": {**CFG["bootstrap"], "replicas": 20}}, CFG_MODELO, REPO, R.splits, R.data_root, tmp_path, "r")
    for f in (FIG.figura_fiabilidad(d, tmp_path / "f.png"), FIG.figura_cobertura(d, tmp_path / "c.png")):
        assert f.is_file() and f.stat().st_size > 5000
