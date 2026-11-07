"""Guarda de acceso a los tests: sin la fila commiteada y coherente en accesos_test.md, no se evalúa."""

import shutil
import subprocess

import pytest

from foliares.eval.guarda_accesos import AccesoNoRegistrado, exigir_fila_accesos, filas_de_la_tabla

CABECERA = "# Accesos\n\n| # | Fecha | Test | Modelo / config | Commit | Motivo |\n|---|---|---|---|---|---|\n"


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """Repo con config y código commiteados (c1) y luego la fila 1 de PD-test que apunta a c1 (c2)."""
    if shutil.which("git") is None:
        pytest.skip("git no está disponible")
    (tmp_path / "configs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "docs" / "bitacora").mkdir(parents=True)
    (tmp_path / "configs" / "03_brecha.yaml").write_text("a: 1\n")
    (tmp_path / "src" / "codigo.py").write_text("x = 1\n")
    (tmp_path / "docs" / "bitacora" / "accesos_test.md").write_text(CABECERA)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "c1")
    c1 = _git(tmp_path, "rev-parse", "--short", "HEAD")
    with open(tmp_path / "docs" / "bitacora" / "accesos_test.md", "a") as f:
        f.write(f"| 1 | 2026-11-05 | PD-test | modelo base, configs/03_brecha.yaml | `{c1}` | brecha |\n")
    _git(tmp_path, "commit", "-q", "-am", "c2: fila de acceso")
    return tmp_path, c1


def test_con_la_fila_commiteada_y_coherente_pasa(repo):
    r, c1 = repo
    f = exigir_fila_accesos(r, 1, "PD-test", "configs/03_brecha.yaml")
    assert f["commit"] == c1 and f["test"] == "PD-test" and len(f["head"]) == 40


def test_fila_inexistente(repo):
    r, _ = repo
    with pytest.raises(AccesoNoRegistrado, match="La fila 2 no está"):
        exigir_fila_accesos(r, 2, "PD-test", "configs/03_brecha.yaml")


def test_fila_escrita_pero_sin_commitear(repo):
    r, c1 = repo
    with open(r / "docs" / "bitacora" / "accesos_test.md", "a") as f:
        f.write(f"| 2 | 2026-11-05 | PV-test | modelo base, configs/03_brecha.yaml | `{c1}` | brecha |\n")
    with pytest.raises(AccesoNoRegistrado, match="sin commitear"):
        exigir_fila_accesos(r, 2, "PV-test", "configs/03_brecha.yaml")


def test_otro_test(repo):
    r, _ = repo
    with pytest.raises(AccesoNoRegistrado, match="es de 'PD-test'"):
        exigir_fila_accesos(r, 1, "PV-test", "configs/03_brecha.yaml")


def test_otra_config(repo):
    r, _ = repo
    with pytest.raises(AccesoNoRegistrado, match="no menciona la config"):
        exigir_fila_accesos(r, 1, "PD-test", "configs/otra.yaml")


def test_cambio_de_codigo_despues_de_la_fila(repo):
    r, _ = repo
    (r / "src" / "codigo.py").write_text("x = 2\n")
    _git(r, "commit", "-q", "-am", "c3: cambia el código")
    with pytest.raises(AccesoNoRegistrado, match="cambiaron la config o el código"):
        exigir_fila_accesos(r, 1, "PD-test", "configs/03_brecha.yaml")


def test_commit_inexistente(repo):
    r, _ = repo
    with open(r / "docs" / "bitacora" / "accesos_test.md", "a") as f:
        f.write("| 2 | 2026-11-05 | PD-test | configs/03_brecha.yaml | `deadbeef` | x |\n")
    _git(r, "commit", "-q", "-am", "c3")
    with pytest.raises(AccesoNoRegistrado, match="no existe"):
        exigir_fila_accesos(r, 2, "PD-test", "configs/03_brecha.yaml")


def test_test_desconocido(repo):
    r, _ = repo
    with pytest.raises(AccesoNoRegistrado, match="Test desconocido"):
        exigir_fila_accesos(r, 1, "PD-dev", "configs/03_brecha.yaml")


def test_lee_la_tabla_real_de_accesos():
    from pathlib import Path
    texto = (Path(__file__).resolve().parents[1] / "docs" / "bitacora" / "accesos_test.md").read_text(encoding="utf-8")
    filas = filas_de_la_tabla(texto)
    assert all(isinstance(k, int) and f["test"] for k, f in filas.items())     # las notas en prosa no cuentan como filas


def test_fuera_de_un_repo_git(tmp_path):
    with pytest.raises(AccesoNoRegistrado, match="no es un repositorio git"):
        exigir_fila_accesos(tmp_path, 1, "PD-test", "configs/03_brecha.yaml")


def test_una_fila_ya_usada_se_rechaza(repo):
    r, c1 = repo
    (r / "results" / "x" / "r1").mkdir(parents=True)
    (r / "results" / "x" / "r1" / "tests.json").write_text('{"accesos": {"PD-test": {"fila": 1, "commit": "%s"}}}' % c1)
    _git(r, "add", "results")
    _git(r, "commit", "-q", "-m", "c3: resultado de la evaluación")
    with pytest.raises(AccesoNoRegistrado, match="ya se usó"):
        exigir_fila_accesos(r, 1, "PD-test", "configs/03_brecha.yaml")


def test_un_resultado_sin_bloque_accesos_no_gasta_la_fila(repo):
    r, _ = repo
    (r / "results").mkdir()
    (r / "results" / "otro.json").write_text('{"fila": 1, "macro_f1": 0.5}')
    _git(r, "add", "results")
    _git(r, "commit", "-q", "-m", "c3")
    assert exigir_fila_accesos(r, 1, "PD-test", "configs/03_brecha.yaml")["fila"] == 1


def test_cambio_en_la_config_del_modelo_despues_de_la_fila(repo):
    r, _ = repo
    (r / "configs" / "02_modelo_base.yaml").write_text("lr: 0.1\n")
    _git(r, "add", "configs")
    _git(r, "commit", "-q", "-m", "c3: cambia la config del modelo")
    with pytest.raises(AccesoNoRegistrado, match="cambiaron la config o el código"):
        exigir_fila_accesos(r, 1, "PD-test", "configs/03_brecha.yaml")


def test_las_filas_1_a_4_del_repo_real_ya_estan_gastadas():
    from pathlib import Path
    from foliares.eval.guarda_accesos import filas_usadas
    usadas = filas_usadas(Path(__file__).resolve().parents[1])
    assert {1, 2, 3, 4} <= set(usadas)
    assert all("03_brecha_tests" in usadas[k] for k in (1, 2)) and all("04_aumentacion_tests" in usadas[k] for k in (3, 4))
