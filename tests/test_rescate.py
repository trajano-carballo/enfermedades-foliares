"""Script de rescate de PlantDoc sobre un repositorio git sintético (armado con comandos de bajo nivel, sin checkout),
que tiene los tres casos: nombre con `?`, nombre demasiado largo y dos archivos que difieren solo en mayúsculas."""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rescatar", RAIZ / "scripts" / "rescatar_archivos_plantdoc.py")
rescatar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rescatar)

LARGO = "x" * 150 + ".jpg"
ARBOL = {                                   # ruta en el árbol -> contenido
    "README.md": b"readme",
    "train/Clase/ok.jpg": b"ok",
    "train/Clase/q?x.jpg": b"con-pregunta",
    f"train/Clase/{LARGO}": b"largo",
    "train/Clase/Peach-Leaf.jpg": b"mayuscula",
    "train/Clase/peach-leaf.jpg": b"minuscula",
    "test/Clase/otra.jpg": b"otra",
}


def _git(repo, *args, entrada=None):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(repo), *args], input=entrada, capture_output=True, check=True, env=env).stdout.decode().strip()


@pytest.fixture
def clon(tmp_path):
    """Repo con ARBOL en el índice (sin materializar los archivos) y el disco como lo dejaría NTFS:
    sin el de `?`, sin el largo y con un solo archivo de cada par de mayúsculas."""
    repo = tmp_path / "pd"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "core.ignorecase", "false")
    _git(repo, "config", "core.protectNTFS", "false")           # permite `?` en el índice (en Windows se rechaza al hacer checkout)
    for ruta, contenido in ARBOL.items():
        sha = subprocess.run(["git", "-C", str(repo), "hash-object", "-w", "--stdin"], input=contenido, capture_output=True, check=True).stdout.decode().strip()
        _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{sha},{ruta}")
    tree = _git(repo, "write-tree")
    rev = _git(repo, "commit-tree", tree, "-m", "t")
    for ruta in ("README.md", "train/Clase/ok.jpg", "train/Clase/Peach-Leaf.jpg", "test/Clase/otra.jpg"):
        (repo / ruta).parent.mkdir(parents=True, exist_ok=True)
        (repo / ruta).write_bytes(ARBOL[ruta])
    return repo, rev


def _mapeo_esperado():
    return [
        {"split": "train", "clase": "Clase", "nombre_original": "q?x.jpg", "nombre_local": "q_x.jpg", "motivo": "caracter_invalido_ntfs"},
        {"split": "train", "clase": "Clase", "nombre_original": LARGO, "nombre_local": rescatar.nombre_local_para(LARGO)[0], "motivo": "ruta_demasiado_larga"},
        {"split": "train", "clase": "Clase", "nombre_original": "peach-leaf.jpg", "nombre_local": "peach-leaf__colision_mayusc.jpg", "motivo": "colision_mayusculas_ntfs"},
    ]


def _csv(path, filas):
    rescatar.escribir_csv_mapeo(filas, path, sobrescribir=False)
    return path


def test_nombre_local_y_colision():
    assert rescatar.nombre_local_para("a?b.jpg") == ("a_b.jpg", "caracter_invalido_ntfs")
    assert rescatar.nombre_local_para("ok.jpg") == ("ok.jpg", "ninguno")
    largo, motivo = rescatar.nombre_local_para(LARGO)
    assert motivo == "ruta_demasiado_larga" and len(largo) <= rescatar.MAX_PATH_COMPONENTE and largo.endswith(".jpg")
    assert rescatar.nombre_colision("Peach-Leaf.jpg") == "Peach-Leaf__colision_mayusc.jpg"
    assert rescatar.nombre_colision("sin_extension") == "sin_extension__colision_mayusc"


def test_regenerar_mapeo_distingue_los_tres_motivos(clon):
    repo, rev = clon
    arbol = rescatar.cargar_mapa_sha(repo, rev)
    assert len(rescatar.rutas_del_arbol(arbol)) == 6                          # README.md no cuenta
    faltantes = rescatar.rutas_faltantes(arbol, rescatar.listar_disco(repo))
    assert sorted(faltantes) == sorted(f"train/Clase/{n}" for n in ("q?x.jpg", LARGO, "peach-leaf.jpg"))
    filas = rescatar.regenerar_mapeo(faltantes, arbol)
    assert sorted(filas, key=lambda f: f["motivo"]) == sorted(_mapeo_esperado(), key=lambda f: f["motivo"])
    assert filas[-1]["motivo"] == "colision_mayusculas_ntfs"                  # las colisiones van al final, como en el CSV versionado


def test_aplica_el_mapeo_extrae_por_blob_y_verifica(clon, tmp_path):
    repo, rev = clon
    arbol = rescatar.cargar_mapa_sha(repo, rev)
    mapeo = _mapeo_esperado()
    acciones, conflictos = rescatar.planificar(repo, arbol, mapeo, False, rescatar.listar_disco(repo))
    assert conflictos == [] and {a["accion"] for a in acciones} == {"escribir"}
    assert rescatar.aplicar(repo, acciones) == {"escribir": 3, "mover": 0, "presente": 0}
    assert (repo / "train/Clase/q_x.jpg").read_bytes() == b"con-pregunta"
    assert (repo / "train/Clase/peach-leaf__colision_mayusc.jpg").read_bytes() == b"minuscula"
    v = rescatar.verificar(repo, arbol, mapeo)
    assert v["ok"] and v["esperados"] == 6 and v["en_disco"] == 6
    # idempotente: segunda corrida => todo presente, nada se reescribe
    acciones, conflictos = rescatar.planificar(repo, arbol, mapeo, False, rescatar.listar_disco(repo))
    assert conflictos == [] and rescatar.aplicar(repo, acciones) == {"escribir": 0, "mover": 0, "presente": 3}


def test_nunca_sobrescribe_en_silencio(clon):
    repo, rev = clon
    arbol = rescatar.cargar_mapa_sha(repo, rev)
    ya = repo / "train/Clase/q_x.jpg"
    ya.write_bytes(b"otro contenido")
    acciones, conflictos = rescatar.planificar(repo, arbol, _mapeo_esperado(), False, rescatar.listar_disco(repo))
    assert len(conflictos) == 1 and "contenido distinto" in conflictos[0]
    assert ya.read_bytes() == b"otro contenido"


def test_mapeo_que_nombra_un_archivo_fuera_del_arbol_es_conflicto(clon):
    repo, rev = clon
    arbol = rescatar.cargar_mapa_sha(repo, rev)
    mapeo = [{"split": "train", "clase": "Clase", "nombre_original": "no_esta.jpg", "nombre_local": "no_esta.jpg", "motivo": "x"}]
    _, conflictos = rescatar.planificar(repo, arbol, mapeo, False, rescatar.listar_disco(repo))
    assert len(conflictos) == 1 and "no está en el árbol" in conflictos[0]


def test_original_materializado_por_git_se_avisa_y_se_puede_mover(clon):
    # PC con rutas largas habilitadas: git materializó el archivo largo con su nombre original
    repo, rev = clon
    arbol = rescatar.cargar_mapa_sha(repo, rev)
    (repo / "train/Clase" / LARGO).write_bytes(ARBOL[f"train/Clase/{LARGO}"])
    mapeo = _mapeo_esperado()
    acciones, conflictos = rescatar.planificar(repo, arbol, mapeo, False, rescatar.listar_disco(repo))
    assert len(conflictos) == 1 and "--mover-originales" in conflictos[0] and not (repo / "train/Clase" / mapeo[1]["nombre_local"]).exists()
    acciones, conflictos = rescatar.planificar(repo, arbol, mapeo, True, rescatar.listar_disco(repo))
    assert conflictos == [] and rescatar.aplicar(repo, acciones) == {"escribir": 2, "mover": 1, "presente": 0}
    assert not (repo / "train/Clase" / LARGO).exists()                                     # quedó solo el nombre local
    assert rescatar.verificar(repo, arbol, mapeo)["ok"]                                    # el resultado es igual al de otra PC


def test_verificar_detecta_sha1_distinto_faltantes_y_sobrantes(clon):
    repo, rev = clon
    arbol = rescatar.cargar_mapa_sha(repo, rev)
    mapeo = _mapeo_esperado()
    rescatar.aplicar(repo, rescatar.planificar(repo, arbol, mapeo, False, rescatar.listar_disco(repo))[0])
    (repo / "train/Clase/ok.jpg").write_bytes(b"corrupto")
    (repo / "test/Clase/otra.jpg").unlink()
    (repo / "train/Clase/de_mas.jpg").write_bytes(b"x")
    v = rescatar.verificar(repo, arbol, mapeo)
    assert not v["ok"] and v["sha1_distinto"] == ["train/Clase/ok.jpg"] and v["faltan"] == ["test/Clase/otra.jpg"] and v["sobran"] == ["train/Clase/de_mas.jpg"]


def test_main_de_punta_a_punta_salidas_en_interim_con_lf(clon, tmp_path, capsys):
    repo, rev = clon
    mapeo = _csv(tmp_path / "versionado.csv", _mapeo_esperado())
    salida = tmp_path / "interim"
    argv = ["--repo", str(repo), "--rev", rev, "--mapeo", str(mapeo), "--salida", str(salida), "--regenerar"]
    assert rescatar.main(argv) == 0
    out = capsys.readouterr().out
    assert "IGUAL al versionado" in out and "-> OK" in out
    assert (salida / "plantdoc_rutas_faltantes.txt").read_text(encoding="utf-8").splitlines() == sorted(
        f"train/Clase/{n}" for n in ("q?x.jpg", LARGO, "peach-leaf.jpg"))
    regenerado = salida / "plantdoc_archivos_renombrados.regenerado.csv"
    assert regenerado.is_file() and b"\r" not in regenerado.read_bytes() and b"\r" not in mapeo.read_bytes()
    # segunda corrida: no pisa el CSV regenerado sin --sobrescribir
    with pytest.raises(SystemExit, match="ya existe"):
        rescatar.main(argv)
    assert rescatar.main(argv + ["--sobrescribir"]) == 0


def test_main_con_conflicto_devuelve_1_y_no_escribe(clon, tmp_path, capsys):
    repo, rev = clon
    (repo / "train/Clase/q_x.jpg").write_bytes(b"distinto")
    mapeo = _csv(tmp_path / "versionado.csv", _mapeo_esperado())
    assert rescatar.main(["--repo", str(repo), "--rev", rev, "--mapeo", str(mapeo), "--salida", str(tmp_path / "i")]) == 1
    assert "conflictos" in capsys.readouterr().err
    assert not (repo / "train/Clase/peach-leaf__colision_mayusc.jpg").exists()         # no escribió nada
    assert (repo / "train/Clase/q_x.jpg").read_bytes() == b"distinto"


def test_mapeo_regenerado_distinto_del_versionado_devuelve_1(clon, tmp_path):
    repo, rev = clon
    filas = _mapeo_esperado()[:2]                                                       # el versionado no trae la colisión
    mapeo = _csv(tmp_path / "versionado.csv", filas)
    assert rescatar.main(["--repo", str(repo), "--rev", rev, "--mapeo", str(mapeo), "--salida", str(tmp_path / "i"), "--regenerar", "--sin-verificar"]) == 1


def test_mapeo_con_separador_punto_y_coma_es_error_claro(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("split;clase;nombre_original;nombre_local;motivo\ntrain;C;a;b;x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="faltan las columnas"):
        rescatar.leer_mapeo(f)


def test_el_mapeo_versionado_es_consistente_con_nombre_local_para():
    # las 95 filas que no son colisión se reproducen con la función; las 6 colisiones llevan el sufijo
    filas = rescatar.leer_mapeo(RAIZ / "docs" / "bitacora" / "plantdoc_archivos_renombrados.csv")
    assert len(filas) == 101
    for f in filas:
        if f["motivo"] == rescatar.MOTIVO_COLISION:
            assert f["nombre_local"] == rescatar.nombre_colision(f["nombre_original"])
        else:
            assert rescatar.nombre_local_para(f["nombre_original"]) == (f["nombre_local"], f["motivo"])
    assert sum(f["motivo"] == rescatar.MOTIVO_COLISION for f in filas) == 6
