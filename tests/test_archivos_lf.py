"""Todo lo que escribe el código sale con fin de línea LF (los sha256 de los manifiestos coinciden entre PC) y
el empaquetado no deja archivos parciales."""

import json
import tarfile

import pandas as pd
import pytest

from foliares.data import paquete
from foliares.data import plantdoc_chequeos as pc
from foliares.data import plantdoc_depurado as dp
from foliares.utils.archivos import escribir_csv, escribir_json, escribir_texto
from foliares.utils.git import SIN_GIT, info_repo

DF = pd.DataFrame({"ruta": ["a/ñ.jpg", "b.jpg"], "n": [1, 2]})


def _sin_cr(p):
    b = p.read_bytes()
    assert len(b) > 0 and b"\r" not in b, p.name
    return b


def test_escribir_csv_json_y_texto_con_lf(tmp_path):
    _sin_cr(escribir_csv(DF, tmp_path / "x" / "a.csv"))
    b = _sin_cr(escribir_csv(DF, tmp_path / "b.csv", bom=True))
    assert b.startswith(b"\xef\xbb\xbf")
    j = _sin_cr(escribir_json(tmp_path / "m.meta.json", {"k": "ñ", "lista": [1, 2]}))
    assert json.loads(j.decode("utf-8")) == {"k": "ñ", "lista": [1, 2]} and "ñ".encode() in j
    _sin_cr(escribir_texto(tmp_path / "t.txt", "uno\ndos\n"))


def test_el_mismo_contenido_da_los_mismos_bytes(tmp_path):
    escribir_csv(DF, tmp_path / "a.csv")
    (tmp_path / "b.csv").write_bytes(DF.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_guards_escriben_con_lf(tmp_path):
    plan = pd.DataFrame({"id": ["TL-001"], "split": ["train"], "ruta": ["r"], "revisor_1": [""], "revisor_2": [""], "motivo": [""]})
    assert pc.guardar_planilla(plan, tmp_path / "planilla.csv") == "creada"
    pares = pd.DataFrame({"ruta_a": ["a"], "ruta_b": ["b"], "decision_equipo": [""]})
    assert dp.guardar_pares_7_10(pares, tmp_path / "pares.csv") == "creado"
    _sin_cr(tmp_path / "planilla.csv")
    _sin_cr(tmp_path / "pares.csv")


def test_paquete_escribe_manifiesto_y_suma_con_lf(tmp_path):
    root = tmp_path / "root"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "1.jpg").write_bytes(b"uno")
    tar = tmp_path / "out" / "ds.tar"
    paquete.empaquetar(["raw/1.jpg"], root, tar, "ds")
    _sin_cr(tar.with_name("ds.tar.sha256"))
    with tarfile.open(tar) as tf:
        assert b"\r" not in tf.extractfile("MANIFIESTO_ds.csv").read()


def test_info_repo_fuera_de_un_repo(tmp_path):
    r = info_repo(tmp_path)
    assert r == {"commit": SIN_GIT, "cambios_sin_commitear": None}


def test_info_repo_en_este_repo():
    from foliares.utils.paths import raiz_repo
    r = info_repo(raiz_repo())
    assert r["commit"] == SIN_GIT or len(r["commit"]) == 40


# ------------------------------------------------------------------ empaquetado sin parciales
@pytest.fixture
def arbol(tmp_path):
    root = tmp_path / "root"
    (root / "d").mkdir(parents=True)
    for k in range(3):
        (root / "d" / f"{k}.jpg").write_bytes(b"x" * (k + 1))
    return root, [f"d/{k}.jpg" for k in range(3)]


def _archivos(carpeta):
    return sorted(p.name for p in carpeta.iterdir())


def test_empaquetar_que_falla_no_deja_parciales(arbol, tmp_path, monkeypatch):
    root, rutas = arbol
    out = tmp_path / "out"
    llamadas = []
    original = tarfile.TarFile.add

    def falla_en_la_tercera(self, *a, **k):
        llamadas.append(1)
        if len(llamadas) == 3:
            raise OSError("disco lleno (simulado)")
        return original(self, *a, **k)

    monkeypatch.setattr(tarfile.TarFile, "add", falla_en_la_tercera)
    with pytest.raises(OSError, match="simulado"):
        paquete.empaquetar(rutas, root, out / "ds.tar", "ds")
    assert _archivos(out) == []                                  # ni .tar, ni .parcial, ni manifiesto temporal, ni .sha256


def test_un_intento_fallido_no_pisa_un_paquete_anterior(arbol, tmp_path, monkeypatch):
    root, rutas = arbol
    out = tmp_path / "out"
    paquete.empaquetar(rutas, root, out / "ds.tar", "ds")
    antes = {n: (out / n).read_bytes() for n in _archivos(out)}
    assert sorted(antes) == ["ds.tar", "ds.tar.sha256"]
    monkeypatch.setattr(tarfile.TarFile, "add", lambda *a, **k: (_ for _ in ()).throw(OSError("simulado")))
    with pytest.raises(OSError):
        paquete.empaquetar(rutas, root, out / "ds.tar", "ds")
    assert {n: (out / n).read_bytes() for n in _archivos(out)} == antes
    assert paquete.verificar_tar(out / "ds.tar")


def test_empaquetar_con_archivo_faltante_no_crea_nada(arbol, tmp_path):
    root, rutas = arbol
    with pytest.raises(FileNotFoundError):
        paquete.empaquetar(rutas + ["d/no_existe.jpg"], root, tmp_path / "out" / "ds.tar", "ds")
    assert not (tmp_path / "out").exists() or _archivos(tmp_path / "out") == []
