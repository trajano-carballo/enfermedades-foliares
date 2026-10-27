"""El notebook de Colab tiene que poder importar `foliares` en el kernel YA iniciado, sin reiniciar el entorno.

En la primera corrida real (sesión 7) la celda 7 falló con `No module named 'foliares.utils'`: el `pip install -e .` de la
celda 4 no se ve en un kernel ya arrancado (el .pth del editable se procesa al iniciar el intérprete) y `/content/foliares`
(el clon, que se llamaba igual que el paquete) se importaba como paquete namespace vacío. Acá se simula esa situación en un
intérprete aparte (sin el paquete instalado y con un directorio `foliares` alcanzable) y se ejecuta el código REAL de la
celda 4 del notebook.
"""

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "colab_arranque.ipynb"


def _celdas():
    return [c for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _celda(marca: str) -> str:
    c = ["".join(c["source"]) for c in _celdas() if "".join(c["source"]).startswith(marca)]
    assert len(c) == 1, f"debe haber una sola celda que empiece con {marca!r}"
    return c[0]


def test_el_clon_no_se_llama_como_el_paquete():
    ns: dict = {}
    exec(_celda("# 2) Variables de la sesión").split("assert os.path.isdir")[0].replace("import os, subprocess, time, json", "import os, time"), ns)
    assert Path(ns["REPO_DIR"]).name != "foliares"


def test_la_celda_4_hace_visible_foliares_en_un_kernel_ya_iniciado(tmp_path):
    decoy = tmp_path / "content"
    (decoy / "foliares").mkdir(parents=True)                       # como /content/foliares en el Colab de la primera corrida
    codigo = f"""
import os, sys, time
try:
    import foliares.utils.paths
except ModuleNotFoundError as e:
    print("ANTES:", e)                                             # reproduce el error de la celda 7
else:
    sys.exit(3)                                                    # el paquete ya es importable: no se puede simular
ns = {{"os": os, "time": time, "T": {{}}, "sh": lambda *a, **k: 0, "REPO_DIR": {str(REPO)!r}}}
exec({_celda("# 4) Instalar el paquete")!r}, ns)
from foliares.utils.paths import cargar_rutas
import foliares
print("DESPUES:", foliares.__file__)
assert os.path.realpath(foliares.__file__).startswith(os.path.realpath({str(REPO / 'src')!r}))
assert ns["SRC"] in os.environ["PYTHONPATH"].split(os.pathsep)
"""
    # -S: sin site-packages ni .pth (el editable no está activo, como en el kernel ya iniciado); las dependencias
    # (yaml, pandas, …) se encuentran por PYTHONPATH, que no procesa .pth.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(decoy), sysconfig.get_paths()["purelib"]])}
    r = subprocess.run([sys.executable, "-S", "-c", codigo], capture_output=True, text=True, env=env, cwd=tmp_path)
    if r.returncode == 3:
        pytest.skip("foliares está instalado como paquete regular en este entorno: no se puede simular el kernel sin instalar")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ANTES: No module named 'foliares.utils'" in r.stdout and "DESPUES:" in r.stdout


def test_sin_la_correccion_el_import_falla(tmp_path):
    # Control: la misma situación SIN el bootstrap de la celda 4 reproduce el error (si esto dejara de fallar, el test
    # anterior ya no estaría probando nada).
    (tmp_path / "content" / "foliares").mkdir(parents=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path / "content"), sysconfig.get_paths()["purelib"]])}
    r = subprocess.run([sys.executable, "-S", "-c", "from foliares.utils.paths import cargar_rutas"],
                       capture_output=True, text=True, env=env, cwd=tmp_path)
    if "No module named 'foliares.utils'" not in r.stderr:
        pytest.skip("foliares está instalado como paquete regular en este entorno")
    assert r.returncode != 0
