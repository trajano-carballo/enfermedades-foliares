"""`scripts/colab_humo.py`: `--salida` es obligatorio (sin default dentro del repo) y el notebook lo pasa."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_sin_salida_argparse_falla_sin_escribir_nada(tmp_path):
    # DATA_ROOT vacío: aunque el script no exigiera --salida, fallaría al leer las imágenes sin escribir nada en el repo.
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "colab_humo.py"), "--config", str(REPO / "configs" / "colab_humo.yaml"),
                        "--fase", "entrenar"], capture_output=True, text=True, cwd=REPO, env={**os.environ, "DATA_ROOT": str(tmp_path)})
    assert r.returncode == 2 and "the following arguments are required: --salida" in r.stderr, r.stdout + r.stderr


def test_el_notebook_pasa_salida_en_las_dos_fases():
    celdas = ["".join(c["source"]) for c in json.loads((REPO / "notebooks" / "colab_arranque.ipynb").read_text(encoding="utf-8"))["cells"]
              if c["cell_type"] == "code"]
    base = [c for c in celdas if c.startswith("# 9) Fase 1")]
    assert len(base) == 1 and "--salida {SALIDA}" in base[0]
    assert any(c.startswith("# 10) Fase 2") and "{base} --fase reanudar" in c for c in celdas)    # reutiliza `base`
