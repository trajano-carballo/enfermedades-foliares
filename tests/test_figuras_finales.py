"""Sesión 14: figuras y tablas finales salen solo de los JSON versionados de results/ (sin cómputo nuevo)."""

import sys
from pathlib import Path

import pytest

from foliares.eval import finales as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import figuras_finales  # noqa: E402


def test_formato_ic_y_no_definido():
    assert F.ic({"estimacion": 0.80683, "ic_bajo": 0.76817, "ic_alto": 0.84539}) == "0,807 [0,768; 0,845]"
    assert F.ic({"estimacion": None, "nota": "no definido (0 respondidas)"}) == "no definido (0 respondidas)"
    assert F.ic(None) == F.NO_MEDIDO
    assert F.num(2202.0, 0) == "2.202"


def test_script_escribe_figuras_y_tablas_con_los_numeros_de_results(tmp_path):
    assert figuras_finales.main(["--config", str(REPO / "configs" / "07_figuras_finales.yaml"), "--figuras", str(tmp_path)]) == 0
    pngs = {p.name for p in tmp_path.glob("*.png")}
    assert {"brecha_base_vs_variante.png", "matrices_base.png", "matrices_variante.png", "fiabilidad_pd_dev.png",
            "cobertura_pd_dev.png", "fiabilidad_pd_test.png", "latencia_tamano.png"} <= pngs
    assert not any(n.startswith("cam_") for n in pngs)          # las grillas de CAM muestran PlantDoc: no se copian (licencia)
    md = (tmp_path / "tablas.md").read_text(encoding="utf-8")
    for esperado in ("0,807 [0,768; 0,845]", "0,830 [0,792; 0,867]", "-0,004 [-0,030; 0,023]",      # brechas y pareada PD-test
                     "0,676 [0,635; 0,717]", "0,021 [0,015; 0,067]",                               # ECE PD-test antes/después
                     "no definido (0 respondidas)", "6,76 MB"):
        assert esperado in md
    assert "| NO |" not in md                    # todas las fuentes vienen de un árbol limpio
    assert "Generado desde el commit `" in md


def test_rechaza_fuente_fuera_de_results(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("figuras: x\nfuentes: {a: data/splits/plantdoc_particion.csv}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fuera de results"):
        figuras_finales.main(["--config", str(cfg), "--figuras", str(tmp_path)])
