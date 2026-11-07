"""Diferencia pareada variante − base del macro-F1 en PV-test y PD-test, desde predicciones YA guardadas (sesión 10).

Uso:
  python scripts/comparar_con_base.py --config configs/04_evaluacion.yaml --salida <dir fuera del repo> --run-tests <run_id de la variante>
No reevalúa ningún modelo ni lee imágenes: compara las predicciones por imagen de la pasada única de la variante
(`<salida>/04_aumentacion/<run-tests>/predicciones_{pv,pd}_test.csv`) con las del base (`<salida>/03_brecha/<referencia_base.run_tests>/`),
sobre las mismas imágenes (verificado), con el bootstrap pareado del protocolo. Comprueba que el macro-F1 y su IC de cada modelo
coincidan con su `tests.json`. Escribe `comparacion_base.json` y lo copia a `results/04_aumentacion/<run_id>/`. Solo informa.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

from foliares.eval.evaluacion import comparar_con_base, exigir_fuera_del_repo
from foliares.utils.paths import cargar_rutas
from foliares.utils.sesion import Medidor, escribir_json, nuevo_run_id, registro_base

ARCHIVOS = {"PV-test": "predicciones_pv_test.csv", "PD-test": "predicciones_pd_test.csv"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path)
    ap.add_argument("--run-tests", required=True, help="run_id de la pasada única de la variante")
    a = ap.parse_args()
    R = cargar_rutas()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    cm = yaml.safe_load((R.repo / cfg["config_modelo"]).read_text(encoding="utf-8"))
    registro = registro_base(R.repo, R.splits, {"evaluacion": cfg, "modelo": cm}, {"bootstrap": cfg["bootstrap"]["semilla"]})
    salida = exigir_fuera_del_repo(a.salida, R.repo)
    nombres = [c["pv"] for c in cm["clases"]]
    dir_var = salida / cfg["experimento"] / a.run_tests
    dir_base = salida / "03_brecha" / cfg["referencia_base"]["run_tests"]
    t_var = json.loads((dir_var / "tests.json").read_text(encoding="utf-8"))
    t_base = json.loads((dir_base / "tests.json").read_text(encoding="utf-8"))
    if t_var["geometria"] != t_base["geometria"]:
        raise ValueError("Las dos pasadas no usaron la misma geometría")
    resultados = {}
    with Medidor() as med:
        for test, archivo in ARCHIVOS.items():
            pv, pb = pd.read_csv(dir_var / archivo), pd.read_csv(dir_base / archivo)
            if list(pv.ruta) != list(pb.ruta) or not (pv.etiqueta.to_numpy() == pb.etiqueta.to_numpy()).all():
                raise ValueError(f"{test}: las predicciones de la variante y del base no son de las mismas imágenes")
            r = comparar_con_base(pv.etiqueta.to_numpy(), pv.pred.to_numpy(), pb.pred.to_numpy(), nombres, cfg["bootstrap"])
            for modelo, ref in (("variante", t_var), ("base", t_base)):
                esperado = ref["resultados"][test]["base_macro_f1_ic"]
                if not all(math.isclose(r[modelo][k], esperado[k], abs_tol=1e-12) for k in ("estimacion", "ic_bajo", "ic_alto")):
                    raise ValueError(f"{test}: el macro-F1 de {modelo} no coincide con su tests.json")
            resultados[test] = {"n": int(len(pv)), **r}
    run_id = nuevo_run_id(f"{cfg['experimento']}_comparacion")
    datos = {**registro, "fase": "comparacion_con_base", "run_id": run_id, "run_tests_variante": a.run_tests,
             "run_tests_base": cfg["referencia_base"]["run_tests"], "geometria": t_var["geometria"],
             "origen": "predicciones por imagen ya guardadas de las dos pasadas únicas; no se reevaluó ningún modelo",
             "brecha": {"variante": t_var["brecha_macro_f1_pv_test_menos_pd_test"], "base": t_base["brecha_macro_f1_pv_test_menos_pd_test"]},
             "resultados": resultados, **med.resumen(), "solo_informativo": "no se concluye sobre diferencias dentro del IC"}
    ruta = escribir_json(salida / cfg["experimento"] / run_id / "comparacion_base.json", datos)
    destino = R.repo / "results" / cfg["experimento"] / run_id
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta, destino / "comparacion_base.json")
    print(json.dumps({"run_id": run_id, "cambios_sin_commitear": datos["cambios_sin_commitear"],
                      **{t: r["diferencia_variante_menos_base"] for t, r in resultados.items()}}, indent=2))
    print("JSON:", ruta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
