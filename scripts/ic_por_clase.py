"""IC bootstrap de exactitud y recall/precisión/F1 por clase para PV-test, PD-test y PD-dev, desde las predicciones guardadas.

Uso:
  python scripts/ic_por_clase.py --config configs/03_brecha.yaml --salida <dir fuera del repo> \\
      --run-dev <run_id fase dev> --run-tests <run_id fase tests>
No reevalúa el modelo ni lee imágenes: lee `<salida>/03_brecha/<run_tests>/predicciones_{pv,pd}_test.csv` y
`<salida>/03_brecha/<run_dev>/predicciones_pd_dev_<geometría>.csv`. Usa `foliares.eval.bootstrap` (estratificado, réplicas y
semilla de la config). Comprueba que el n, el macro-F1 y su IC coincidan con los de `tests.json` y `dev_geometria.json`. Escribe
`ic_por_clase.json` en `<salida>/03_brecha/<run_id>/` y lo copia a `results/03_brecha/<run_id>/` del repo.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

from foliares.eval.evaluacion import exigir_fuera_del_repo
from foliares.eval.ic_por_clase import ic_de_predicciones
from foliares.utils.paths import cargar_rutas
from foliares.utils.sesion import Medidor, escribir_json, nuevo_run_id, registro_base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path, help="carpeta fuera del repo (la de las sesiones 8 y 9)")
    ap.add_argument("--run-dev", required=True)
    ap.add_argument("--run-tests", required=True)
    a = ap.parse_args()
    R = cargar_rutas()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    cm = yaml.safe_load((R.repo / cfg["config_modelo"]).read_text(encoding="utf-8"))
    registro = registro_base(R.repo, R.splits, {"evaluacion": cfg, "modelo": cm}, {"bootstrap": cfg["bootstrap"]["semilla"]})
    salida = exigir_fuera_del_repo(a.salida, R.repo)
    bs, geo = cfg["bootstrap"], cfg["geometria"]
    nombres = [c["pv"] for c in cm["clases"]]
    exp = salida / cfg["experimento"]
    tests = json.loads((exp / a.run_tests / "tests.json").read_text(encoding="utf-8"))
    nombre_dev = next(n for n in ("dev_geometria.json", "dev_informativo.json") if (exp / a.run_dev / n).is_file())
    dev = json.loads((exp / a.run_dev / nombre_dev).read_text(encoding="utf-8"))
    dev = dev.get("regla_a", dev)                 # sesión 9: dentro de regla_a; etapa 4: arriba (dev_informativo.json)
    if tests["geometria"] != geo:
        raise ValueError(f"tests.json se evaluó con {tests['geometria']!r} y la config dice {geo!r}")
    fuentes = {"pv_test": (exp / a.run_tests / "predicciones_pv_test.csv", tests["resultados"]["PV-test"]["base_macro_f1_ic"],
                           tests["resultados"]["PV-test"]["n"]),
               "pd_test": (exp / a.run_tests / "predicciones_pd_test.csv", tests["resultados"]["PD-test"]["base_macro_f1_ic"],
                           tests["resultados"]["PD-test"]["n"]),
               "pd_dev": (exp / a.run_dev / f"predicciones_pd_dev_{geo}.csv", dev["macro_f1_por_geometria"][geo],
                          json.loads((exp / a.run_dev / nombre_dev).read_text(encoding="utf-8"))["n_pd_dev"])}
    resultados, consistencia = {}, {}
    with Medidor() as med:
        for nombre, (ruta, ref, n_ref) in fuentes.items():
            pred = pd.read_csv(ruta)
            r = ic_de_predicciones(pred, nombres, bs["replicas"], bs["semilla"], bs["nivel"])
            resultados[nombre] = r
            igual = {k: math.isclose(r["macro_f1"][k], ref[k], rel_tol=0, abs_tol=1e-12) for k in ("estimacion", "ic_bajo", "ic_alto")}
            consistencia[nombre] = {"n_igual": r["n"] == n_ref, "macro_f1_igual_al_json_de_la_evaluacion": all(igual.values()),
                                    "detalle": igual}
    ok = all(v["n_igual"] and v["macro_f1_igual_al_json_de_la_evaluacion"] for v in consistencia.values())
    if not ok:
        raise ValueError(f"Las predicciones no coinciden con los JSON de la evaluación: {consistencia}")
    run_id = nuevo_run_id(f"{cfg['experimento']}_ic")
    datos = {**registro, "fase": "ic_por_clase", "run_id": run_id, "run_dev": a.run_dev, "run_tests": a.run_tests, "geometria": geo,
             "modelo_base": cfg["modelo_base"], "bootstrap": bs,
             "origen": "predicciones por imagen ya guardadas de los runs indicados; no se reevaluó el modelo ni se leyó ninguna imagen",
             "convenciones": "remuestreo con reposición estratificado por clase verdadera, mismas réplicas para todas las métricas, IC por "
                             "percentiles; precisión, recall o F1 con denominador 0 valen 0",
             "orden_clases": nombres, "consistencia_con_la_evaluacion": consistencia, **resultados, **med.resumen(),
             "nota_memoria": "memoria_pico_cpu_mb es el máximo del proceso entero"}
    ruta = escribir_json(exp / run_id / "ic_por_clase.json", datos)
    destino = R.repo / "results" / cfg["experimento"] / run_id
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta, destino / "ic_por_clase.json")
    print(json.dumps({"run_id": run_id, "cambios_sin_commitear": datos["cambios_sin_commitear"], "commit": datos["commit"],
                      "exactitud": {k: round(v["exactitud"]["estimacion"], 4) for k, v in resultados.items()}}, indent=2))
    print("JSON:", ruta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
