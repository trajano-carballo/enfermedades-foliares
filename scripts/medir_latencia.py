"""Latencia y tamaño de la app (etapa 6, sesión 12, fase 4): p50 y p95 de punta a punta (leer el archivo -> respuesta) con y sin
CAM, en CPU, sobre una muestra fija (semilla de la config) de PV-val y PD-dev más las imágenes externas válidas de Commons;
tamaño del archivo de pesos. No toca ningún test ni calcula ninguna métrica de desempeño.

Uso:
  python scripts/medir_latencia.py --config configs/06_app.yaml --salida <carpeta fuera del repo con el checkpoint>
Escribe results/06_app/<run_id>/latencia.json. Para levantar la app y sacar las capturas:
  python app/app.py --config configs/06_app.yaml --salida <la misma carpeta>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from foliares.data.dataset import filas_particion
from foliares.eval.evaluacion import _nombres, cargar_modelo_base, exigir_fuera_del_repo
from foliares.eval.latencia import cronometrar, entorno_cpu, muestra_fija, resumen_ms
from foliares.data.externas import leer_planilla
from foliares.utils.paths import cargar_rutas
from foliares.utils.sesion import Medidor, escribir_json, nuevo_run_id, registro_base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    R = cargar_rutas()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    cal = yaml.safe_load((R.repo / cfg["calibracion"]).read_text(encoding="utf-8"))
    cm = yaml.safe_load((R.repo / cal["config_modelo"]).read_text(encoding="utf-8"))
    L = cfg["latencia"]
    registro = registro_base(R.repo, R.splits, {"app": cfg, "calibracion": cal, "modelo": cm}, {"muestra": L["semilla"]})
    salida = exigir_fuera_del_repo(a.salida, R.repo)
    run_id = a.run_id or nuevo_run_id(f"{cfg['experimento']}_latencia")
    nombres, nombres_pd, _ = _nombres(cm)

    # Muestra fija: solo PV-val y PD-dev (las particiones de test no se piden) y las externas válidas (no duplicadas).
    pv = filas_particion(R.splits, "plantvillage", "val", "validacion", nombres)
    pdv = filas_particion(R.splits, "plantdoc", "dev", "evaluacion", nombres, nombres_pd)
    externas = [f for f in leer_planilla(R.repo / cfg["externas"]["planilla"]) if not f["resultado_duplicados"].startswith("DUPLICADA")]
    conjuntos = {"PV-val": [R.data_root / r for r in muestra_fija(pv, L["muestra_por_conjunto"], L["semilla"])],
                 "PD-dev": [R.data_root / r for r in muestra_fija(pdv, L["muestra_por_conjunto"], L["semilla"])],
                 "Commons": [R.data_root / cfg["externas"]["carpeta"] / f["archivo"] for f in externas]}

    ckpt = salida / cal["modelo_base"]["checkpoint"]
    with Medidor() as med:
        t0 = time.perf_counter()
        modelo = cargar_modelo_base(salida, cal, len(nombres))      # incluye la verificación del sha256 del checkpoint
        carga_s = time.perf_counter() - t0
        tiempos = {}
        for con_cam in (False, True):
            clave = "con_cam" if con_cam else "sin_cam"
            for nombre, rutas in conjuntos.items():
                tiempos[(nombre, clave)] = cronometrar(modelo, rutas, nombres, cal["temperatura"], cal["tau"], cal["geometria"],
                                                       cal["tamano"], con_cam, L["calentamiento"], L["repeticiones"])
    resultados = {}
    for clave in ("sin_cam", "con_cam"):
        todos = [t for (n, c), ts in tiempos.items() if c == clave for t in ts]
        resultados[clave] = {"todas": resumen_ms(todos), **{n: resumen_ms(tiempos[(n, clave)]) for n in conjuntos}}
    datos = {**registro, "fase": "latencia", "run_id": run_id, "entorno": entorno_cpu(),
             "parametros": {"muestra_por_conjunto": L["muestra_por_conjunto"], "semilla": L["semilla"], "calentamiento": L["calentamiento"],
                            "repeticiones": L["repeticiones"], "imagenes_por_conjunto": {n: len(r) for n, r in conjuntos.items()},
                            "geometria": cal["geometria"], "tamano": cal["tamano"]},
             "muestra": {n: [Path(r).relative_to(R.data_root).as_posix() for r in rutas] for n, rutas in conjuntos.items()},
             "que_mide": "predecir_imagen de punta a punta (leer el archivo, preproceso, forward, calibración, top-3 y, con CAM, el mapa "
                         "superpuesto), en CPU, sin la capa de red de Gradio; el sistema operativo puede tener los archivos en caché "
                         "en las repeticiones",
             "resultados_ms": resultados, "sobrecosto_cam_p50_ms": resultados["con_cam"]["todas"]["p50_ms"] - resultados["sin_cam"]["todas"]["p50_ms"],
             "carga_del_modelo_s": carga_s,
             "pesos": {"archivo": cal["modelo_base"]["checkpoint"], "bytes": ckpt.stat().st_size, "MB": ckpt.stat().st_size / 1e6,
                       "MiB": ckpt.stat().st_size / 2**20, "sha256": cal["modelo_base"]["sha256"]},
             "tiempos_crudos_ms": {f"{n}/{c}": [round(t * 1000, 3) for t in ts] for (n, c), ts in tiempos.items()},
             **med.resumen(), "nota_memoria": "memoria_pico_cpu_mb es el máximo del proceso entero"}
    destino = R.repo / "results" / cfg["experimento"] / run_id
    escribir_json(destino / "latencia.json", datos)
    print(json.dumps({"run_id": run_id, "cambios_sin_commitear": datos["cambios_sin_commitear"], "entorno": datos["entorno"],
                      "imagenes_por_conjunto": datos["parametros"]["imagenes_por_conjunto"],
                      "p50_p95_ms": {c: {n: [round(v["p50_ms"], 1), round(v["p95_ms"], 1)] for n, v in r.items()} for c, r in resultados.items()},
                      "sobrecosto_cam_p50_ms": round(datos["sobrecosto_cam_p50_ms"], 1), "carga_del_modelo_s": round(carga_s, 2),
                      "pesos_MB": round(datos["pesos"]["MB"], 2)}, indent=2, ensure_ascii=False))
    print("JSON en:", destino / "latencia.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
