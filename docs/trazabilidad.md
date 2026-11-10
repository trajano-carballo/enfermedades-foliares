# Trazabilidad de los números del informe

Cada número que va a `docs/informe_final.md` tiene acá una fila: **número → archivo → config → commit del run → fila de
`accesos_test.md`** (si viene de un test). Sesión 14, 2026-11-10. No hay ningún número calculado para este documento: todo sale
de archivos ya versionados.

**Cómo se verifica.** `tests/test_trazabilidad.py` relee las dos tablas de abajo y comprueba, para cada fila cuya clave no
empieza con `derivado` ni `—`, que el valor escrito coincide con la fuente al redondeo mostrado, y que el commit de cada JSON es
el del run y su árbol estaba limpio. Clave de un JSON: ruta con puntos (un segmento numérico indexa una lista; varias claves se
separan con `;`). Clave de un CSV: condiciones `col=valor` / `col!=valor` unidas con `&`, y el valor es el número de filas.
Las filas `derivado` se verificaron a mano en la sesión 14 a partir de la misma fuente (la fórmula está en la clave).

Convenciones: coma decimal, punto de miles; IC 95 % bootstrap estratificado por clase, 2.000 réplicas, semilla 42
(`docs/protocolo_evaluacion.md` §b). Modelo base = `02_modelo_base_20261104_201229`; variante = `04_aumentacion_20261106_232146`.

## Total de accesos a cada test (de `docs/bitacora/accesos_test.md`)

| Test | Evaluaciones con modelo | Filas | Qué se evaluó |
|---|---|---|---|
| PV-test (2.170 imágenes) | **2** | 1, 3 | modelo base (fila 1) y variante (fila 3), una pasada cada uno |
| PD-test (300 imágenes) | **3** | 2, 4, 5 | modelo base (fila 2), variante (fila 4) y variante calibrada con T y τ fijos (fila 5) |

Exposición sin modelo, también registrada en `accesos_test.md`: en la revisión humana de etiquetas (asistida por IA, un revisor) se
miraron 130 imágenes de PlantDoc, de las que **34 caen en el test congelado de PlantDoc** (nota de la sesión 6). El resto de las notas
(hashing perceptual, sha256 de paquetes, figuras regeneradas desde JSON ya guardados) no pasa ningún test por un modelo.

## Fuentes

| Alias | Archivo | Config | Commit del run | Accesos |
|---|---|---|---|---|
| `PV_SPLIT` | `data/splits/plantvillage_particion.csv` | `configs/congelar_splits.yaml` | `b79d3a2` | — |
| `PD_SPLIT` | `data/splits/plantdoc_particion.csv` | `configs/congelar_splits.yaml` | `b79d3a2` | — |
| `REVISION` | `docs/bitacora/revision_etiquetas_planilla.csv` | — (revisión humana, decisión D1) | `4cca764` | — |
| `PARES` | `docs/bitacora/plantdoc_pares_7_10_revision.csv` | — (revisión humana, decisión D3) | `4cca764` | — |
| `ACCESOS` | `docs/bitacora/accesos_test.md` | — | — | — |
| `BASE_ENT` | `results/02_modelo_base/02_modelo_base_20261104_201229/resultados.json` | `configs/02_modelo_base.yaml` | `74ff5de` | — |
| `GEO` | `results/03_brecha/03_brecha_dev_20261104_210006/dev_geometria.json` | `configs/03_brecha.yaml` | `bc8006a` | — (PD-dev) |
| `DIAG` | `results/03_brecha/03_brecha_dev_20261104_210006/diagnostico_dev.json` | `configs/03_brecha.yaml` | `60e0a2b` | — (PD-dev) |
| `T_BASE` | `results/03_brecha/03_brecha_tests_20261105_081814/tests.json` | `configs/03_brecha.yaml` | `09a1e3e` | filas 1 (PV-test) y 2 (PD-test) |
| `IC_BASE` | `results/03_brecha/03_brecha_ic_20261105_085644/ic_por_clase.json` | `configs/03_brecha.yaml` | `3437441` | filas 1 y 2 (predicciones guardadas) |
| `CAM_BASE` | `results/03_brecha/03_brecha_cam_20261105_090654/cam_muestra.json` | `configs/03_brecha.yaml` | `488676b` | — (PD-dev y PV-val) |
| `VAR_ENT` | `results/04_aumentacion/04_aumentacion_20261106_232146/resultados.json` | `configs/04_aumentacion.yaml` | `8e91fad` | — |
| `VAR_DEV` | `results/04_aumentacion/04_aumentacion_dev_informativo_20261106_210332/dev_informativo.json` | `configs/04_evaluacion.yaml` | `0a094ac` | — (PD-dev) |
| `T_VAR` | `results/04_aumentacion/04_aumentacion_tests_20261106_210645/tests.json` | `configs/04_evaluacion.yaml` | `9d0f5a7` | filas 3 (PV-test) y 4 (PD-test) |
| `COMP` | `results/04_aumentacion/04_aumentacion_comparacion_20261106_211523/comparacion_base.json` | `configs/04_evaluacion.yaml` | `6fa21ed` | filas 1–4 (predicciones guardadas) |
| `IC_VAR` | `results/04_aumentacion/04_aumentacion_ic_20261106_211534/ic_por_clase.json` | `configs/04_evaluacion.yaml` | `6fa21ed` | filas 3 y 4 (predicciones guardadas) |
| `CAM_VAR` | `results/04_aumentacion/04_aumentacion_cam_20261106_211557/cam_muestra.json` | `configs/04_evaluacion.yaml` | `6fa21ed` | — (PD-dev y PV-val) |
| `CAL_DEV` | `results/05_calibracion/05_calibracion_dev_20261107_095030/calibracion_dev.json` | `configs/05_calibracion.yaml` | `6e5196a` | — (PD-dev) |
| `CAL_TEST` | `results/05_calibracion/05_calibracion_test_20261107_095821/calibracion_test.json` | `configs/05_calibracion.yaml` | `5bff0d7` | fila 5 (PD-test) |
| `APP_PV` | `results/06_app/06_app_pv_val_20261110_093958/pv_val_informativo.json` | `configs/06_app.yaml` | `2411ff9` | — (PV-val) |
| `APP_EXT` | `results/06_app/06_app_externas_20261110_144834/verificacion_externas.json` | `configs/06_app.yaml` | `0fe0a8b` | — (solo hash) |
| `APP_LAT` | `results/06_app/06_app_latencia_20261110_150137/latencia.json` | `configs/06_app.yaml` | `2126058` | — (PV-val, PD-dev, Commons) |
| `CFG_BASE` | `configs/02_modelo_base.yaml` | (es la config) | — | — |
| `CFG_EVAL` | `configs/03_brecha.yaml` | (es la config) | — | — |
| `CFG_VAR` | `configs/04_aumentacion.yaml` | (es la config) | — | — |
| `CFG_CAL` | `configs/05_calibracion.yaml` | (es la config) | — | — |
| `DECISIONES` | `docs/bitacora/decisiones.md` | — (decisiones del equipo) | — | — |
| `MAPEO` | `docs/mapeo_clases.md` | — (mapeo de clases) | — | — |
| `PROPUESTA` | `docs/propuesta_entregable1.md` | — (Entregable 1, cifras bibliográficas) | — | — |
| `INVENTARIO` | `docs/exploracion_datasets.md` | — (inventario de la sesión 2) | — | — |

Las figuras y tablas de `docs/figuras_finales/` salen de estas mismas fuentes con `scripts/figuras_finales.py`
(`configs/07_figuras_finales.yaml`, commit `9f2ccb8`, árbol limpio); `docs/figuras_finales/tablas.md` repite la lista de fuentes.

## Números

### Datos

| ID | Número | Valor | Fuente | Clave |
|---|---|---|---|---|
| D1 | PV-train, 12 clases (imágenes) | 11.968 | `PV_SPLIT` | `clase_eval=True&particion=train` |
| D2 | PV-val, 12 clases | 2.207 | `PV_SPLIT` | `clase_eval=True&particion=val` |
| D3 | PV-test, 12 clases | 2.170 | `PV_SPLIT` | `clase_eval=True&particion=test` |
| D4 | PV excluidas (tope por clase y zona de mosaic), 12 clases | 3.210 | `PV_SPLIT` | `clase_eval=True&particion=excluida` |
| D5 | Pool de PlantDoc en las 12 clases | 1.098 | `PD_SPLIT` | `estado!=fuera_de_alcance` |
| D6 | PlantDoc excluidas por duplicado con etiquetas contradictorias | 62 | `PD_SPLIT` | `estado=excluida_contradictoria` |
| D7 | PlantDoc excluidas por copia repetida de la misma clase | 30 | `PD_SPLIT` | `estado=excluida_duplicada` |
| D8 | PlantDoc excluidas por la revisión humana | 10 | `PD_SPLIT` | `estado=excluida_revision` |
| D9 | PlantDoc conservadas | 996 | `PD_SPLIT` | `estado=conservada` |
| D10 | PD-dev | 696 | `PD_SPLIT` | `particion_propuesta=dev` |
| D11 | PD-test | 300 | `PD_SPLIT` | `particion_propuesta=test` |
| D12 | PD-test que viene de la carpeta *test* oficial de PlantDoc | 24 | `PD_SPLIT` | `particion_propuesta=test&split_oficial=test` |
| D13 | Planilla de revisión: filas revisadas | 124 | `REVISION` | `revisor_1!=` |
| D14 | Planilla de revisión: `mantener` / `descartar` | 114 | `REVISION` | `revisor_1=mantener` |
| D15 | (ídem) `descartar` | 10 | `REVISION` | `revisor_1=descartar` |
| D16 | Pares de distancia 7–10 revisados como la misma foto | 3 | `PARES` | `decision_equipo=misma_foto` |
| D17 | Imágenes de PD-test por clase (mín. – máx.) | 16 – 41 | `T_BASE` | `derivado: min y max de resultados.PD-test.base.soporte_por_clase` |
| D19 | Clases con par en PlantDoc antes de quitar arañas; imágenes de arañas en PlantDoc train; en test | 13; 2; 0 | `MAPEO` | `— sección Tomate y decisión Sesión 5 (1)` |
| D18 | Imágenes de la revisión humana que caen en el test congelado de PlantDoc | 34 | `ACCESOS` | `— nota de la sesión 6` |

### Modelo y entrenamiento

| ID | Número | Valor | Fuente | Clave |
|---|---|---|---|---|
| M1 | Parámetros de MobileNetV3-Small con el cabezal de 12 clases | 933.932 | `VAR_ENT` | `parametros_totales` |
| M2 | Parámetros entrenables de la variante | 357.468 | `VAR_ENT` | `parametros_entrenables` |
| M3 | Parámetros entrenables del base (Linear 576 → 12) | 6.924 | `BASE_ENT` | `derivado: 576 × 12 + 12 (configs/02_modelo_base.yaml, cabezal GAP → Linear)` |
| M4 | Desbalance de PV-train: máx., mín., cociente | 1.500; 262; 5,73 | `BASE_ENT` | `control_desbalance.max; control_desbalance.min; control_desbalance.cociente_max_min` |
| M5 | Macro-F1 en PV-val del base (selección del checkpoint) | 0,9559 | `BASE_ENT` | `pv_val.macro_f1` |
| M6 | Época elegida del base | 68 | `BASE_ENT` | `entrenamiento.epoca_elegida` |
| M7 | Tiempo del base: extracción de features y entrenamiento del cabezal (s, CPU) | 344,6; 9,8 | `BASE_ENT` | `tiempo_extraccion_s; tiempo_entrenamiento_s` |
| M8 | Macro-F1 en PV-val de la variante | 0,9658 | `VAR_ENT` | `pv_val.macro_f1` |
| M9 | Época elegida de la variante (de 20) | 18 | `VAR_ENT` | `epoca_elegida` |
| M10 | Tiempo de la variante (s) y memoria pico de GPU (MB), Tesla T4 | 1.424; 273 | `VAR_ENT` | `tiempo_s; memoria_pico_gpu_mb` |

### Brecha, matrices y variante

| ID | Número | Valor | Fuente | Clave |
|---|---|---|---|---|
| E1 | Macro-F1 en PD-dev, geometría `aplastar` (base) | 0,124 [0,099; 0,148] | `GEO` | `regla_a.macro_f1_por_geometria.aplastar` |
| E2 | Macro-F1 en PD-dev, `resize_recorte` | 0,136 [0,111; 0,160] | `GEO` | `regla_a.macro_f1_por_geometria.resize_recorte` |
| E3 | Macro-F1 en PD-dev, `relleno` | 0,118 [0,094; 0,142] | `GEO` | `regla_a.macro_f1_por_geometria.relleno` |
| E4 | Diferencia pareada `resize_recorte` − `aplastar` | 0,012 [-0,011; 0,035] | `GEO` | `regla_a.diferencia_contra_aplastar.resize_recorte` |
| E5 | Diferencia pareada `relleno` − `aplastar` | -0,006 [-0,023; 0,011] | `GEO` | `regla_a.diferencia_contra_aplastar.relleno` |
| E6 | Macro-F1 PV-test, base | 0,937 [0,922; 0,950] | `T_BASE` | `resultados.PV-test.base_macro_f1_ic` |
| E7 | Macro-F1 PV-test sin mosaic, base | 0,951 [0,941; 0,960] | `T_BASE` | `resultados.PV-test.base_macro_f1_sin_mosaic_ic` |
| E8 | Macro-F1 PD-test, base | 0,130 [0,093; 0,168] | `T_BASE` | `resultados.PD-test.base_macro_f1_ic` |
| E9 | Brecha PV-test − PD-test, base | 0,807 [0,768; 0,845] | `T_BASE` | `brecha_macro_f1_pv_test_menos_pd_test` |
| E10 | Macro-F1 del trivial (siempre Tomato late blight): PV-test; PD-test | 0,010; 0,015 | `T_BASE` | `resultados.PV-test.trivial_macro_f1_ic.estimacion; resultados.PD-test.trivial_macro_f1_ic.estimacion` |
| E11 | Exactitud PV-test, base (no es titular) | 0,955 [0,946; 0,962] | `IC_BASE` | `pv_test.exactitud` |
| E12 | Exactitud PD-test, base | 0,187 [0,157; 0,220] | `IC_BASE` | `pd_test.exactitud` |
| E13 | Exactitud PD-dev, base | 0,175 [0,152; 0,198] | `IC_BASE` | `pd_dev.exactitud` |
| E14 | PD-test, base: predicciones de Tomato late blight (de 300) | 175 | `T_BASE` | `derivado: suma de la columna Tomato___Late_blight de resultados.PD-test.base.matriz_confusion` |
| E15 | PD-test, base: errores atraídos por Tomato late blight; errores totales | 147; 244 | `T_BASE` | `resultados.PD-test.base.analisis_errores.errores_hacia_esa_clase; resultados.PD-test.base.analisis_errores.errores` |
| E16 | PD-test, base: Potato late blight predicha como Tomato late blight (de 27) | 24 | `T_BASE` | `derivado: celda [Potato___Late_blight, Tomato___Late_blight] de la matriz de PD-test` |
| E17 | PD-test, base: filas cuya diagonal es el máximo (de 12) | 1 | `T_BASE` | `derivado: argmax de cada fila de la matriz de PD-test` |
| E18 | Recall de Tomato late blight en PD-test, base | 0,933 [0,833; 1,000] | `IC_BASE` | `pd_test.por_clase.Tomato___Late_blight.recall` |
| E19 | Precisión de Tomato late blight en PD-test, base | 0,160 [0,142; 0,179] | `IC_BASE` | `pd_test.por_clase.Tomato___Late_blight.precision` |
| E20 | PD-test, base: clases con límite superior del recall ≤ 0,25 (de 12) | 7 | `IC_BASE` | `derivado: cuenta de pd_test.por_clase.*.recall.ic_alto ≤ 0,25` |
| E21 | Recall de mosaic en PV-test, base | 0,686 [0,543; 0,829] | `IC_BASE` | `pv_test.por_clase.Tomato___Tomato_mosaic_virus.recall` |
| E22 | PD-dev, base: predicciones de Tomato late blight; n | 417; 696 | `DIAG` | `pd_dev.clase_mas_predicha.n; pd_dev.n` |
| E23 | PD-dev, base: confianza media en aciertos; en errores | 0,874; 0,788 | `DIAG` | `pd_dev.confianza_media.aciertos; pd_dev.confianza_media.errores` |
| E24 | Macro-F1 PV-test, variante | 0,956 [0,943; 0,967] | `T_VAR` | `resultados.PV-test.base_macro_f1_ic` |
| E25 | Macro-F1 PV-test sin mosaic, variante | 0,966 [0,957; 0,974] | `T_VAR` | `resultados.PV-test.base_macro_f1_sin_mosaic_ic` |
| E26 | Macro-F1 PD-test, variante | 0,126 [0,087; 0,165] | `T_VAR` | `resultados.PD-test.base_macro_f1_ic` |
| E27 | Brecha PV-test − PD-test, variante | 0,830 [0,792; 0,867] | `T_VAR` | `brecha_macro_f1_pv_test_menos_pd_test` |
| E28 | Diferencia pareada variante − base, PV-test | 0,019 [0,004; 0,033] | `COMP` | `resultados.PV-test.diferencia_variante_menos_base` |
| E29 | Diferencia pareada variante − base, PD-test | -0,004 [-0,030; 0,023] | `COMP` | `resultados.PD-test.diferencia_variante_menos_base` |
| E30 | Macro-F1 PD-dev, variante (informativo) | 0,146 [0,121; 0,171] | `VAR_DEV` | `comparacion_con_base.variante` |
| E31 | Diferencia pareada variante − base, PD-dev (informativo) | 0,022 [-0,003; 0,047] | `VAR_DEV` | `comparacion_con_base.diferencia_variante_menos_base` |
| E32 | Exactitud PV-test, variante (no es titular) | 0,967 [0,959; 0,974] | `IC_VAR` | `pv_test.exactitud` |
| E33 | Exactitud PD-test, variante | 0,173 [0,137; 0,210] | `IC_VAR` | `pd_test.exactitud` |
| E34 | PD-test, variante: predicciones de Tomato late blight (de 300) | 128 | `T_VAR` | `derivado: suma de la columna Tomato___Late_blight de resultados.PD-test.base.matriz_confusion` |
| E35 | Recall de Tomato late blight en PD-test, variante | 0,600 [0,433; 0,767] | `IC_VAR` | `pd_test.por_clase.Tomato___Late_blight.recall` |
| E36 | PD-test, variante: Potato late → Tomato late (de 27) | 22 | `T_VAR` | `derivado: celda [Potato___Late_blight, Tomato___Late_blight] de la matriz de PD-test` |
| E37 | PD-test, variante: filas cuya diagonal es el máximo (de 12) | 2 | `T_VAR` | `derivado: argmax de cada fila de la matriz de PD-test` |
| E39 | PD-test, base: Potato early blight predicha como Tomato late blight (de 29); Potato late blight (de 27) | 14; 29; 24; 27 | `T_BASE` | `derivado: celdas [Potato___Early_blight, Tomato___Late_blight] y [Potato___Late_blight, Tomato___Late_blight] y sumas de esas filas de la matriz de PD-test` |
| E40 | Semiancho aproximado del IC del macro-F1 en PD-test | 0,04 | `T_BASE` | `derivado: (0,168 − 0,093) / 2 ≈ 0,04, de resultados.PD-test.base_macro_f1_ic (E8)` |
| E38 | PD-test, variante: clases con límite superior del recall ≤ 0,25 (de 12) | 8 | `IC_VAR` | `derivado: cuenta de pd_test.por_clase.*.recall.ic_alto ≤ 0,25` |

### CAM (cualitativo: ilustración, no evidencia)

| ID | Número | Valor | Fuente | Clave |
|---|---|---|---|---|
| X1 | Imágenes de la muestra fija (24 de PD-dev + 12 de PV-val) | 36 | `CAM_BASE` | `n_imagenes` |
| X2 | Aciertos del base en la muestra: PD-dev (de 24); PV-val (de 12) | 9; 10 | `CAM_BASE` | `derivado: cuenta de muestra.* con clase_real = clase_predicha, por origen` |
| X3 | Aciertos de la variante en la misma muestra: PD-dev; PV-val | 8; 11 | `CAM_VAR` | `aciertos.plantdoc-dev; aciertos.plantvillage-val` |
| X4 | Imágenes de PD-dev de la muestra que vienen de la carpeta *test* oficial | 2 | `CAM_BASE` | `n_plantdoc_dev_de_la_carpeta_test_oficial` |

### Calibración y abstención

| ID | Número | Valor | Fuente | Clave |
|---|---|---|---|---|
| C1 | Regla 1: macro-F1 en PV-val del base; de la variante | 0,9559; 0,9658 | `CAL_DEV` | `regla_modelo.macro_f1_pv_val.base; regla_modelo.macro_f1_pv_val.variante` |
| C2 | Temperatura final (todo PD-dev) | 14,97 | `CAL_DEV` | `temperatura_final` |
| C3 | Temperatura por partición (mín.; máx.) | 13,87; 15,56 | `CAL_DEV` | `temperatura_por_particion.0; temperatura_por_particion.3` |
| C4 | NLL en PD-dev con T = 1; con T final (diagnóstico, sin IC) | 8,89; 2,34 | `CAL_DEV` | `nll_pd_dev.T_1; nll_pd_dev.T_final` |
| C5 | ECE en PD-dev, antes (T = 1) | 0,628 [0,601; 0,655] | `CAL_DEV` | `antes_T_1.ece` |
| C6 | ECE en PD-dev, después (fuera de partición) | 0,040 [0,020; 0,066] | `CAL_DEV` | `despues_fuera_de_particion.ece` |
| C7 | Cobertura con τ = 0,5 en PD-dev, antes | 0,943 [0,925; 0,958] | `CAL_DEV` | `antes_T_1.cobertura` |
| C8 | Cobertura con τ = 0,5 en PD-dev, después | 0,000 [0,000; 0,000] | `CAL_DEV` | `despues_fuera_de_particion.cobertura` |
| C9 | Respondidas en PD-dev, antes; después | 656 [644; 667]; 0 [0; 0] | `CAL_DEV` | `antes_T_1.n_respondidas; despues_fuera_de_particion.n_respondidas` |
| C10 | Acierto sobre lo respondido en PD-dev, antes | 0,210 [0,186; 0,236] | `CAL_DEV` | `antes_T_1.acierto_respondidas` |
| C11 | Errores de alta confianza en PD-dev, antes (fracción del total) | 0,744 [0,716; 0,772] | `CAL_DEV` | `antes_T_1.errores_alta_confianza` |
| C12 | Exactitud de la variante en PD-dev | 0,210 [0,185; 0,234] | `CAL_DEV` | `antes_T_1.exactitud` |
| C13 | ECE en PD-test, antes (T = 1) | 0,676 [0,635; 0,717] | `CAL_TEST` | `antes_T_1.ece` |
| C14 | ECE en PD-test, después (T = 14,97) | 0,021 [0,015; 0,067] | `CAL_TEST` | `despues_T_final.ece` |
| C15 | Cobertura con τ = 0,5 en PD-test, antes | 0,960 [0,937; 0,980] | `CAL_TEST` | `antes_T_1.cobertura` |
| C16 | Cobertura con τ = 0,5 en PD-test, después | 0,000 [0,000; 0,000] | `CAL_TEST` | `despues_T_final.cobertura` |
| C17 | Respondidas en PD-test, antes; después | 288 [281; 294]; 0 [0; 0] | `CAL_TEST` | `antes_T_1.n_respondidas; despues_T_final.n_respondidas` |
| C18 | Acierto sobre lo respondido en PD-test, antes | 0,181 [0,141; 0,220] | `CAL_TEST` | `antes_T_1.acierto_respondidas` |
| C19 | Errores de alta confianza en PD-test, antes (fracción del total) | 0,787 [0,743; 0,827] | `CAL_TEST` | `antes_T_1.errores_alta_confianza` |
| C20 | Errores de alta confianza en PD-test, después | 0,000 [0,000; 0,000] | `CAL_TEST` | `despues_T_final.errores_alta_confianza` |
| C28 | Curva de cobertura en PD-dev sin calibrar: máximo acierto sobre lo respondido; su τ; su cobertura | 0,272; 0,98; 0,328 | `CAL_DEV` | `curva_cobertura.antes_T_1.98.acierto_respondidas; curva_cobertura.antes_T_1.98.tau; curva_cobertura.antes_T_1.98.cobertura` |
| C29 | Fiabilidad de PD-test, T calibrada: imágenes del bin cuyo acierto es 1 | 2 | `CAL_TEST` | `fiabilidad.despues_T_final.5.n` |
| C30 | NLL de una predicción uniforme sobre 12 clases (ln 12) | 2,48 | `CAL_DEV` | `derivado: ln(12) = 2,4849` |
| C21 | τ | 0,5 | `CAL_TEST` | `tau` |
| C22 | PV-val con T = 1: cobertura | 0,998 [0,995; 1,000] | `APP_PV` | `referencia_sin_calibrar_T1.cobertura` |
| C23 | PV-val con T = 1: respondidas | 2.202 | `APP_PV` | `referencia_sin_calibrar_T1.n_respondidas.estimacion` |
| C24 | PV-val con T = 1: acierto sobre lo respondido | 0,971 [0,964; 0,978] | `APP_PV` | `referencia_sin_calibrar_T1.acierto_respondidas` |
| C25 | PV-val con T = 14,97: cobertura | 0,000 [0,000; 0,000] | `APP_PV` | `resultados.cobertura` |
| C26 | PV-val: exactitud de la variante | 0,971 [0,963; 0,977] | `APP_PV` | `resultados.exactitud` |
| C27 | PV-val con T = 14,97: confianza máxima calibrada (mín.; mediana; máx.) | 0,116; 0,207; 0,427 | `APP_PV` | `confianza_maxima_calibrada.min; confianza_maxima_calibrada.p50; confianza_maxima_calibrada.max` |

### Aplicación

| ID | Número | Valor | Fuente | Clave |
|---|---|---|---|---|
| A1 | Muestra de latencia: imágenes de PV-val; PD-dev; Commons | 50; 50; 4 | `APP_LAT` | `parametros.imagenes_por_conjunto.PV-val; parametros.imagenes_por_conjunto.PD-dev; parametros.imagenes_por_conjunto.Commons` |
| A2 | PV-val sin CAM: p50; p95 (ms) | 14,1; 16,4 | `APP_LAT` | `resultados_ms.sin_cam.PV-val.p50_ms; resultados_ms.sin_cam.PV-val.p95_ms` |
| A3 | PV-val con CAM: p50; p95 (ms) | 19,8; 22,7 | `APP_LAT` | `resultados_ms.con_cam.PV-val.p50_ms; resultados_ms.con_cam.PV-val.p95_ms` |
| A4 | PD-dev sin CAM: p50; p95 (ms) | 24,0; 165,4 | `APP_LAT` | `resultados_ms.sin_cam.PD-dev.p50_ms; resultados_ms.sin_cam.PD-dev.p95_ms` |
| A5 | PD-dev con CAM: p50; p95 (ms) | 31,4; 172,3 | `APP_LAT` | `resultados_ms.con_cam.PD-dev.p50_ms; resultados_ms.con_cam.PD-dev.p95_ms` |
| A6 | Commons sin CAM: p50; p95 (ms) | 23,7; 33,6 | `APP_LAT` | `resultados_ms.sin_cam.Commons.p50_ms; resultados_ms.sin_cam.Commons.p95_ms` |
| A7 | Commons con CAM: p50; p95 (ms) | 32,3; 39,9 | `APP_LAT` | `resultados_ms.con_cam.Commons.p50_ms; resultados_ms.con_cam.Commons.p95_ms` |
| A8 | Total, solo p50: sin CAM; con CAM (ms) | 15,7; 22,0 | `APP_LAT` | `resultados_ms.sin_cam.todas.p50_ms; resultados_ms.con_cam.todas.p50_ms` |
| A9 | Sobrecosto del CAM, p50 total (ms) | 6,4 | `APP_LAT` | `sobrecosto_cam_p50_ms` |
| A10 | Pesos: bytes; MB | 6.759.450; 6,76 | `APP_LAT` | `pesos.bytes; pesos.MB` |
| A11 | Carga del modelo (s) | 0,10 | `APP_LAT` | `carga_del_modelo_s` |
| A12 | Memoria pico del proceso (MB) | 596 | `APP_LAT` | `memoria_pico_cpu_mb` |
| A13 | Hilos de torch | 6 | `APP_LAT` | `entorno.hilos_torch` |
| A14 | Imágenes contra las que se buscaron duplicados de Commons: PlantVillage; PlantDoc | 108.611; 2.578 | `APP_EXT` | `comparadas.PlantVillage (color y segmented); comparadas.PlantDoc (train y test oficiales)` |
| A15 | Imágenes de Commons: probadas; duplicadas (descartadas); válidas | 6; 2; 4 | `APP_EXT` | `derivado: imagenes (6) y resultado_duplicados de cada una (2 "DUPLICADA")` |
| A17 | Llamadas lentas que dominan el p95 global (de 312 por serie) | 18 | `DECISIONES` | `— Sesión 12: cómo se reporta la latencia` |
| A16 | Umbral de phash | 6 | `APP_EXT` | `umbral_phash` |

### Parámetros fijados en las configs y en las decisiones

| ID | Número | Valor | Fuente | Clave |
|---|---|---|---|---|
| P1 | Clases de entrenamiento y evaluación | 12 | `CFG_BASE` | `derivado: largo de clases` |
| P2 | Tamaño de entrada (px) | 224 | `CFG_BASE` | `preproceso.tamano` |
| P3 | Semilla | 42 | `CFG_BASE` | `semilla` |
| P4 | Base: LR; batch; máx. épocas; paciencia | 0,001; 256; 100; 10 | `CFG_BASE` | `entrenamiento.lr; entrenamiento.batch; entrenamiento.max_epocas; entrenamiento.paciencia` |
| P5 | Variante: bloques descongelados de `features` | 11; 12 | `CFG_VAR` | `finetune.bloques_entrenables.0; finetune.bloques_entrenables.1` |
| P6 | Variante: LR; batch; máx. épocas; paciencia | 0,0001; 64; 20; 5 | `CFG_VAR` | `finetune.lr; finetune.batch; finetune.max_epocas; finetune.paciencia` |
| P7 | Aumentación: escala del recorte; rotación (grados); brillo; contraste; saturación | 0,6; 1,0; 20; 0,3; 0,3; 0,3 | `CFG_VAR` | `aumentacion.escala_recorte.0; aumentacion.escala_recorte.1; aumentacion.rotacion_grados; aumentacion.brillo; aumentacion.contraste; aumentacion.saturacion` |
| P8 | Bootstrap: réplicas; nivel; semilla | 2.000; 0,95; 42 | `CFG_EVAL` | `bootstrap.replicas; bootstrap.nivel; bootstrap.semilla` |
| P9 | Muestra de CAM: por clase de PD-dev; por clase de PV-val | 2; 1 | `CFG_EVAL` | `cam.por_clase_pd_dev; cam.por_clase_pv_val` |
| P10 | Grilla de T: desde; hasta; valores | 0,5; 1.000; 200 | `CFG_CAL` | `temperatura_grilla.desde; temperatura_grilla.hasta; temperatura_grilla.n` |
| P11 | Bins del ECE; particiones para el ECE fuera de partición | 15; 5 | `CFG_CAL` | `ece_bins; particiones.k` |
| P12 | Partición de PlantDoc dev/test (%); umbral de phash para duplicados | 70/30; 6 | `DECISIONES` | `— Sesión 5 (2)` |
| P13 | Zona de descarte de mosaic (imágenes); tope de PV-train por clase | 20; 1.500 | `DECISIONES` | `— Sesión 5 (3) y (4)` |

### Referencias e inventario (no son resultados del proyecto)

| ID | Número | Valor | Fuente | Clave |
|---|---|---|---|---|
| R1 | Mohanty et al. (2016): exactitud en el test retenido de PlantVillage; en imágenes de otras condiciones (%) | 99,35; 31,4 | `PROPUESTA` | `— §1, cita bibliográfica` |
| R2 | PlantVillage color (commit fijado): imágenes; clases | 54.305; 38 | `INVENTARIO` | `— §1` |
| R3 | PlantDoc (commit fijado): imágenes en train + test; carpetas de clase | 2.578; 28 | `INVENTARIO` | `— §2` |
| R5 | Propuesta: imágenes por clase del test oficial de PlantDoc | 8; 12 | `PROPUESTA` | `— §3 y §7` |
| R6 | Propuesta: precisión mínima de la "cobertura útil" | 0,80 | `PROPUESTA` | `— §6, métricas de aplicación` |
| R4 | PlantDoc según el paper: imágenes; clases | 2.598; 27 | `PROPUESTA` | `— §3 (cita de Singh et al., 2020)` |
