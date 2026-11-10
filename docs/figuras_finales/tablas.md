# Tablas del documento final

Generado por `scripts/figuras_finales.py --config configs/07_figuras_finales.yaml` desde los JSON de `results/` (sin cómputo nuevo). IC 95 % bootstrap estratificado, 2.000 réplicas, semilla 42.

Generado desde el commit `9f2ccb8`; cambios sin commitear al generar: no.

## Macro-F1 base vs. variante (filas 1–4)

| | Base | Variante | Diferencia pareada (variante − base) |
|---|---|---|---|
| PV-test (n = 2.170) | 0,937 [0,922; 0,950] | 0,956 [0,943; 0,967] | 0,019 [0,004; 0,033] |
| PV-test sin mosaic | 0,951 [0,941; 0,960] | 0,966 [0,957; 0,974] | no medido |
| PD-test (n = 300) | 0,130 [0,093; 0,168] | 0,126 [0,087; 0,165] | -0,004 [-0,030; 0,023] |
| Brecha (PV-test − PD-test) | 0,807 [0,768; 0,845] | 0,830 [0,792; 0,867] | no medido |
| Trivial PD-test (siempre Tomato late blight) | 0,015 [0,015; 0,015] | 0,015 [0,015; 0,015] | — |

## Calibración y abstención: PV-val contra PD-test (fila 5)

| | PV-val, T = 1 | PV-val, T = 14,97 | PD-test, T = 1 | PD-test, T = 14,97 |
|---|---|---|---|---|
| ECE (15 bins) | no medido | no medido | 0,676 [0,635; 0,717] | 0,021 [0,015; 0,067] |
| Cobertura con τ | 0,998 [0,995; 1,000] | 0,000 [0,000; 0,000] | 0,960 [0,937; 0,980] | 0,000 [0,000; 0,000] |
| Respondidas | 2.202 [2.197; 2.206] | 0 [0; 0] | 288 [281; 294] | 0 [0; 0] |
| Acierto sobre lo respondido | 0,971 [0,964; 0,978] | no definido (0 respondidas) | 0,181 [0,141; 0,220] | no definido (0 respondidas) |
| Errores de alta confianza (fracción del total) | no medido | no medido | 0,787 [0,743; 0,827] | 0,000 [0,000; 0,000] |
| Exactitud (no cambia con T) | 0,971 [0,963; 0,977] | 0,971 [0,963; 0,977] | 0,173 [0,137; 0,210] | 0,173 [0,137; 0,210] |

τ = 0,5. n: PV-val 2.207, PD-test 300. PV-val no es un test (informativo); PD-test es la pasada única de la fila 5.

## Latencia y tamaño de la app

| Conjunto | p50 sin CAM | p95 sin CAM | p50 con CAM | p95 con CAM |
|---|---|---|---|---|
| PV-val (150 llamadas) | 14,1 | 16,4 | 19,8 | 22,7 |
| PD-dev (150 llamadas) | 24,0 | 165,4 | 31,4 | 172,3 |
| Commons (12 llamadas) | 23,7 | 33,6 | 32,3 | 39,9 |
| Total (312 llamadas) | 15,7 | — | 22,0 | — |

ms, CPU AMD Ryzen 5 5500U with Radeon Graphics, 6 hilos de torch, torch 2.14.1+cpu; sin la red de Gradio. Pesos: 6.759.450 bytes (6,76 MB). Carga del modelo: 0,10 s. Sobrecosto del CAM (p50 total): 6,4 ms. Memoria pico del proceso: 596 MB.

## Fuentes

| Clave | JSON | Commit | Árbol limpio |
|---|---|---|---|
| tests_base | `results/03_brecha/03_brecha_tests_20261105_081814/tests.json` | `09a1e3e` | sí |
| tests_variante | `results/04_aumentacion/04_aumentacion_tests_20261106_210645/tests.json` | `9d0f5a7` | sí |
| comparacion | `results/04_aumentacion/04_aumentacion_comparacion_20261106_211523/comparacion_base.json` | `6fa21ed` | sí |
| ic_base | `results/03_brecha/03_brecha_ic_20261105_085644/ic_por_clase.json` | `3437441` | sí |
| ic_variante | `results/04_aumentacion/04_aumentacion_ic_20261106_211534/ic_por_clase.json` | `6fa21ed` | sí |
| calibracion_dev | `results/05_calibracion/05_calibracion_dev_20261107_095030/calibracion_dev.json` | `6e5196a` | sí |
| calibracion_test | `results/05_calibracion/05_calibracion_test_20261107_095821/calibracion_test.json` | `5bff0d7` | sí |
| pv_val_app | `results/06_app/06_app_pv_val_20261110_093958/pv_val_informativo.json` | `2411ff9` | sí |
| latencia | `results/06_app/06_app_latencia_20261110_150137/latencia.json` | `2126058` | sí |

Grillas de CAM copiadas (no se regeneran desde results/): `docs/figuras_sesion9/cam_grilla_clases_1a6.png`, `docs/figuras_sesion9/cam_grilla_clases_7a12.png`, `docs/figuras_sesion10/cam_variante_clases_1a6.png`, `docs/figuras_sesion10/cam_variante_clases_7a12.png`
