# Clasificador de enfermedades foliares: brecha laboratorio → campo

Proyecto Integrador AIAI — Aplicaciones de Aprendizaje Automático (LIDIA, UTEC ITRN).
César Carballo · Juan Lucas Pimentel · Victor Uría — Prof. Juan Pedro de León.

## Qué es

Entrenamos un clasificador de enfermedades foliares sobre **PlantVillage** (laboratorio) y lo
evaluamos sobre **PlantDoc** (campo), que nunca se usa en entrenamiento. El resultado principal
no es la accuracy de laboratorio sino la **brecha ID→OOD en macro-F1**, por qué ocurre y cuánto
se recupera con dos intervenciones: aumentación dirigida y supresión de fondo. La
explicabilidad se resuelve con **CAM** (no se usa Grad-CAM en ninguna etapa). El subconjunto
común de clases parte de tomate y papa (maíz queda excluido), evaluando si sumar pimiento.

El producto final es un asistente de orientación (nunca diagnóstico) que se **abstiene** cuando
no está en condiciones de opinar. Propuesta completa: [`docs/propuesta_entregable1.md`](docs/propuesta_entregable1.md).

## Estado

Etapa 0 — estructura del repositorio. Ver [`handoff.md`](handoff.md) para el estado al día.

## Estructura

```
data/
  raw/plantvillage, raw/plantdoc   descargas originales (no versionadas)
  interim/                         datos tras controles de integridad y mapeo
  processed/                       tensores/features listos para entrenar
  splits/                          particiones congeladas (CSV, SÍ versionadas)
configs/                           un YAML por experimento
notebooks/                         exploración y reportes, numerados por etapa (01_, 02_…)
src/foliares/
  data/        descarga, integridad, mapeo de clases, deduplicado perceptual, splits
  models/      backbone MobileNetV3-Small + cabezal GAP→Linear
  training/    loop de entrenamiento, aumentaciones
  evaluation/  macro-F1, matrices de confusión, bootstrap, calibración, cobertura/abstención
  explain/     CAM (deploy y análisis; no se usa Grad-CAM)
scripts/                           puntos de entrada por línea de comandos
app/                               demo en Gradio
artifacts/checkpoints/             pesos entrenados (no versionados)
results/metrics, results/figures   salidas versionadas que respaldan el documento
docs/                              propuesta, bitácoras, documento final
tests/                             tests unitarios
```

## Reglas del experimento

1. PlantDoc **nunca** entra al entrenamiento ni a la validación.
2. Las particiones se congelan en `data/splits/` al cierre de la etapa 1 y no se regeneran.
3. Toda evaluación sobre un test se registra en [`docs/bitacora/accesos_test.md`](docs/bitacora/accesos_test.md).
4. Toda decisión metodológica se registra en [`docs/bitacora/decisiones.md`](docs/bitacora/decisiones.md).
5. La accuracy en PlantVillage no se reporta como resultado titular.

## Datos

- PlantVillage (Hughes y Salathé, 2015) — Kaggle `emmarex/plantdisease`.
- PlantDoc (Singh et al., 2020) — GitHub `pratikkayal/PlantDoc-Dataset`.

Uso académico, sin redistribución: los datos no se suben al repositorio.

## Entorno

Pendiente de definir en etapa 0 (Colab gratuito para entrenamiento, local para la app).
