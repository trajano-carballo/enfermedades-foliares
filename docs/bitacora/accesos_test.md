# Registro de accesos a los conjuntos de test

Cada evaluación sobre un test congelado (PlantVillage-test o PlantDoc) se registra acá.
El total se publica en el documento final (mitigación de sobreajuste al test).

| # | Fecha | Test | Modelo / config | Commit | Motivo |
|---|---|---|---|---|---|

**Nota (2026-09-25, sesión 2):** en esta sesión se inventarió PlantDoc (conteos por clase,
integridad de archivos, resoluciones) pero **no se evaluó ningún modelo ni se calculó
ninguna métrica de desempeño** sobre ninguno de los dos tests. No corresponde ninguna fila
en la tabla de arriba. Detalle en `docs/exploracion_datasets.md`.

**Nota (2026-10-01, sesión 3):** Sesión 3: inspección visual de n=45 imágenes de test de PlantDoc
(3 al azar por clase candidata, semilla 42, solo para verificar etiquetas y tipo de contenido) y
estadísticas de resolución/relación de aspecto de las 131 imágenes de test de las clases candidatas
(solo cabeceras de archivo), sin ningún modelo. Además, las 102 imágenes de test de tomate/papa/pimiento
entraron al cálculo de hashing perceptual (casi-duplicados train↔test y contra PlantVillage), también
sin ningún modelo ni métrica de desempeño. Notebook `notebooks/00b_chequeos_previos.ipynb`.
