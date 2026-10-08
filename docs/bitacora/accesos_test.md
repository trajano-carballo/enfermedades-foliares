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

**Nota (2026-10-08, sesión 4):** sin ningún modelo ni métrica de desempeño. (1) Las 102 imágenes de test de las 13 carpetas candidatas de PlantDoc
entraron al cálculo de hashing perceptual (8 variantes) y a la comparación de pares con train y entre sí. (2) **Miradas a ojo** en grillas de pares
(`docs/figuras_sesion4/pares_*.png`): 12 imágenes distintas de test (las que tienen un gemelo en train a distancia ≤ 6), y el par `irish-blight-symptoms…`
(su copia de test es una de esas 12). (3) Se generaron hojas de contacto con 16 imágenes de test (8 de `Tomato leaf`, 8 de `Bell_pepper leaf`) para la revisión
humana de etiquetas; de ellas yo vi **una** (`TL-062`, al verificar el formato de la hoja) y las otras 15 las verán los revisores. (4) La partición propuesta para
PlantVillage no tocó ningún test de PlantDoc: el test de PlantVillage es un candidato en `data/interim/` (no congelado) y no se evaluó nada sobre él.
Notebooks `notebooks/00c_chequeos_plantdoc.ipynb` y `notebooks/01_preparacion_datos.ipynb`.
