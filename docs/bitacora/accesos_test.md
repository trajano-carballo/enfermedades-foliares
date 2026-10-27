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

**Nota (2026-10-08, sesión 5):** sin ningún modelo ni métrica de desempeño. (1) Mirada a ojo: `docs/figuras_sesion5/ejemplos_contradictorios.png` incluye **1 imagen de la carpeta test oficial** de
PlantDoc (`Potato leaf late blight/1421_0.jpeg…`, ya vista en la sesión 4); la grilla `pares_7_10.png` no tiene ninguna imagen de test oficial. (2) Las 102 imágenes de test oficial de las 12 clases entraron
al cálculo de hashing perceptual (caché de la sesión 4) y a la depuración: 6 quedan excluidas por tener un gemelo con etiqueta contradictoria, 96 se reparten entre dev (72) y test propuesto (24). (3) No se
generaron ni miraron imágenes del test propuesto (303 imágenes): la partición es un candidato y solo se usaron rutas y estados. Notebook `notebooks/01b_depuracion_plantdoc.ipynb`.

**Nota de corrección (2026-10-09, sesión 5d) a la nota de la sesión 5:** esa nota dice que la grilla `pares_7_10.png` "no tiene ninguna imagen de test oficial"; sigue siendo cierto (los 3 pares son de la carpeta *train* oficial), pero la grilla
**sí contiene 2 imágenes que caen en el test propuesto** de PlantDoc, ambas de `Potato leaf early blight`: `early-blight-or-target-spot-alternaria-solani-lesions-on-a-tomato-by9j8r.jpg` (par 1, tomate early ↔ papa early) y `80109626.jpg` (par 2).
Por lo tanto el punto (3) de aquella nota ("no se generaron ni miraron imágenes del test propuesto") no es exacto: la figura, generada en la sesión 5, las incluye (2 de las 303). Sin ningún modelo ni métrica de desempeño.

**Nota (2026-10-09, sesión 5c):** Sesión 5c (reconstrucción del entorno en PC personal). Sin modelos ni métricas. Se recalcularon sobre data/raw/plantdoc (commit 5467f601…) el hash perceptual (phash, 64 bits, 8 variantes: 4 rotaciones × espejado) y los grupos de duplicados de las 12 clases de alcance, vía los notebooks 00c_chequeos_plantdoc y 01b_depuracion_plantdoc. Esto incluye las 102 imágenes de la carpeta test oficial de esas clases. Fue un cómputo automático sobre el contenido de los archivos: ninguna imagen de test (oficial o propuesto) fue abierta ni mirada, y no se calculó ninguna métrica de desempeño. Se regeneraron las grillas de pares de las sesiones 4 y 5 y las hojas de contacto de revisión (Tomato leaf, Bell_pepper leaf, que incluyen 16 imágenes de test oficial) sin abrirlas. Reproduce los números de la sesión 5: 1.098 en el pool, 60 excluidas contradictorias, 28 duplicadas, 1.010 conservadas, dev 707 / test 303. En la misma sesión se rescataron por hash de blob los 101 archivos no representables en NTFS (sin cambios en el mapeo versionado).

**Nota (2026-10-09, sesión 5d):** sin ningún modelo ni métrica de desempeño. Se re-ejecutaron `01` y `01b` (copias en `data/interim/ejecuciones_pc_personal/`): las 102 imágenes de test oficial entraron a la depuración (hashes desde la caché existente) y a la selección de los pares 7–10.
Se regeneraron, sin abrirlas, `docs/figuras_sesion5/ejemplos_contradictorios.png` (1 imagen de test oficial) y `docs/figuras_sesion5/pares_7_10.png` (0 de test oficial, 2 del test propuesto; ver la corrección de arriba); los PNG versionados se restauraron a su versión del commit (los píxeles regenerados eran idénticos).
Ninguna imagen de test (oficial o propuesto) fue abierta ni mirada en esta sesión; no se generó ninguna otra figura con imágenes de test y no se tocó `data/splits/`.

**Nota (2026-10-23, sesión 6; la revisión se hizo antes del 2026-10-15, fecha del commit 4cca764):** revisión humana de etiquetas de PlantDoc, sin ningún modelo ni métrica de desempeño. Se miraron **130 imágenes**: 124 de la planilla (`Tomato leaf` 63 y `Bell_pepper leaf` 61) y 6 de los 3 pares de distancia 7–10. Según el manifiesto y el candidato de la 5d, **16 son de la carpeta test oficial** (las 8 + 8 de la planilla; los pares son todos de train oficial) y **39 caen en el test propuesto** (37 de la planilla y 2 de los pares); 2 imágenes están en ambos conjuntos, así que son 53 las distintas que tocan algún test. Modalidad: revisión asistida por IA en el chat web, con la decisión del usuario fila por fila (un solo revisor, D1 y D3). El usuario le pasó las imágenes a esa IA, que las describió y propuso una decisión; la decisión final fue del usuario. No se entrenó, evaluó ni calibró ningún modelo. Los CSV resultantes están en `docs/bitacora/`. La sesión 6 solo leyó los CSV y las rutas; no abrió imágenes.

Sobre el candidato de la sesión 6 (el que se congela): de las 130 imágenes miradas, 34 caen en el test propuesto de PlantDoc (34 de la planilla, 0 de los pares), 1 está además en el test oficial; 49 distintas tocan algún test. Este es el número que se publica como exposición del test congelado.

**Nota (2026-10-27, sesión 7):** sin ningún modelo ni métrica de desempeño sobre ningún test. (1) `empaquetar_subconjunto.py` y `verificar_paquete.py` leyeron los bytes (sha256 y copia al `.tar`) de todos los archivos del subconjunto, que incluyen las 2.170 imágenes de PlantVillage-test (color) y sus 2.170 segmented, y de PlantDoc las 102 de la carpeta test oficial y las 300 del test propuesto (378 distintas). Fue un cómputo automático sobre el contenido de los archivos: ninguna imagen de test fue decodificada, abierta ni mirada. (2) La prueba de humo (corrida local en CPU; la de Colab la ejecuta el usuario) solo lee imágenes de PlantVillage-train: `foliares.data.lote_humo` no tiene parámetro de partición, rechaza rutas que no sean de PlantVillage y tiene tests que lo verifican; no usa PlantDoc ni ningún test. (3) No se tocó `data/splits/`.

**Nota (2026-10-27, sesión 7, Colab):** sin ningún modelo ni métrica de desempeño sobre ningún test. En las tres sesiones de Colab (commits clonados `e941355`, `bdd9805` y `161b553`) las celdas 5–6 del notebook copiaron desde Drive, descomprimieron y verificaron por sha256 (`verificar_paquete.py`) los mismos archivos del paquete, que incluyen las 2.170 imágenes de PlantVillage-test (color) y sus 2.170 segmented, y de PlantDoc las 102 de la carpeta test oficial y las 300 del test propuesto (378 distintas): fue un cómputo sobre los bytes de los archivos, sin decodificar, abrir ni mirar ninguna imagen de test. La prueba de humo de Colab (fase 1 en `bdd9805`; fases 1 y 2 en `161b553`; la primera sesión no llegó a esa celda) solo leyó PlantVillage-train (`foliares.data.lote_humo` no admite otra partición). **Corrección** a la nota anterior de esta misma fecha, que decía que la prueba de humo de Colab "la ejecuta el usuario": ya se ejecutó, en las sesiones indicadas, con ese alcance. No se tocó `data/splits/`.
