# Cierre de datos — sesión 5 (candidatos finales; el equipo congela)

> Candidatos, **nada congelado**: `data/splits/` sigue con solo `.gitkeep`. Sin modelos, sin métricas, ninguna carpeta de PlantDoc renombrada.
> Datasets en los commits fijados (PlantVillage `7f7ecc7e…`, PlantDoc `5467f601…`), verificados con `git log -1`; no se bajó nada.
> Código nuevo: `src/foliares/data/{plantdoc_depurado,congelar,paquete}.py`, `scripts/{congelar_splits,empaquetar_subconjunto,verificar_paquete}.py`.
> Notebooks: `01_preparacion_datos.ipynb` (sección 2c-bis), `01b_depuracion_plantdoc.ipynb`, `colab_arranque.ipynb` (sin ejecutar).
> Configs: `01b_depuracion_plantdoc.yaml`, `congelar_splits.yaml`, `paquete_subconjunto.yaml`. Tests: **43 pasan** (`python -m pytest`).
> **Actualización 2026-10-09 (sesión 5d):** las decisiones D1–D3 de `decisiones.md` superan lo que este documento dice de la revisión humana y de los pares 7–10: la planilla la completa **un** revisor
> (`revisor_1`; ya no "ambos"), los pares 7–10 los revisa un revisor (`misma_foto` se une a los grupos de duplicados) y el resultado es COMPLETO solo con ambas cosas completas. Donde abajo se lee "ambos revisores"
> o "solo se listan", vale D1/D3. Las unidades "MB" y "GB" de este documento son **MiB y GiB** (2^20 y 2^30 bytes; ver §4).

## Resumen

1. **PlantVillage, 12 clases:** train / val / test = **11.968 / 2.207 / 2.170**, exactamente lo esperado. Quitar Target Spot, Potato healthy y arañas no cambia la
   partición de las demás clases (test que lo confirma).
2. **PlantDoc depurado:** de 1.098 imágenes del pool (12 clases) quedan **1.010 conservadas**: 60 excluidas por etiquetas contradictorias (29 grupos),
   28 por duplicado de la misma clase, 0 por revisión humana (**la planilla está vacía → todo PROVISIONAL**). Partición propuesta: **dev 707 / test 303**.
   **Sin alertas:** mínimo de dev 37 (≥ 30) y mínimo de test 16 (≥ 10).
3. **Las 6 imágenes de la carpeta test oficial con gemelo contradictorio quedan excluidas** (las decisiones del equipo excluyen todas las copias). El test
   propuesto (303) viene casi todo de la carpeta *train* oficial (279) y solo 24 de la test oficial; **no es comparable con el test oficial de PlantDoc**.
4. **Ni `Tomato leaf` ni `Bell_pepper leaf` están en ningún grupo de duplicados** (0 imágenes): la depuración automática no las toca; su única depuración es la revisión humana.
5. Los scripts de congelamiento y empaquetado están listos y probados sobre directorios temporales; **no se ejecutaron**.

## Decisiones del equipo registradas hoy (detalle en `docs/bitacora/decisiones.md`)

12 clases; partición propia de PlantDoc 70/30 con exclusión previa (a)–(d); mosaic zona 20 (con sensibilidad ID con y sin mosaic); val/test de PlantVillage
sin sesiones sin grupo y tope proporcional de 1.500 (**excepción: mosaic**, que no tiene grupos de hoja y se parte por rangos con zona 20, así que sí aporta a val y test: 36 y 35 imágenes); Colab opción A; invariante 1 de CLAUDE.md reescrito; el docente validó la partición propia y la exclusión previa.
`CLAUDE.md` actualizado (Contexto, invariante 1, Colab).

## 1. PlantVillage — 12 clases (`clase_eval`)

`clase_eval` = las 13 con `clase_comun` menos arañas. Tabla sobre el candidato (semilla 42, zona 20, tope 1.500); hojas entre paréntesis:

| Clase | Total | Train | Val | Test | Excl. tope | Excl. zona |
|---|---:|---|---|---|---:|---:|
| Pepper,_bell___Bacterial_spot | 997 | 695 (123) | 171 (26) | 131 (26) | 0 | 0 |
| Pepper,_bell___healthy | 1.478 | 1.017 (115) | 228 (25) | 233 (25) | 0 | 0 |
| Potato___Early_blight | 1.000 | 696 (174) | 152 (38) | 152 (38) | 0 | 0 |
| Potato___Late_blight | 1.000 | 696 (174) | 152 (38) | 152 (38) | 0 | 0 |
| Tomato___Bacterial_spot | 2.127 | 1.488 (372) | 320 (80) | 319 (80) | 0 | 0 |
| Tomato___Early_blight | 1.000 | 696 (174) | 152 (38) | 152 (38) | 0 | 0 |
| Tomato___Late_blight | 1.909 | 1.500 (162) | 136 (34) | 136 (34) | 137 | 0 |
| Tomato___Leaf_Mold | 952 | 664 (166) | 144 (36) | 144 (36) | 0 | 0 |
| Tomato___Septoria_leaf_spot | 1.771 | 1.467 (174) | 152 (38) | 152 (38) | 0 | 0 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5.357 | 1.500 (372) | 412 (103) | 412 (103) | 3.033 | 0 |
| Tomato___Tomato_mosaic_virus | 373 | 262 | 36 | 35 | 0 | 40 |
| Tomato___healthy | 1.591 | 1.287 (174) | 152 (38) | 152 (38) | 0 | 0 |
| **Total (12 clases)** | **19.555** | **11.968 (2.180)** | **2.207 (494)** | **2.170 (494)** | **3.170** | **40** |

**Coincidencia con lo esperado (≈ 11.968 / 2.207 / 2.170): exacta.** Cómo se llega: el candidato de 15 clases tenía 14.648 / 2.483 / 2.446; las tres clases que salen
aportan train 1.404 (Target Spot) + 104 (Potato healthy) + 1.172 (arañas) = 2.680, val 24 + 252 = 276 y test 24 + 252 = 276 (Target Spot no tiene val/test).
Las hojas de mosaic y Target Spot son 0 (sin grupo).
Sensibilidad mosaic (se reporta el macro-F1 ID con y sin mosaic): sin mosaic quedan 11.706 / 2.171 / 2.135.
Test nuevo (`tests/test_particion_pv.py`): construir el candidato solo con las 12 clases da **exactamente** las mismas filas que filtrar el candidato de 15
(cada clase usa su propia semilla derivada). Archivos: `data/interim/particion_candidato_plantvillage.csv` y `.meta.json` (no versionados).

## 2. PlantDoc — depuración y partición propia

### 2a. Contradictorios por combinación de clases (phash ≤ 6, pool de 12 clases)

| Combinación de clases del grupo | Grupos | Imágenes | De ellas en test oficial |
|---|---:|---:|---:|
| Potato early blight \| Potato late blight | 8 | 16 | 3 |
| Potato early blight \| Tomato Early blight | 4 | 9 | 1 |
| Potato early blight \| Tomato late blight | 3 | 6 | 0 |
| Tomato Septoria \| Tomato bacterial spot | 3 | 6 | 1 |
| Potato late blight \| Tomato late blight | 2 | 4 | 0 |
| Tomato Early blight \| Tomato Septoria | 2 | 4 | 0 |
| Potato early \| Potato late \| Tomato late blight (3 clases) | 1 | 3 | 1 |
| 6 combinaciones con 1 grupo de 2 imágenes (Bell_pepper leaf spot\|Septoria; Potato early\|Septoria; Potato late\|Tomato Early; Tomato Early\|Tomato late; mosaic\|yellow virus; yellow virus\|mold) | 6 | 12 | 0 |
| **Total** | **29** | **60** | **6** |

Por **pares** (≤ 6): papa early↔late 9; papa early↔tomate early 5; papa early↔tomate late 4; papa late↔tomate late 3; Septoria↔bacterial 3; tomate early↔Septoria 2; los 6 restantes, 1 cada uno.
**Papa early↔late** es la combinación más frecuente (9 pares; 8 grupos de esas dos clases y un grupo de 3 clases); **tomate↔papa** suma 14 pares en 5 combinaciones (papa early↔tomate early 5, early↔tomate late 4,
early↔Septoria 1; papa late↔tomate late 3, late↔tomate early 1). **Ningún par involucra `Tomato leaf` ni `Bell_pepper leaf`** (0 de 124 imágenes en cualquier grupo; matriz completa en el notebook 01b).
Ejemplo visual de cada combinación con ≥ 2 pares (6): `docs/figuras_sesion5/ejemplos_contradictorios.png` (en la figura hay 1 imagen de test oficial).

### 2b. Pares de rango 7–10 (solo para revisión del equipo; no se excluyó nada)

Son **3 pares**, ninguno con imagen de test oficial: 1 entre clases distintas (`early-blight-or-target-spot-alternaria-solani…`, tomate early ↔ papa early, d=8) y 2 de la misma clase en train
(papa early `80109626.jpg` ↔ `potato-early-blight-alternaria-alternata…`, d=10; papa late `B2650062-Potato_late_blight-SPL.jpg` ↔ `potato-late-blight-phytophthora…`, d=10).
Grilla `docs/figuras_sesion5/pares_7_10.png`; listado con columna `decision_equipo` vacía: `docs/bitacora/plantdoc_pares_7_10_revision.csv`.

### 2c. Manifiesto depurado (`data/interim/manifiesto_plantdoc_depurado.csv`, no versionado)

Columnas: `ruta`, `clase`, `split_oficial`, `nombre_original`, `grupo_hash`, `tam_grupo_hash`, `clases_en_grupo`, `copia_conservada`, `estado`, `particion_propuesta`, `revision_estado`.
Estados: conservada 1.010 · excluida_contradictoria 60 · excluida_duplicada 28 · excluida_revision 0 · fuera_de_alcance 2 (las 2 imágenes de arañas).

| Clase | Pool antes | Pool después | Excl. contradictoria | Excl. duplicada | Excl. revisión | **Dev** | **Test** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tomato leaf bacterial spot | 110 | 103 | 3 | 4 | 0 | 72 | 31 |
| Tomato Early blight leaf | 88 | 76 | 9 | 3 | 0 | 53 | 23 |
| Tomato leaf late blight | 111 | 99 | 7 | 5 | 0 | 69 | 30 |
| Tomato mold leaf | 91 | 90 | 1 | 0 | 0 | 63 | 27 |
| Tomato Septoria leaf spot | 151 | 138 | 7 | 6 | 0 | 97 | 41 |
| Tomato leaf yellow virus | 76 | 72 | 2 | 2 | 0 | 50 | 22 |
| Tomato leaf mosaic virus | 54 | 53 | 1 | 0 | 0 | 37 | 16 |
| Tomato leaf | 63 | 63 | 0 | 0 | 0 | 44 | 19 |
| Potato leaf early blight | 117 | 98 | 17 | 2 | 0 | 69 | 29 |
| Potato leaf late blight | 105 | 90 | 12 | 3 | 0 | 63 | 27 |
| Bell_pepper leaf spot | 71 | 67 | 1 | 3 | 0 | 47 | 20 |
| Bell_pepper leaf | 61 | 61 | 0 | 0 | 0 | 43 | 18 |
| **Total** | **1.098** | **1.010** | **60** | **28** | **0** | **707** | **303** |

**Alertas (test < 10 o dev < 30): ninguna.** Mínimos: dev 37 (mosaic), test 16 (mosaic). Papa early/late son las que más pierden (16 % y 14 % del pool).
Procedencia: de las 102 imágenes de la carpeta test oficial en las 12 clases, 96 quedan conservadas (24 al test propuesto, 72 al dev) y 6 se excluyen; de las 914 conservadas de la carpeta
train oficial, 279 van al test propuesto y 635 al dev.

**Estado de la revisión humana: PROVISIONAL** (124 de 124 filas sin completar; 0 filas con valores fuera del vocabulario). Cuando los revisores completen la planilla se vuelve a ejecutar
`01b_depuracion_plantdoc.ipynb`: se excluye una imagen solo si ambos escribieron `descartar`. **Efecto esperado de cerrar la revisión:** como la exclusión es previa al split y cada clase usa su
propia semilla, solo cambian las particiones de `Tomato leaf` y `Bell_pepper leaf`; las otras 10 clases quedan idénticas. La planilla vacía se detecta y marca PROVISIONAL en el `.meta.json`,
y `congelar_splits.py` se niega a congelar mientras siga así (salvo `--permitir-provisional`).

### 2d. Avisos sobre la regla aplicada tal cual (no los corregí)

- El par `2643371-version…` (papa early) ↔ `LateBlight03.jpg` (papa late), d=6, **a ojo no parece la misma foto** (sesión 4). La regla mecánica lo trata como contradictorio y excluye ambas (2 imágenes de papa).
- El par `LateBlight-Leaflet-Large-Spo…` ↔ `Tomato-late-blight-leaf-Marg…` (tomate late, d=6) era **dudoso**: se conserva una copia y se excluye la otra.
- Aplicar excepciones manuales contradice "regla escrita de antemano"; el efecto es de 2–3 imágenes. Lo dejo al equipo (opciones abajo).

### 2e. Tests (`tests/test_plantdoc_depurado.py`)

Sintéticos y reales (los reales se saltean si no están el dataset y la caché de hashes): ninguna imagen en dev y test a la vez; ningún `grupo_hash` con imágenes en ambos lados; la exclusión de contradictorias quita
**todas** las copias (incluidas las de test oficial) y ninguna conservada pertenece a un grupo contradictorio; la copia conservada de un grupo de una clase es la primera por ruta ordenada; reproducibilidad con la
semilla (incluso con las filas en otro orden: el id de grupo no depende del orden) y otra semilla cambia la partición; vocabulario de la revisión (solo `descartar`+`descartar` excluye; `Descartar` no se interpreta y se reporta;
planilla vacía → PROVISIONAL, sin exclusiones); revisión previa al split.

## 3. `scripts/congelar_splits.py` (NO ejecutado)

`python scripts/congelar_splits.py --config configs/congelar_splits.yaml [--dry-run]`. Copia `particion_candidato_plantvillage.csv` → `data/splits/plantvillage_particion.csv` y
`manifiesto_plantdoc_depurado.csv` → `data/splits/plantdoc_particion.csv`; escribe `SHA256SUMS.txt` y `README_VERSION.md` (fecha, commit del repo y si hay cambios sin commitear, commits de los datasets, estado de la
revisión humana, conteos por partición, semilla/zona/tope). Valida todo **antes** de escribir y se niega si: falta algún candidato o su `.meta.json`; la revisión está PROVISIONAL (`--permitir-provisional` lo salta);
o el destino ya existe (`--force` solo con decisión del equipo; las particiones no se regeneran tras la etapa 1). `--dry-run` calcula los sha256 sin escribir. Probado solo con directorios temporales.
Orden recomendado para el equipo: (1) completar la planilla, (2) re-ejecutar `01b`, (3) `--dry-run`, (4) ejecutar, (5) commitear `data/splits/`.

## 4. Empaquetado y Colab (opción A) — instrucciones

Tamaño calculado (solo `stat`, sin leer ni copiar): PlantVillage 12 clases color + segmented **39.110 archivos, 459 MiB**; PlantDoc 12 clases (cualquier estado) **1.098 archivos, 326 MiB**; ≈ 785 MiB en total.
(**MiB = 2^20 bytes**, no 10^6: verificado en la sesión 5d sumando los bytes de los manifiestos; en decimal son 481 MB, 341 MB y 822 MB. Lo mismo vale para el "~0,8 GB" siguiente, que son GiB.)
Todos los archivos listados existen en esta máquina.

**En la PC con disco** (con los datasets en `DATA_ROOT` y los notebooks 01 y 01b ya ejecutados, para que existan `manifiesto_plantvillage.csv` y `manifiesto_plantdoc_depurado.csv` en `interim/`):
```
pip install -e . --no-deps
python scripts/empaquetar_subconjunto.py --config configs/paquete_subconjunto.yaml --salida <carpeta_de_salida>
```
Genera `plantvillage_subconjunto.tar`, `plantdoc_subconjunto.tar` y sus `.tar.sha256`. Necesita ~0,8 GiB libres en la salida; calcula el sha256 de cada archivo (lee todo una vez más). Cada `.tar` lleva como
primer miembro `MANIFIESTO_<dataset>.csv` (`ruta, sha256, bytes`). Los nombres dentro del `.tar` son la `ruta` de los manifiestos, con los nombres **saneados** de PlantDoc: al descomprimir en Linux/Colab las rutas
valen tal cual (se evita el problema de los 101 nombres no representables en NTFS). Subir los 4 archivos a `<Drive>/foliares/paquetes/`.

**En Colab:** abrir `notebooks/colab_arranque.ipynb` (completar `DRIVE_DIR` y `REPO_URL`; no hay URL del repo registrada en el proyecto). Monta Drive, clona el repo, `pip install -e . --no-deps`, copia y descomprime los `.tar`
en `/content/data`, fija `DATA_ROOT=/content/data`, corre `scripts/verificar_paquete.py --tar …` (sha256 de cada `.tar` y de cada archivo contra su manifiesto; código de salida ≠ 0 si algo falla), comprueba que
`data/splits/README_VERSION.md` esté en el repo, y registra versión de torch, GPU, commit y semillas en `<Drive>/foliares/checkpoints/sesion_*.json` (ese directorio es el de checkpoints).
Los `.tar` no incluyen los CSV de particiones: esos viajan en el repo (`data/splits/`) una vez congelados. `scripts/verificar_paquete.py` también se puede correr en la PC (`--data-root`).

## 5. Cambios que necesita `docs/propuesta_entregable1.md` (no se editó)

| Dónde (propuesta) | Dice | Debería decir |
|---|---|---|
| §3 Variables / Preparación (línea "la intersección más rica está en tomate y maíz") | Mapeo con tomate y maíz | Tomate, papa y pimiento; **maíz fuera**; **12 clases** (13 con par menos arañas); Target Spot y Potato healthy fuera del entrenamiento |
| §3 tabla de datos, PlantDoc | 2.598 imágenes, 13 especies, 27 categorías; "prueba de 8 a 12 imágenes por clase"; rol "Evaluación OOD" | Copia fijada: 28 carpetas y 2.578 imágenes (el "27 = 17+10" es del paper; ver `mapeo_clases.md`). Rol: **dev** (selección de variantes, temperatura, umbral, preproceso) y **test** propios, partición 70/30 |
| §3 Preparación (casi-duplicados) | Solo contaminación PlantVillage↔PlantDoc | Agregar: duplicados *dentro* de PlantDoc, incluidos 29 grupos con etiquetas contradictorias (60 imágenes) pese a que el paper declara haber quitado duplicados entre clases; exclusión previa al split |
| §3 / §8 etapa 1 | Particiones con grupos de hoja no mencionados | Partición de PlantVillage **por hoja** (`clase:::hoja`), sin grupo → solo train, mosaic por rangos con zona 20; val/test sin sesiones sin grupo; tope de 1.500 |
| §4 (PlantDoc "solo evalúa"; "PlantDoc nunca toca el entrenamiento") | Absoluto sobre PlantDoc | Invariante nuevo: no entra al entrenamiento ni a la validación/early stopping de PlantVillage; PlantDoc-dev elige variantes, temperatura, umbral y geometría; PlantDoc-test se evalúa una vez por variante final con acceso registrado |
| §4 "umbral de abstención fijado con el circuito de evaluación"; §8 etapa 5 "ajuste de temperatura en validación" | Calibración sobre validación (de PlantVillage) | **Calibración y umbral sobre PlantDoc-dev**, no sobre la validación de PlantVillage |
| §6 y §7 (Grad-CAM, "Atajo por fondo: Grad-CAM y experimento de control"); §8 etapa 3 ("Grad-CAM sobre una muestra") y etapa 6 ("mapa de atención") | Grad-CAM | **CAM** (GAP→Linear, mismo forward) en las etapas 3 y 6 y en el análisis; Grad-CAM descartado del alcance; quitar la cita de Selvaraju |
| §3 Variables ("redimensionada a 224×224") | Preproceso fijo | La geometría (aplastar / resize+recorte / relleno) se elige con PlantDoc-dev; mismo preproceso para ambos dominios |
| §6 Métricas, §7 "Test OOD pequeño (8 a 12 por clase)" | n de 8–12 por clase | n real de PlantDoc-test: **303 imágenes, 16–41 por clase**; PlantVillage-test 2.170 en 12 clases. Siguen los IC bootstrap |
| §6 Brecha ID→OOD | Un solo número | Reportar el macro-F1 in-distribution **con y sin mosaic** (sensibilidad) |
| §7 Riesgos | — | Agregar: ruido de etiqueta en PlantDoc (grupos contradictorios), revisión humana de `Tomato leaf`/`Bell_pepper leaf`, test propio no comparable con el oficial |
| §8 etapa 1 | "descarga… congelamiento" | Evidencia: manifiestos, scripts de congelamiento y verificación por sha256, empaquetado para Colab |

## Decisiones que quedan para el equipo (con costo)

1. **Cerrar la revisión humana** (dos personas, por separado). Hasta entonces todo es PROVISIONAL y `congelar_splits.py` no congela. Costo de esperar: nada se puede congelar; costo de congelar provisional: si luego se descartan imágenes habría que
   re-congelar `Tomato leaf`/`Bell_pepper leaf` (contra el invariante de tests congelados).
2. **Los 2–3 casos que a ojo no son duplicados (2d).** (a) Aplicar la regla mecánica sin excepción (lo implementado; costo: 2 imágenes de papa excluidas de más, regla pura); (b) excepción manual documentada (costo: rompe "regla
   escrita de antemano", habría que decir en el informe que se tocó a mano, y volver a correr 01b y re-validar la lista).
3. **Los 3 pares de 7–10** (`plantdoc_pares_7_10_revision.csv`): decidir si se excluyen. Costo de excluir: ≤ 3 imágenes más; costo de no excluir: posible duplicado residual. **Corrección (sesión 5d):** ninguno toca el test *oficial*, pero **2 de los 3 pares cruzan el dev y el test propuestos** (una copia en cada lado: el par tomate early ↔ papa early y el par de papa early `80109626.jpg` ↔ `potato-early-blight-alternaria-alternata…`), así que no excluirlos dejaría una fuga dev→test; el 3.º (papa late) cae entero en dev. D3 los pasa a revisión por un revisor.
4. **El test propuesto no incluye casi nada del test oficial** (24 de 303). Esto ya cuenta con la validación del docente; queda como advertencia para el informe: los números no son comparables con trabajos que usen el test oficial.
5. **Quién escribe `data/splits/` y cuándo**: lo hace el equipo, tras (1), con el `--dry-run` previo.
6. **`REPO_URL` y `DRIVE_DIR`** del notebook de Colab: no están definidos en el proyecto.

## Cosas que NO se hicieron

Ningún modelo, ninguna métrica; nada en `data/splits/`; no se ejecutaron `congelar_splits.py`, `empaquetar_subconjunto.py` ni `verificar_paquete.py` sobre datos reales; `colab_arranque.ipynb` sin ejecutar; no se editó la propuesta;
no se instaló torch (disco local ~2,9 GB libres); la rama de torch de `fijar_semillas` sigue sin probarse. Las citas del paper (ar5iv) se leyeron con una herramienta de lectura de la página: verificar a mano antes de citarlas.
