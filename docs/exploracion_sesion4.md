# Entorno, manifiesto de PlantVillage y chequeos de PlantDoc — sesión 4

> Candidato y evidencia; **nada congelado**. Sin modelos, sin métricas de desempeño, nada escrito en
> `data/splits/`, ninguna carpeta de PlantDoc renombrada, ninguna depuración ejecutada.
> Código: `src/foliares/` (nuevo: `utils/`, `data/manifiesto_pv.py`, `data/particion_pv.py`,
> `data/plantdoc_chequeos.py`). Notebooks: `notebooks/01_preparacion_datos.ipynb` (bloque 2) y
> `notebooks/00c_chequeos_plantdoc.ipynb` (bloque 3). Configs: `configs/paths.yaml`,
> `configs/01_preparacion_datos.yaml`, `configs/00c_chequeos_plantdoc.yaml`. Semilla 42.
> Commits de los datasets verificados con `git log -1` (PlantVillage `7f7ecc7e…`, PlantDoc `5467f601…`): coinciden.

## Resumen para el equipo

1. **PlantDoc tiene mucho más ruido de etiqueta duplicada de lo que mostró la sesión 3.** Con duplicados
   buscados entre *todas* las clases (no solo dentro de cada clase): hay **33 pares entre clases distintas** (26 dentro de train, 7 train↔test;
   a ojo, 31 de los 32 pares ≤ 6 son la misma foto con etiquetas distintas). **6 de las 102 imágenes de test** (5,9 %) tienen
   una copia exacta en train con *otra* etiqueta (p. ej. papa early blight vs. late blight, tomate vs. papa).
   Más las 6 imágenes de test con copia de la misma clase que ya conocíamos: **12 de 102 imágenes de test
   (11,8 %) tienen un gemelo en train**. La regla "sacar de dev, nunca de test" no las toca.
2. **La regla fijada de antemano para el split oficial depurado se cumple numéricamente** (≥ 5 de test y ≥ 40 de
   dev por clase) **si se evalúan 12 clases**; con las 13 carpetas falla solo por arañas (2 en dev, 0 en test).
   **Pero** el propio enunciado de la regla exceptúa "defectos nuevos en el test", y el punto 1 es candidato a
   serlo (las 6 imágenes de test con gemelo de etiqueta contradictoria). Qué cuenta como defecto lo decide el equipo.
3. **Zona de descarte de mosaic: la regla fijada de antemano elige 20 números** (10 → 4,39 % de cortes con fuga;
   15 → 1,77 %; 20 → 0,84 %). Con matices importantes (ver 2b): el criterio se aplicó sobre la proporción
   *agregada*, y por tanda individual varía de 0 % a 11 %.
4. **Candidato de particiones de PlantVillage** guardado (no versionado), con tests de no-fuga verdes (23 tests en
   total, 16 de ellos de PlantVillage). Hallazgo de diseño a decidir: val y test **no contienen ninguna** de las
   sesiones sin grupo (laboratorio, `GHLB*`, `GH_HL`, `UF.GRC_YLCV_Lab`, Target Spot…), mientras que train sí.
5. **Un par color/segmented con convención irregular** (1 de 22.787). El resto, 22.786, sigue `<stem>_final_masked.jpg`.

## Bloque 1 — Entorno

Hecho: `requirements.txt` (torch, torchvision, gradio, imagehash, scikit-learn, pyyaml, pandas, numpy, pillow,
matplotlib, pytest + ipykernel/nbformat/nbclient para ejecutar notebooks), `pyproject.toml` con el paquete
`foliares` instalable (`pip install -e .`), `configs/paths.yaml`, `src/foliares/utils/paths.py` y `utils/seeds.py`.

- **Rutas.** `DATA_ROOT` (por defecto `data`, relativo a la raíz del repo) se sobreescribe con la variable de
  entorno `DATA_ROOT`. Los manifiestos guardan rutas **relativas a `DATA_ROOT`** en formato posix, así que el
  mismo CSV sirve en las dos máquinas y en Colab. Las particiones congeladas viven en `SPLITS_DIR` (dentro del
  repo, versionadas), no bajo `DATA_ROOT`. Ninguna ruta absoluta en código ni en notebooks nuevos (los notebooks
  de sesiones 2–3 conservan sus rutas relativas `data/raw/...`; no se tocaron).
- **Semillas.** `fijar_semillas(semilla, determinista=False)` (python, numpy, torch si está) y
  `semilla_derivada(base, *claves)` (SHA-256, estable entre máquinas y de orden de ejecución). Cada partición
  por clase y cada muestreo usa una semilla derivada de (semilla, clase[, sesión]), así que el candidato de una clase
  no cambia si se agregan o quitan otras (hay un test de eso).
- **Verificado:** `pip install -e . --no-deps` y 23 tests pasan en esta máquina. **No verificado:** la rama de
  torch de `fijar_semillas` (torch no está instalado acá; el disco local tiene 2,8 GB libres). Sin cambios en
  `requirements.txt` hay que instalarlo donde se entrene.
- **Versiones.** `requirements.txt` fija solo cotas mínimas. Opción para después: congelar con `pip freeze`
  al cerrar la etapa 1 (costo: mantener un segundo archivo).

### Colab ↔ repo: opciones (decisión pendiente del equipo)

Tamaños medidos del subconjunto candidato: PlantVillage color 353 MB, segmented 196 MB, PlantDoc (13 carpetas)
326 MB → **≈ 0,9 GB**; los clones completos pesan 2,57 GB + 2,10 GB. Los datos no se versionan.

| Opción | Cómo funciona | Costo |
|---|---|---|
| **A. Subconjunto empaquetado en Drive** | Un `.tar` por dataset (solo clases candidatas) en Drive; la sesión de Colab lo descomprime a `/content` y apunta `DATA_ROOT` ahí; el repo se clona por `git` | Hay que armar y mantener los `.tar` (re-generar si cambia el alcance). Rápido por sesión (~minutos). Ocupa ~0,9 GB de Drive. Hay que verificar hashes del `.tar` contra los commits fijados |
| **B. Clonar los datasets en cada sesión** | Colab hace `git clone` de ambos repos a los commits fijados (sparse para PlantVillage) | Sin Drive, reproducible por hash, pero 4,7 GB de descarga por sesión (clon completo) o ~1–2 GB con sparse; la sesión gratuita se corta y se repite |
| **C. Leer directo de Drive montado** | `DATA_ROOT` apunta a Drive | Lo más simple de configurar, pero leer ~23.000 archivos chicos desde Drive FUSE es muy lento y se nota en cada época; no recomendable para entrenar |

Código: en cualquiera de las tres, `git clone` del repo + `pip install -e .` al inicio; los resultados y
checkpoints se escriben a Drive (checkpointing obligatorio). El push desde Colab requiere un token de GitHub
(costo: manejar el secreto en Colab); alternativa: bajar los resultados y pushear desde una máquina local.
**Riesgo concreto a decidir con cualquiera de las opciones:** los 101 archivos de PlantDoc con nombres no válidos
en NTFS están guardados con nombre *local* saneado en estos manifiestos (`ruta`); en Linux/Colab el nombre real
contiene `?` o es otro. Los manifiestos de PlantDoc traen también `nombre_original`; habría que usar ese
campo (o aplicar el mismo mapeo) en la máquina que no sea Windows. No lo resolví: depende de la opción elegida.

## Bloque 2 — Manifiesto y candidato de particiones de PlantVillage

### 2a. Manifiesto

`data/interim/manifiesto_plantvillage.csv` (no versionado): **22.787 imágenes, 15 clases** (tomate 10, papa 3,
pimiento 2). Columnas: `ruta_color`, `ruta_segmented`, `segmented_existe`, `segmented_irregular`, `clase`, `cultivo`,
`clase_comun`, `clase_plantdoc_candidata`, `sesion`, `num`, `leaf_id` (`clase:::hoja` o vacío), `tiene_grupo`, `uuid`, `archivo`.
`clase_comun` = tiene par en PlantDoc según `docs/mapeo_clases.md` (**13 de 15**; sin par: Target_Spot y Potato healthy);
no implica decisión (p. ej. arañas figura como común aunque PlantDoc tenga 0 imágenes de test).

**Par color/segmented.** Convención verificada: `<stem de color>_final_masked.jpg` (la extensión de color puede ser
`.JPG`/`.jpg`; la de segmented es siempre `.jpg`). Resultado sobre las 15 clases: **0 imágenes de color sin par**,
**0 segmented sin color**, y **1 par irregular**: `Tomato___Spider_mites…/0db2da38-…___Com.G_SpM_FL 1378.JPG` cuyo
segmented perdió el prefijo `uuid___` (`Com.G_SpM_FL 1378_final_masked.jpg`); queda enlazado y marcado con
`segmented_irregular=True`. Los conteos por clase de color y segmented coinciden en las 15.

Con grupo de hoja: **16.036**; sin grupo: **6.751** (coincide con sesión 3).

### 2b. Mosaic virus: validación de la zona de descarte *antes* de aplicarla

**Método.** Sobre las tandas (clase, sesión) que sí tienen grupo y numeración (16 tandas), se sortean 10.000 pares de
cortes al azar por tanda (cada tramo ≥ 5 % de la tanda), se aplica una zona de descarte de N números después de cada
corte y se cuenta en qué proporción de cortes alguna hoja real queda con imágenes en ≥ 2 tramos (train/val/test).
160.000 cortes por zona. Criterio fijado de antemano: proporción ≤ 1 % → se aplica; si no, 15 y 20; si tampoco,
mosaic solo a train.

| Zona (números) | Cortes con fuga | Proporción (IC 95 % Wilson) | ¿≤ 1 %? |
|---:|---:|---|---|
| 10 | 7.026 / 160.000 | **4,39 %** (4,29–4,49) | no |
| 15 | 2.834 / 160.000 | **1,77 %** (1,71–1,84) | no |
| 20 | 1.342 / 160.000 | **0,84 %** (0,80–0,88) | **sí** |

**Por regla fijada de antemano, mosaic se corta con zona 20.** Resultado en el candidato: train 262 (2047–2309),
descarte 40 (2310–2329 y 2366–2385), val 36 (2330–2365), test 35 (2386–2420). Para comparar, con zona 10 serían 262/46/45
(20 descartadas) y con 15, 262/41/40 (30).

**Matices que el equipo debería conocer (no cambian la regla, sí cómo leerla):**
- La proporción es **agregada** sobre las 16 tandas (todas pesan igual). Es mi operacionalización de "la proporción de
  cortes"; el enunciado no distingue agregado de por tanda. Por tanda, a zona 20: 14 de 16 tandas dan < 1 % (la mayoría 0 %);
  `Tomato___Septoria_leaf_spot|Matt.S_CG` 1,23 %; y **`Tomato___Bacterial_spot|UF.GRC_BS_Lab Leaf` 11,25 %** (con zona
  10: 42,39 %). Si el criterio fuera "toda tanda ≤ 1 %", **ninguna de las tres zonas pasaría** (→ mosaic solo a train).
  Tabla por tanda en el notebook §2b y en `data/interim/validacion_zona_por_unidad.csv`.
- Las hojas de laboratorio abarcan rangos amplios de numeración; no sabemos si `PSU_CG` (mosaic) se parece a las tandas
  de campo (0 %) o a las de laboratorio (11–42 %). Mosaic no tiene grupos, así que **esto es un proxy, no una medición
  sobre mosaic**.
- Los rangos son bloques de captura: val y test de mosaic son de un tramo temporal distinto al de train (decisión ya
  tomada). Orden usado: train = números más bajos, val = intermedios, test = más altos. Con 36 y 35 imágenes, las
  métricas por clase en val/test para mosaic tendrán intervalos anchos.

### 2c. Candidato de particiones

`data/interim/particion_candidato_plantvillage.csv` (**CANDIDATO**, no versionado; `data/splits/` solo tiene `.gitkeep`).
Reglas: por hoja (`clase:::hoja`), por clase, 70/15/15 con semilla 42; sin grupo → solo train (mosaic: rangos, zona 20);
tope de 1.500 imágenes de train por clase con muestreo estratificado por sesión. Columnas: `particion_base` (antes del
tope), `particion` (final: train/val/test/excluida), `motivo_exclusion` (`tope_train` / `zona_descarte`), `regla`, `semilla`.

Imágenes y hojas por clase y partición (hojas = `clase:::hoja` distintas; mosaic y Target_Spot no tienen hojas):

| Clase | ¿común? | Total | Train img (hojas) | Val img (hojas) | Test img (hojas) | Excl. tope | Excl. zona |
|---|:-:|---:|---|---|---|---:|---:|
| Pepper,_bell___Bacterial_spot | sí | 997 | 695 (123) | 171 (26) | 131 (26) | 0 | 0 |
| Pepper,_bell___healthy | sí | 1.478 | 1.017 (115) | 228 (25) | 233 (25) | 0 | 0 |
| Potato___Early_blight | sí | 1.000 | 696 (174) | 152 (38) | 152 (38) | 0 | 0 |
| Potato___Late_blight | sí | 1.000 | 696 (174) | 152 (38) | 152 (38) | 0 | 0 |
| Potato___healthy | no | 152 | 104 (26) | 24 (6) | 24 (6) | 0 | 0 |
| Tomato___Bacterial_spot | sí | 2.127 | 1.488 (372) | 320 (80) | 319 (80) | 0 | 0 |
| Tomato___Early_blight | sí | 1.000 | 696 (174) | 152 (38) | 152 (38) | 0 | 0 |
| Tomato___Late_blight | sí | 1.909 | 1.500 (162) | 136 (34) | 136 (34) | 137 | 0 |
| Tomato___Leaf_Mold | sí | 952 | 664 (166) | 144 (36) | 144 (36) | 0 | 0 |
| Tomato___Septoria_leaf_spot | sí | 1.771 | 1.467 (174) | 152 (38) | 152 (38) | 0 | 0 |
| Tomato___Spider_mites… | sí | 1.676 | 1.172 (293) | 252 (63) | 252 (63) | 0 | 0 |
| Tomato___Target_Spot | no | 1.404 | 1.404 (0) | 0 | 0 | 0 | 0 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | sí | 5.357 | 1.500 (372) | 412 (103) | 412 (103) | 3.033 | 0 |
| Tomato___Tomato_mosaic_virus | sí | 373 | 262 (0) | 36 (0) | 35 (0) | 0 | 40 |
| Tomato___healthy | sí | 1.591 | 1.287 (174) | 152 (38) | 152 (38) | 0 | 0 |
| **Total** | | **22.787** | **14.648 (2.499)** | **2.483 (563)** | **2.446 (563)** | **3.170** | **40** |

(El tope solo actuó en Late_blight y YLCV; Septoria llega a 1.467, bajo el tope. Excluidas totales: 3.210 = 3.170 + 40.)

**Hallazgos de diseño (no son decisiones):**
- **val y test no contienen ninguna sesión sin grupo.** En train quedan, por ejemplo, Late blight `GHLB*`/`GH_HL` (906 de 1.500
  imágenes de train; incluye el fondo negro de `GHLB_PS`), Septoria `JR_Sept.L.S` + `Keller.St_CG` (771), YLCV `UF.GRC_YLCV_Lab`
  (867 de 1.500), healthy `GH_HL Leaf` (590) y todo Target Spot (1.404). Consecuencia: la val de esas clases mide
  generalización a sesiones de campo/invernadero con grupo, no a las de laboratorio que sí se ven en train. Es consecuencia
  directa de D1-opción A; se deja a la vista por si el equipo quiere otra cosa.
- **Tope y YLCV:** el tope descarta 3.033 de 4.533 imágenes de train de esa clase; el muestreo estratificado deja la sesión de
  laboratorio en 58 % de lo que queda (867 de 1.500) porque era 58 % del origen. Con otro reparto cambia la composición de train.
- Mosaic y Target Spot no tienen hojas; sus conteos de hoja son 0.

### 2d. Tests (pytest)

`tests/test_particion_pv.py` (16 tests) y `tests/test_plantdoc_chequeos.py` (7): **23 pasan**. Cubren: ninguna hoja en más de una
partición (sobre `particion_base`, antes del tope); la hoja incluye la clase (un mismo número de hoja en dos clases son hojas distintas);
val/test solo con grupo (más mosaic, que es la única excepción y lleva `regla=rango_zona20`); sin grupo → train; zona de descarte
respetada en mosaic; tope ≤ 1.500 y solo sobre train; ninguna imagen de PlantDoc en el manifiesto (por ruta y por **igualdad de bytes**
contra todo PlantDoc train/test); reproducibilidad con la semilla (incluso con las filas en otro orden) y que otra semilla cambie la
partición; la partición de una clase no depende de las demás; la simulación detecta fuga cuando la hoja abarca más que la zona.
Los tests con datos reales se saltean si no están los datasets; los sintéticos corren siempre.

## Bloque 3 — Chequeos de PlantDoc (sin ningún modelo)

13 carpetas candidatas, train + test juntos: **1.100 imágenes** (998 train, 102 test). phash de 64 bits, 8 variantes
(4 rotaciones × espejado), distancia mínima identidad-vs-variantes en ambos sentidos. Primera corrida de hashing: 59 s,
~180 MB de RAM (cacheado en `data/interim/`).

### 3a. Duplicados por categoría

| Categoría | ≤ 6 (estricto) | 7–10 (exploratorio) | ≤ 10 |
|---|---:|---:|---:|
| misma clase train↔test | **6** | 0 | 6 |
| misma clase train↔train | **23** | 2 | 25 |
| misma clase test↔test | **0** | 0 | 0 |
| **clases distintas** (cualquier combinación) | **32** | 1 | 33 |

Coinciden con la sesión 3 los 6 train↔test y los 25 train↔train (a ≤ 10; a ≤ 6 hay 23 acá contra 22 antes por simetrizar la
distancia en ambos sentidos). **Novedad: los 33 pares entre clases distintas** (26 train↔train, 7 train↔test). Listado completo con
ambas rutas (relativas a `DATA_ROOT`), nombre original y distancia: `docs/bitacora/plantdoc_duplicados.csv` (64 filas).
Grillas (revisión uno por uno): `docs/figuras_sesion4/pares_train_test.png` (6 pares), `pares_clases_distintas_1/2.png` (33 pares) y
`pares_train_train_1/2.png` (25 pares); no hay pares test↔test que mirar.

**Lectura visual** (mía, a confirmar por el equipo; columna `lectura_visual` del CSV):
- train↔test misma clase (6): todos la misma foto (cuatro a distancia 0, dos a distancia 2 con recorte/marca de agua distintos).
- clases distintas ≤ 6 (32): **31 son la misma foto** (en 30 de 32 incluso a distancia 0 o 2); **1 no lo parece** (potato early
  `2643371-version…` vs potato late `LateBlight03.jpg`, d=6: falso positivo probable). El de rango exploratorio (d=8) no se miró uno por uno.
- train↔train ≤ 6 (23): 22 son la misma foto; 1 dudoso (`LateBlight-Leaflet-Large-Spo…` vs `Tomato-late-blight-leaf-Marg…`, d=6).
  Los 2 de distancia 10 son dudosos.
- Hipótesis, **no verificada**: muchos pares con *mismo nombre de archivo* en dos carpetas (6 de los 7 train↔test entre clases distintas;
  en el caso de 3b se confirmó además igualdad de bytes) sugieren que PlantDoc guardó la misma imagen una vez por cada consulta de
  búsqueda que la devolvió, cada una con la etiqueta de la consulta.

**Imágenes de test con gemelo en train (distancia ≤ 6):**

| Clase de test | Gemelo con etiqueta contradictoria | Gemelo de la misma clase | Test total |
|---|---:|---:|---:|
| Potato leaf early blight | 2 | 2 | 8 |
| Potato leaf late blight | 3 | 0 | 8 |
| Tomato Septoria leaf spot | 0 | 1 | 11 |
| Tomato leaf bacterial spot | 1 | 1 | 9 |
| Tomato leaf late blight | 0 | 1 | 10 |
| Tomato leaf yellow virus | 0 | 1 | 6 |
| **Total** | **6** | **6** | (de 102) → **12** |

### 3b. Par con etiquetas contradictorias (`irish-blight-symptoms…`)

`docs/figuras_sesion4/par_irish_blight.png`. Es **el mismo archivo**: bytes idénticos (md5 `bd890a743a88bd8a8a3046bc74a3eda3`,
56.066 bytes, 640×437), distancia phash 0, **mismo nombre completo** `irish-blight-symptoms-on-potato-leaves-atmf8b.jpg` en
`train/Potato leaf early blight/` y en `test/Potato leaf late blight/`. No hay otras copias del nombre en PlantDoc. **Pistas de la
ruta (no se decide la etiqueta):** el nombre dice *irish blight*, nombre común del tizón tardío de la papa (*Phytophthora infestans*),
y no menciona *early* ni *Alternaria*; las dos carpetas son hermanas en las dos particiones. La foto es una hoja de papa con una lesión
parda, con marca de agua de un banco de imágenes. Es una pista, no una etiqueta confirmada.

### 3c. Acción propuesta (no se ejecuta) y números

Regla propuesta: **sacar de dev, nunca de test**. Por grupo de duplicados (componente de pares ≤ 6): si hay una copia en test se
sacan todas las de train; si no, se conserva una y se sacan las demás; si el grupo mezcla clases no se elige etiqueta y se sacan todas las
copias de train (la etiqueta la decide el equipo). Los pares de 7–10 llevan "sin acción propuesta, revisar a ojo". Escenarios, solo como números
(dev = carpeta `train` oficial; test = carpeta `test` oficial; la partición propia dev/test **no está decidida**):

| Clase | dev antes | test | dev A | dev B | dev C |
|---|---:|---:|---:|---:|---:|
| Bell_pepper leaf | 53 | 8 | 53 | 53 | 53 |
| Bell_pepper leaf spot | 62 | 9 | 59 | 58 | 58 |
| Potato leaf early blight | 109 | 8 | 107 | 92 | 90 |
| Potato leaf late blight | 97 | 8 | 94 | 85 | 84 |
| Tomato Early blight leaf | 79 | 9 | 75 | 67 | 66 |
| Tomato Septoria leaf spot | 140 | 11 | 134 | 127 | 127 |
| Tomato leaf | 55 | 8 | 55 | 55 | 55 |
| Tomato leaf bacterial spot | 101 | 9 | 97 | 95 | 95 |
| Tomato leaf late blight | 101 | 10 | 96 | 89 | 89 |
| Tomato leaf mosaic virus | 44 | 10 | 44 | 43 | 43 |
| Tomato leaf yellow virus | 70 | 6 | 68 | 66 | 66 |
| Tomato mold leaf | 85 | 6 | 85 | 84 | 84 |
| Tomato two spotted spider mites leaf | 2 | 0 | 2 | 2 | 2 |
| **Total** | **998** | **102** | **969** | **916** | **912** |

A = solo misma clase ≤ 6 (29 imágenes de dev); B = A + clases distintas ≤ 6 (82); C = B + rango exploratorio ≤ 10 (86; cota superior).
**El test no cambia en ningún escenario.** `Tomato leaf` y `Bell_pepper leaf` (las dudosas) no pierden casi nada (0 y 0 en A/B).
Nota: B incluye el par que a ojo no es duplicado (d=6), con efecto ≤ 1–2 imágenes.

**Regla fijada de antemano** (split oficial depurado si cada clase conserva ≥ 5 de test y dev ≥ 40) — **evaluada, no aplicada**:

| Escenario | 13 carpetas | 12 (sin arañas) | dev mínimo (12) | test mínimo (12) |
|---|---|---|---:|---:|
| A | no cumple (arañas: dev 2, test 0) | **cumple** | 44 | 6 |
| B | no cumple (ídem) | **cumple** | 43 | 6 |
| C | no cumple (ídem) | **cumple** | 43 | 6 |

Márgenes finos: mosaic queda en 43–44 de dev (umbral 40) y yellow virus / mold leaf en test = 6 (umbral 5). La salvedad de la regla ("salvo que
el chequeo encuentre defectos nuevos en el test") **queda abierta para el equipo**: las 6 imágenes de test con gemelo de etiqueta contradictoria
son el hallazgo relevante, más las 6 con gemelo de la misma clase (ya conocidas).

Columnas para ambas opciones (split oficial depurado / partición propia agrupada por hash) en `data/interim/manifiesto_plantdoc.csv`:
`ruta`, `clase`, `split_oficial`, `nombre_original`, `grupo_hash` (componente de pares ≤ 6 entre *todas* las clases; vacío si no tiene
duplicados), `tam_grupo_hash`, `clases_en_grupo`, `etiquetas_contradictorias`, `propuesta_sacar_de_dev_A`, `propuesta_sacar_de_dev_B`,
`revisar_etiqueta`. Hay 57 grupos de duplicados (116 imágenes): 28 de una sola clase (56 imágenes) y **29 con etiquetas contradictorias (60 imágenes)**.
Una partición propia agrupada por `grupo_hash` mantendría juntos los gemelos, pero no resuelve cuál de las etiquetas contradictorias es la correcta.

### 3d. Revisión de etiquetas de `Tomato leaf` y `Bell_pepper leaf`

124 imágenes: `Tomato leaf` 63 (55 dev + 8 test) y `Bell_pepper leaf` 61 (53 dev + 8 test). Material:
- Planilla `docs/bitacora/revision_etiquetas_planilla.csv` con `id, split, ruta, revisor_1, revisor_2, motivo`. Ids `TL-001…TL-063` y `BP-001…BP-061`
  asignados en orden aleatorio (semilla 42): no revelan split ni orden de archivo. Cada revisor completa **su** columna en **su** copia y después se
  juntan (el CSV tiene ambas columnas, no hace falta que una persona vea la otra mientras revisa).
- Hojas de contacto con el id y miniaturas (20 por página, 4 páginas por clase) en `data/interim/revision_etiquetas/` (**no versionadas**; se regeneran
  con el notebook). Cada página lleva el nombre de la carpeta de PlantDoc, no el split.
- Criterios (los aplican personas): (1) no es la especie; (2) no es una hoja (fruto, diagrama, lámina con texto); (3) el nombre del archivo nombra
  una enfermedad **y** se ven lesiones. Nunca por borrosa, recortada o difícil. Se descarta solo si ambos revisores coinciden. Sugerencia de uso de
  `motivo`: el número del criterio (1, 2 o 3). No definí un vocabulario para `revisor_*` (p. ej. `descartar`/`mantener`): lo fija el equipo.
- Como el nombre de archivo importa para el criterio 3, la planilla trae la `ruta` (con el nombre local; en 118 de 124 imágenes coincide con el nombre original, las otras 6 fueron saneadas por NTFS).

Dato incidental: las hojas de contacto se generaron con código; yo solo miré una página (3 imágenes) para verificar el formato.

## Opciones con su costo (decisiones que quedan para el equipo)

1. **Qué son las "12 clases de entrenamiento y evaluación".** El enunciado dice 12 y el mapeo tiene 13 con par. Lo que encaja en los números es
   **13 menos arañas** (0 imágenes de test, 2 de train); entonces Target Spot y Potato healthy (sin par) tampoco serían de evaluación. Lo supuse solo para
   evaluar la regla del split; **no está confirmado** y el manifiesto no lo codifica (usa la columna `clase_comun` = 13). Costo de equivocarse: la regla pasa de
   "cumple" a "no cumple" según se incluya o no arañas.
2. **Defectos nuevos en el test (cláusula de la regla del split).** (a) Considerar las 6 imágenes de test con gemelo contradictorio como defecto → se pasa a
   partición propia agrupada por hash (costo: el test deja de ser el oficial, no comparable con trabajos previos; hay que decidir cómo repartir 1.100 imágenes
   y los grupos contradictorios). (b) No considerarlas defecto y usar el split oficial depurado, reportando que 12/102 imágenes de test tienen gemelo en train
   (costo: el número OOD queda optimista en esas imágenes; con 6 de ellas posiblemente mal etiquetadas en test). (c) Usar el oficial depurado **y** marcar esas 6 como
   excluidas de la evaluación con justificación documentada (costo: viola "nunca sacar de test" de la regla propuesta, hay que decidirlo explícitamente).
3. **Criterio de la zona de descarte de mosaic.** Aplicar la regla tal cual (zona 20, agregado) o endurecerla: (a) agregado ≤ 1 % → zona 20 (candidato actual; riesgo
   de fuga residual si mosaic se parece a las tandas de laboratorio); (b) por tanda, peor caso ≤ 1 % → ninguna zona pasa → mosaic solo a train (costo: 373 imágenes de
   una clase sin val/test, mosaic no se puede evaluar en ID); (c) zona mayor (30/40/60), a validar con la misma simulación (costo: más descarte; con zona 40 mosaic pierde
   ~80 de 373 imágenes).
4. **Val/test sin sesiones sin grupo.** Mantener (D1-A tal cual; val/test más limpios, menos diversos que train) o recuperar grupos por reglas nuevas para algunas sesiones
   (p. ej. Target Spot por número, `GHLB`/`GH_HL` por el número de hoja del nombre; costo: reglas a validar a mano y sin regla para YLCV Lab ni mosaic).
5. **Tope por sesión en YLCV/Late blight.** Con 1.500 y estratificación proporcional, la sesión de laboratorio queda con 58 % de train en YLCV. Alternativa: cuota fija por
   sesión (costo: cambia la composición; hay que justificarla).
6. **Acción sobre los duplicados dentro de dev (82 imágenes en el escenario B).** Sacarlas (costo: ~8 % de dev; `Potato leaf early/late blight` pierden 12–16 %) o agrupar por hash
   y dejarlas dentro de un mismo lado. La acción sobre las etiquetas contradictorias depende de la revisión humana, no de la regla.
7. **Colab ↔ repo** (A/B/C del bloque 1) y, ligado a eso, cómo se manejan los 101 nombres de PlantDoc no representables en NTFS.

## Cosas que NO se hicieron

- Ningún modelo, ni siquiera para extraer características; ninguna métrica. `torch` no se instaló (disco).
- Nada en `data/splits/`; no se renombró ninguna carpeta de PlantDoc; no se movió ni se borró ninguna imagen; ninguna depuración ejecutada.
- No se revisó a ojo cada par de rango exploratorio ni las clases "X leaf" completas (eso es lo que hace la revisión humana de 3d).
- La partición propia dev/test de PlantDoc no se diseñó; solo se dejaron las columnas para ambas opciones.
