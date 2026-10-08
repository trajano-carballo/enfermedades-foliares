# Chequeos previos a las decisiones — sesión 3

> Exploratorio, nada congelado. Sin modelos, sin métricas de desempeño, nada escrito en
> `data/splits/`. Ninguna decisión se toma acá: es evidencia para el equipo y el docente.
> Detalle ejecutable (código, tablas completas, grillas): `notebooks/00b_chequeos_previos.ipynb`;
> config: `configs/00b_chequeos_previos.yaml`; figuras: `docs/figuras_sesion3/`.
> Datasets en los commits fijados (PlantVillage `7f7ecc7e…`, PlantDoc `5467f601…`), hashes
> verificados con `git log -1`. Alcance: tomate, papa, pimiento; **manzana solo como candidata de
> inventario** (no entra al alcance).

## Resumen para la reunión

1. **PlantVillage: agrupar por hoja es posible, pero cubre solo el 63 % del tomate y deja afuera sesiones
   enteras.** Los 6.751 sin grupo de tomate+pimiento son, salvo 5 imágenes sueltas, sesiones de
   captura completas (YLCV de laboratorio, Target Spot, mosaic virus, parte de Late blight y
   Septoria, `GH_HL Leaf`). Agrupar por numeración contigua no sirve como sustituto.
2. **PlantDoc: el split oficial train/test tiene fuga.** 6 pares train↔test de la misma clase son
   la misma imagen (o casi), verificados a ojo. Además hay 25 pares dentro de train. Lo que se elija
   sobre dev se filtra al test si se usa el split oficial tal cual.
3. **PlantVillage↔PlantDoc: sin contaminación real detectada.** Con las 8 variantes, los pares más
   cercanos son falsos positivos (hojas aisladas sobre fondo liso). `TomatoLateBlightTop.jpg`
   **no** es la misma foto que su par de PlantVillage.
4. **Las clases "X leaf" de PlantDoc no son homogéneamente "sanas"** (ver §3): incluyen plantas
   enteras, frutos, stock photos y algún caso de otra especie o con síntomas. La propuesta de
   mapearlas a `*___healthy` sigue abierta.

## 1. Grupos de hoja (PlantVillage, carpeta color)

### 1a. Reconciliación con el loader oficial

Mapeo imagen→hoja por match exacto del nombre (sin el prefijo `uuid___`) contra
`leaf_grouping/filtered_leafmaps/<clase>.csv`. La hoja se identifica como (clase, `Leaf #`), porque
`Leaf #` solo es único dentro de la clase. **Tomate: 18.160 imágenes, 11.411 con grupo por CSV y
11.411 por la lógica del loader (`plant_village.py`): coincide.** Coincide también clase por clase,
con una excepción:

| Diferencia | Detalle |
|---|---|
| `Apple___Black_rot` (manzana, solo inventario) | CSV filtrado: no existe. El loader sí mapea las 621 imágenes (el leaf-map oficial las trae bajo el prefijo `JR_FrgE.S`, que en `leaf_maps/` se llama `Apple_Frogeye Spot.csv`). Es un problema de nombres de clase en el repo de origen. |

Otras observaciones de la reconciliación:

- **`Tomato___Target_Spot`: 0 de 1.404 con grupo, por un desajuste de nombre.** El CSV usa el
  prefijo `Com.G_FL`; las imágenes se llaman `Com.G_TgS_FL`. Con match exacto (y con el loader) no
  matchea nada; por número, los 1.404 coinciden. **No se aplicó** ese match no exacto; si el equipo
  lo quiere, es una regla nueva a decidir. (Target Spot no tiene equivalente en PlantDoc.)
- `Tomato___Tomato_mosaic_virus`: no tiene CSV en `filtered_leafmaps` (373 imágenes sin grupo).
- Los CSV filtrados están topeados cerca de 1.000 filas por clase; por eso Late blight, Septoria,
  YLCV y healthy tienen sesiones sin grupo.
- Filas de CSV sin imagen en el directorio: Bacterial_spot 1, Early_blight 1, YLCV 2, healthy 1.
  En `Apple___healthy` hay 6 imágenes con el mismo nombre base y uuid distinto (1.645 imágenes, 1.639 filas).
- Nombres con "(conflicted copy …)": **0 en los directorios de imágenes**, 1 en un CSV
  (`RS_Erly.B 6341 (Kelsee Baranowski's conflicted copy 2015-10-06).JPG`, Early blight) que no
  matchea ninguna imagen; por eso `RS_Erly.B 6341.JPG` queda sin grupo.
- Nombres con decimales: 394 (`GHLB Leaf 2.1 Day 16.JPG`, 124 en Late blight y 270 en healthy), todos
  en sesiones sin grupo. El nombre trae el número de hoja y el día (serie temporal de laboratorio):
  es un posible identificador de hoja alternativo, **no verificado**.

### 1b. Tabla por clase

| Clase | Imágenes | Con grupo | Sin grupo | Hojas | Img/hoja (media; mín–máx) |
|---|---:|---:|---:|---:|---|
| Tomato___Bacterial_spot | 2.127 | 2.127 | 0 | 532 | 4,00; 3–4 |
| Tomato___Early_blight | 1.000 | 999 | 1 | 250 | 4,00; 3–4 |
| Tomato___Late_blight | 1.909 | 920 | 989 | 230 | 4,00; 4–4 |
| Tomato___Leaf_Mold | 952 | 952 | 0 | 238 | 4,00; 4–4 |
| Tomato___Septoria_leaf_spot | 1.771 | 1.000 | 771 | 250 | 4,00; 4–4 |
| Tomato___Spider_mites | 1.676 | 1.676 | 0 | 419 | 4,00; 4–4 |
| Tomato___Target_Spot | 1.404 | 0 | 1.404 | — | — |
| Tomato___YLCV | 5.357 | 2.738 | 2.619 | 685 | 4,00; 3–4 |
| Tomato___Tomato_mosaic_virus | 373 | 0 | 373 | — | — |
| Tomato___healthy | 1.591 | 999 | 592 | 250 | 4,00; 3–4 |
| Potato___Early_blight | 1.000 | 1.000 | 0 | 250 | 4,00; 4–4 |
| Potato___Late_blight | 1.000 | 1.000 | 0 | 250 | 4,00; 4–4 |
| Potato___healthy | 152 | 152 | 0 | 38 | 4,00; 4–4 |
| Pepper,_bell___Bacterial_spot | 997 | 997 | 0 | 175 | 5,70; 4–17 |
| Pepper,_bell___healthy | 1.478 | 1.476 | 2 | 165 | 8,95; 1–17 |
| *Apple___Apple_scab (candidata)* | 630 | 630 | 0 | 108 | 5,83; 1–15 |
| *Apple___Black_rot (candidata)* | 621 | 0 (CSV) / 621 (loader) | — | — | — |
| *Apple___Cedar_apple_rust (candidata)* | 275 | 275 | 0 | 69 | 3,99; 3–4 |
| *Apple___healthy (candidata)* | 1.645 | 1.645 | 0 | 257 | 6,40; 1–11 |

Por cultivo: tomate 18.160 (11.411 con grupo, 6.749 sin), papa 2.152 (todas con grupo), pimiento
2.475 (2.473 con grupo, 2 sin).

### 1c. Clase × sesión de captura × con/sin grupo

Sesión = prefijo del nombre antes del número. **Verificado: las imágenes sin grupo son sesiones
enteras**, con 5 excepciones sueltas.

| Clase | Sesión | Con grupo | Sin grupo |
|---|---|---:|---:|
| Tomato___Bacterial_spot | GCREC_Bact.Sp | 1.756 | 0 |
| | UF.GRC_BS_Lab Leaf | 371 | 0 |
| Tomato___Early_blight | RS_Erly.B | 999 | **1** (`6341`, ver 1a) |
| Tomato___Late_blight | RS_Late.B | 920 | 0 |
| | GHLB, GHLB Leaf, GHLB_PS Leaf/leaf, GH_HL Leaf | 0 | **989** |
| Tomato___Leaf_Mold | Crnl_L.Mold | 952 | 0 |
| Tomato___Septoria_leaf_spot | Matt.S_CG | 1.000 | 0 |
| | JR_Sept.L.S | 0 | **497** |
| | Keller.St_CG | 0 | **274** |
| Tomato___Spider_mites | Com.G_SpM_FL | 1.676 | 0 |
| Tomato___Target_Spot | Com.G_TgS_FL | 0 | **1.404** (ver 1a) |
| Tomato___YLCV | YLCV_GCREC | 1.470 | 0 |
| | YLCV_NREC | 1.268 | 0 |
| | **UF.GRC_YLCV_Lab** | 0 | **2.619** |
| Tomato___Tomato_mosaic_virus | PSU_CG | 0 | **373** |
| Tomato___healthy | RS_HL | 999 | 0 |
| | GH_HL Leaf | 0 | **590** |
| | (2 archivos sueltos: una foto de fruto, otra `CG`) | 0 | 2 |
| Potato___* (3 clases) | RS_Early.B / RS_LB / RS_HL | 2.152 | 0 |
| Pepper,_bell___Bacterial_spot | JR_B.Spot 589, NREC_B.Spot 408 | 997 | 0 |
| Pepper,_bell___healthy | JR_HL | 1.476 | 0 |
| | (2 archivos sueltos: `Screen Shot`, `bell-pepper-plant-`) | 0 | 2 |

Confirmado lo que sugería el chequeo previo: **YLCV `UF.GRC_YLCV_Lab` = 2.619 imágenes, todas sin
grupo**. De las 6.751 imágenes sin grupo (tomate + pimiento), 6.746 son sesiones completas; las otras
5 son sueltas (la `6341` por el conflicted copy y 4 archivos que no son fotos de la serie, entre ellos
un fruto de tomate dentro de `Tomato___healthy`).
Hallazgo secundario: `Tomato___Late_blight` y `healthy` tienen sesiones de laboratorio con fondo
negro (`GHLB_PS`, ver `pv_tomate.png`), un dominio visual distinto al resto.

### 1d. Chequeos de sanidad

- **Ninguna hoja cruza sesiones**: 0 de 4.166 hojas (con grupo, candidatas + manzana).
- **Hojas en más de una clase:** por construcción (clase, `Leaf #`) no puede ocurrir en los CSV. Pero
  sobre el leaf-map oficial, **1.377 claves de nombre de archivo colisionan entre clases** de
  PlantVillage (p. ej. `RS_HL 6251` existe en `Apple___healthy` y en `Soybean___healthy`;
  `Potato___healthy` con Soybean/Strawberry; `Tomato___healthy` con Blueberry). Son nombres iguales de
  hojas distintas (el loader desambigua por clase), no hojas compartidas. Implicancia: **no se puede
  agrupar por nombre de archivo a secas; hay que usar (clase, hoja).**
- Nombres con "(conflicted copy)" y decimales: ver 1a.

### 1e. Agrupar por numeración contigua no es viable

Sobre las imágenes **con grupo** de tomate+papa+pimiento (3.732 hojas): bloques de números
consecutivos por (clase, sesión), con tolerancia t.

| t | Bloques | Bloque mayor | Hojas reales partidas |
|---:|---:|---:|---:|
| 1 | 2.856 | 631 | 33,36 % |
| 2 | 1.446 | 1.046 | 17,93 % |
| 3 | 874 | 1.046 | 9,78 % |
| 5 | 458 | 1.046 | 3,40 % |
| 10 | 183 | 1.046 | **0,62 %** |

Se confirma el chequeo previo: con t=10 casi no parte hojas (0,62 %), pero porque colapsa todo en
bloques gigantes (**1.046 imágenes** en un solo bloque, YLCV); con t chico parte muchas hojas
(Potato Early blight: 42 % con t=3, Septoria 49 %). No hay una tolerancia que dé bloques de tamaño
útil sin partir hojas. (Manzana, como referencia: 2,53 % con t=10, bloque mayor 959.)

**Tomato mosaic virus (373 imágenes)** es una sola tanda numerada: sesión `PSU_CG`, números 2047–2420,
373 distintos, hueco máximo 2. Con t≥2 es **un único bloque de 373**; con t=1 son 2 bloques (206 + 167).
Es una clase sin grupo y además sin estructura de numeración que permita separarla.

### 1f. No se generó ninguna partición. `data/splits/` solo contiene `.gitkeep`.

## 2. Casi-duplicados (phash de 64 bits, solo clases candidatas)

Tomate+papa+pimiento: PlantVillage 22.787 imágenes; PlantDoc 1.100 (998 train + 102 test; sin
manzana). Umbral ≤6 estricto, ≤10 solo exploratorio.

### 2a. Dentro de PlantDoc (misma clase)

| | ≤6 plano | ≤6 con 8 variantes | ≤10 con 8 variantes |
|---|---:|---:|---:|
| **train↔test** | 4 | **6** | 6 |
| train↔train | 20 | 22 | 25 |

Los 6 pares train↔test (`pares_pd_train_test.png`, verificados visualmente):
4 con distancia 0 (Septoria, bacterial spot, late blight, YLCV: misma imagen) y 2 de papa early blight
con distancia 2 (misma foto de banco de imágenes, con recorte/marca de agua distintos). Los pares
train↔train (`pares_pd_train_train.png`) son en su mayoría distancia 0: misma imagen duplicada dentro
de train (algunas rotadas), incluso con nombres de archivo distintos. Listado completo con rutas y
distancias: notebook §2a.

**Lectura:** el split oficial no es limpio: ~6 % de las imágenes de test de las clases candidatas
(6 de 102) tienen un duplicado en train. Y cualquier partición dev/val propia dentro de train
heredará los 25 duplicados internos si no se agrupa por imagen antes de partir.

### 2b. PlantVillage ↔ PlantDoc (PlantDoc con 4 rotaciones × espejado)

| Umbral | Solo identidad (como sesión 2) | Con 8 variantes |
|---|---:|---:|
| ≤6 | 1 | 3 |
| ≤10 | 43 | 277 (138 de la misma especie, 139 de especies distintas; 92 con imagen de test) |

Los 3 pares ≤6 (sin duplicados reales):
1. PlantDoc test `Potato leaf late blight/irish-blight-symp…` ↔ PlantVillage `Tomato___Spider_mites…`, d=6, variante 2.
2. PlantDoc train `Potato leaf early blight/irish-blight-sy…` ↔ la misma imagen de PlantVillage, d=6, variante 2.
   **Hallazgo lateral:** los pares 1 y 2 son la misma foto de PlantDoc, una vez en train con etiqueta
   `Potato leaf early blight` y otra en test con etiqueta `Potato leaf late blight` (se ven idénticas
   en la grilla). Es un duplicado **con etiquetas contradictorias** entre train y test, hallado de
   rebote: no se buscaron sistemáticamente duplicados entre clases distintas dentro de PlantDoc.
3. PlantDoc train `Tomato leaf late blight/TomatoLateBlightTop.jpg` ↔ PlantVillage
   `Tomato___Late_blight/3de67462-…___RS_Late.B 5148.JPG`, d=6, variante 0.

Grilla de los 12 pares de menor distancia (elegidos por distancia, `pares_pv_pd_menor_distancia.png`):
**todos son falsos positivos**: hojas aisladas sobre fondo claro/liso (PlantDoc) contra hojas sobre
fondo gris (PlantVillage), sin el mismo contenido. El phash a 64 bits es poco discriminante para
este tipo de imagen; el umbral ≤10 produce mayormente ruido (la mitad cruza especies).
Con las 8 variantes el conteo crece (1→3, 43→277) por más comparaciones, no por más duplicados
reales. Los pares marcados train↔test están listados aparte en 2a. Todos los pares con ruta y
distancia: notebook §2b.

### 2d. `TomatoLateBlightTop.jpg`

**No es la misma foto** (`par_TomatoLateBlightTop.png`): PlantDoc es una hoja de tomate con tizón
tardío sobre fondo gris claro (3111×2364 px, marca "N Gregory"); PlantVillage es otra hoja, sobre
una superficie gris oscura con sombra (256×256). El par es un falso positivo del hash (hojas
aisladas con fondo liso). No hay motivo para excluir ninguna de las dos.
**Conclusión sobre contaminación PlantVillage→PlantDoc: no se encontró ninguna con este método.**
Límite: phash no detecta recortes fuertes ni cambios de color/escala grandes.

## 3. Muestra visual (semilla 42)

`random.Random(42)`, rutas ordenadas con `sorted()` antes de muestrear, orden de llamadas fijo del
notebook (la clase de enfermedad de contraste también se eligió al azar: Septoria en tomate,
`Bell_pepper leaf spot` en pimiento —la única—, `Apple rust leaf` en manzana).

### 3a. Clases "sanas" de PlantDoc, desde TRAIN (24 al azar de cada una)
(`pd_train_Tomato_leaf.png`, `pd_train_Bell_pepper_leaf.png`, `pd_train_Apple_leaf.png`)

- **`Tomato leaf`**: mezcla de planta entera en campo/invernadero, hojas sueltas sobre fondo blanco
  (stock photos), primeros planos y una lámina de varias hojas. En la muestra hay al menos una
  hoja de albahaca, y rótulos de archivo como `Bacterial-leafspot…`, `late_blight_tomato_lea…` y
  `curling-issues`: **señales de que no toda la clase es "sana"** (revisar manualmente la clase
  completa sería lo siguiente, no se hizo).
- **`Bell_pepper leaf`**: plantas de pimiento en maceta/almácigo, mayormente verdes; incluye pimienta
  negra/blanca (`stock-photo-black-pepper`, `white-pepper-plant`), que no es pimiento morrón.
- **`Apple leaf`**: heterogénea —manzanas (frutos) enteras o cortadas, ramas con frutos, hojas
  sueltas de banco de imágenes—; poca "hoja sana en el árbol".

Visualmente son **clases de fondo "sin diagnóstico de enfermedad visible", no comparables en estilo
a PlantVillage `healthy`** (hoja única, fondo uniforme). Eso da evidencia a favor de tratarlas
con cautela, pero la decisión de mapearlas o no a `healthy` sigue abierta.
Contraste con enfermedad (12 de cada una): ver `pd_train_contraste_*.png`.

### 3b. PlantVillage (8 por clase; 4 por sesión en clases con varias sesiones)
(`pv_tomate.png`, `pv_papa.png`, `pv_pimiento.png`, `pv_manzana.png`; etiqueta `G` = con grupo)

Todas las sesiones son hoja única sobre fondo gris uniforme, excepto: `GHLB_PS Leaf` (hoja recortada
sobre **fondo negro**, Late blight), `GH_HL Leaf`/`GHLB` (hojas más pálidas y de laboratorio) y
`UF.GRC_*_Lab` (YLCV y Bacterial spot, laboratorio). Es decir, **las sesiones sin grupo no se ven
distintas de las con grupo salvo en esos casos de laboratorio**: el motivo de que no tengan grupo es
de anotación, no visual. Una imagen de `Tomato___healthy` es un fruto rojo, no una hoja
(ruido de etiqueta en origen).

### 3c. PlantDoc test (3 por clase candidata; SOLO etiquetas y tipo de contenido)
(`pd_test_tomate.png`, `pd_test_papa.png`, `pd_test_pimiento.png`, `pd_test_manzana.png`; n = 45, ver
`accesos_test.md`)

Las etiquetas son coherentes con el nombre de la carpeta en las imágenes vistas. Tipo de contenido:
mezcla de planta en campo, hojas de primer plano, imágenes con texto/diagrama (p. ej. una lámina
"#20 Bacterial Spot and Speck", un collage de tres paneles en YLCV) e imágenes de banco de fotos.
Estas imágenes **no se usaron para decidir preproceso ni hiperparámetros.**

### 3d. Resolución y relación de aspecto (sin modelo)

PlantDoc "dev" = carpeta `train` oficial (la partición dev/test propia no está decidida).
PlantVillage: **22.787 de 22.787 imágenes candidatas son 256×256** (aspecto 1,0).

| Cultivo | Split | n | Mediana w×h | Lado menor p10 | % lado menor <224 | Aspecto med. (p10–p90) | % retrato |
|---|---|---:|---|---:|---:|---|---:|
| tomate | dev | 677 | 640×561 | 236 | 8,1 | 1,33 (0,75–1,51) | 24,2 |
| tomate | test | 69 | 638×600 | 250 | 7,2 | 1,33 (0,84–1,59) | 14,5 |
| papa | dev | 206 | 750×532 | 296 | 1,9 | 1,33 (0,85–1,68) | 15,0 |
| papa | test | 16 | 564×490 | 283 | 6,2 | 1,33 (0,76–1,48) | 18,8 |
| pimiento | dev | 115 | 800×667 | 300 | 3,5 | 1,33 (0,75–1,50) | 28,7 |
| pimiento | test | 17 | 640×582 | 264 | 11,8 | 1,33 (0,75–1,50) | 23,5 |
| manzana (cand.) | dev | 244 | 752×600 | 261 | 6,6 | 1,30 (0,75–1,50) | 25,0 |
| manzana (cand.) | test | 29 | 768×514 | 229 | 10,3 | 1,33 (0,59–1,56) | 20,7 |

Por clase (notebook §3d): dev y test se parecen en el aspecto típico (4:3, ~1,33) y en que ~1/4 de
las imágenes son retrato; difieren en el tamaño de las clases chicas por el n de test (6–11
imágenes), donde los porcentajes son ruidosos. Los mínimos de lado menor llegan a 69 px (test,
mosaic virus) y 85 px (dev, `Tomato mold leaf`): hay imágenes que en 224 px se reescalan hacia arriba.
**Lectura:** el cambio de dominio es de contenido, no de formato: PlantDoc trae miles de resoluciones
y relaciones de aspecto distintas contra un único 256×256 cuadrado en PlantVillage; el preproceso
común (resize/crop a 224) deforma o recorta de forma distinta a cada dominio.

## Cosas que NO se hicieron

- Ningún modelo, ninguna característica extraída, ninguna métrica de desempeño.
- Nada en `data/splits/`; no se renombraron las carpetas `train`/`test` de PlantDoc.
- No se decidió ningún mapeo, partición ni exclusión de imágenes.
- No se verificó a ojo cada par train↔train (solo la grilla de los 12 de menor distancia) ni las
  clases "X leaf" completas.

## Decisiones que quedan para el equipo (con costo)

1. **Cómo agrupar PlantVillage para partir sin fuga de hoja.** (a) Usar solo las imágenes con grupo
   (11.411 de tomate + papa + pimiento: pierde ~37 % del tomate y clases enteras: Target Spot, mosaic
   virus; mosaic virus hoy no tiene grupo). (b) Tratar cada sesión sin grupo como un bloque que va
   entero a un lado de la partición (sin fuga intra-sesión, pero sesgo de dominio por sesión).
   (c) Recuperar grupos con reglas nuevas (match por número en Target Spot; número de hoja en el nombre
   para `GHLB`/`GH_HL`); costo: validar a mano, y no hay regla para mosaic/YLCV Lab.
2. **Qué hacer con los duplicados de PlantDoc.** (a) Quitar de train los 6 duplicados del test y
   deduplicar train antes de cualquier partición dev/val (costo: ~30 imágenes menos, pocas por
   clase). (b) Partición propia agrupada por hash de imagen. (c) Dejar el split oficial y reportar
   la fuga (el número OOD queda optimista). Esto también condiciona la decisión de split oficial vs.
   propio de `mapeo_clases.md`.
3. **Qué hacer con las clases "X leaf"** (ver `docs/mapeo_clases.md`).
