# Exploración de datasets — PlantVillage y PlantDoc

> Sesión 2, etapa 0 — exploratorio, nada congelado. No se entrenó ningún modelo ni se
> calculó ninguna métrica de desempeño. Resumen en español para la bitácora y para la
> reunión con el docente; el detalle ejecutable (código, tablas completas, imágenes de
> muestra) está en `notebooks/00_exploracion_datasets.ipynb`.

## 0. Procedencia y verificación de commits

| Dataset | Fuente | Commit | Fecha de descarga |
|---|---|---|---|
| PlantVillage | github.com/spMohanty/PlantVillage-Dataset | `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a` | 2026-09-25 |
| PlantDoc | github.com/pratikkayal/PlantDoc-Dataset | `5467f6012d78d1c446145d5f582da6096f852ae8` | 2026-09-25 |

Ambos hashes se verificaron contra el `HEAD` remoto antes de clonar y contra `git log -1`
en el clon local después. PlantVillage se bajó con sparse-checkout (`raw/color`,
`raw/segmented`, `leaf_grouping`; se omitió `raw/grayscale`, fuera del diseño
experimental). Detalle completo, incluida una nota técnica sobre el filesystem, en
`docs/bitacora/decisiones.md`.

## 1. Inventario PlantVillage (color vs. segmented)

- 38 clases en ambas variantes.
- **color: 54.305 imágenes. segmented: 54.306.** Única diferencia:
  `Grape___Esca_(Black_Measles)` tiene 1383 en color y 1384 en segmented (+1). No es
  clase candidata (tomate/papa/pimiento); queda documentado, no investigado a fondo.
- Integridad: **0 archivos ilegibles** en color y en segmented. **0 imágenes en escala de
  grises** detectadas dentro de `color/` (heurística: variación máxima entre canales RGB
  ≤ 4 en todos los píxeles de la imagen).
- Resolución: color es 100 % uniforme a 256×256. segmented tiene 54.302/54.306 a
  256×256 y 4 imágenes con otra resolución (466×512, 324×512, 470×512, 335×512); una de
  ellas cae en una clase candidata (`Tomato___Spider_mites Two-spotted_spider_mite`).

## 2. Inventario PlantDoc (train / test oficiales)

**28 clases** en la unión de train y test. Oficial: **2.342 en train, 236 en test (2.578
en total)**. Tabla completa por clase en el notebook §2. La propuesta cita 2.598 imágenes
para PlantDoc; el árbol oficial de este commit da 2.578 en train+test — diferencia de 20
sin resolver, queda documentada para que el equipo la revise si le parece relevante.

Confirmado formalmente: **`Tomato two spotted spider mites leaf` tiene 2 imágenes en
train y 0 en test.** Con el split oficial no hay forma de evaluar esa clase fuera de
dominio.

Integridad: **0 archivos ilegibles** (después del rescate técnico, ver abajo).
Resoluciones: **1.382 valores distintos** solo en train, contra un único valor fijo en
todo PlantVillage — es una medida directa de la heterogeneidad de "campo" que motiva el
proyecto.

### Nota técnica: PlantDoc en un filesystem Windows/NTFS (no es una decisión metodológica)

101 de los 2.581 archivos del árbol oficial de PlantDoc no se pudieron escribir con su
nombre original en este filesystem:
- 87 tienen `?` en el nombre (vienen de URLs con query string), carácter inválido en NTFS.
- 8 tienen una ruta que supera el límite de longitud de Windows.
  (87 + 8 + 6 es el conteo por orden de chequeo: primero `?`, después ruta larga. 2 de los 87 archivos con `?` también superan el límite de
  ruta larga, así que en la clasificación exclusiva del CSV (`motivo`) son **85** `caracter_invalido_ntfs` + **10** `ruta_demasiado_larga` + **6** `colision_mayusculas_ntfs` = 101.)
- 6 se pierden silenciosamente por colisión de mayúsculas/minúsculas: PlantDoc tiene
  pares de archivos que difieren solo en capitalización (ej. `Peach-Leaf.jpg` y
  `peach-leaf.jpg`), válidos como dos archivos distintos en git (case-sensitive) pero que
  NTFS (case-insensitive) colapsa en uno solo al hacer checkout — sin error visible.

Los 101 se rescataron extrayendo el contenido por hash de blob (`git cat-file`, no por
ruta) y guardándolo con un nombre local saneado. El árbol oficial de git no se tocó ni se
perdió ninguna imagen. Mapeo nombre original → nombre local en
`docs/bitacora/plantdoc_archivos_renombrados.csv`. Los conteos de esta sección salen de
`git ls-tree` (autoritativos), no del disco, así que no están afectados por este problema
— se verificó que el conteo en disco post-rescate coincide exactamente con el oficial.

### Procedimiento para materializar PlantDoc en Windows (sesión 5c, PC personal; reproducible con el script)

Las tres causas de arriba hacen que **el checkout por defecto falle entero** y que haya que armar el árbol en tres pasos. Los logs de la 5c quedaron en
`data/interim/ejecuciones_pc_personal/` (no versionados).

1. **Checkout por defecto: falla.** `git checkout` del árbol completo termina con 87 errores `invalid path` (los nombres con `?`) y no deja la copia de trabajo usable
   (`checkout_plantdoc.log`, 87 líneas).
2. **Checkout por pathspec literal de las rutas representables.** Se arma la lista con `git ls-tree -r -z` menos los 87 nombres con `?`: **2.494 rutas** (3 archivos de la raíz + 2.491 imágenes), separadas por NUL
   (`pathspec_validos.bin`), y se hace el checkout solo de esas, con pathspec *literal* para que `[`, `*` u otros caracteres de los nombres no se interpreten como patrón:
   `git -C data/raw/plantdoc --literal-pathspecs checkout HEAD --pathspec-from-file=<lista> --pathspec-file-nul` (forma del comando; la lista y el log exactos son los de la 5c).
   Resultado: `Updated 2488 paths` y 6 líneas `unable to create file … Filename too long` en `checkout_selectivo_plantdoc.log`. **Qué rutas largas se materializan depende del largo del prefijo del repo en cada PC**
   (el límite de 260 caracteres cuenta la ruta absoluta); por eso el paso 3 no se apoya en lo que haya quedado en el disco. Las 6 colisiones de mayúsculas están en la lista y git no avisa: queda una sola de cada par.
3. **Rescate por hash de blob** con `scripts/rescatar_archivos_plantdoc.py --rev 5467f6012d78d1c446145d5f582da6096f852ae8`: aplica **siempre** las 101 filas de `docs/bitacora/plantdoc_archivos_renombrados.csv`
   (extrae cada blob con `git cat-file` al nombre local; las colisiones llevan el sufijo `__colision_mayusc` y motivo `colision_mayusculas_ntfs`). Nunca sobrescribe: si un destino existe con otro contenido, lista el conflicto y no escribe nada;
   si git materializó un original que el mapeo renombra (rutas largas habilitadas), avisa y, con `--mover-originales`, lo renombra al nombre local (así el resultado es el mismo en todas las PC). Escribe
   la lista de rutas faltantes (`git ls-tree` contra el disco) y, con `--regenerar`, el mapeo recalculado desde cero para compararlo con el versionado; ambos en `data/interim/` (por defecto no toca el CSV versionado).
4. **Verificación sha1 contra el árbol** (último paso del script, `--sin-verificar` lo omite): el sha1 de blob de cada archivo del disco (con su nombre local si está en el mapeo) debe coincidir con el de `git ls-tree`, sin faltantes ni sobrantes.
   En la 5d: 2.578 esperados, 2.578 en disco, 0 faltan, 0 con sha1 distinto, 0 sobran; el mapeo regenerado (101 filas) es igual al versionado.

## 3. Verificación formal de la exclusión de maíz (leaf_grouping)

Se reprodujo la lógica exacta de `_generate_examples` del loader oficial
(`data/raw/plantvillage/plant_village.py`) contra `leaf_grouping/leaf-map.json`, para
medir qué porcentaje de imágenes de cada cultivo candidato obtiene un `leaf_id` real
(mapeado a una hoja identificada) contra un `leaf_id` de respaldo (no encontrado en el
mapa oficial).

| Cultivo | n imágenes (color) | Mapeadas | Cobertura |
|---|---:|---:|---:|
| Maíz | 3.852 | 0 | **0,00 %** |
| Tomate | 18.160 | 11.411 | 62,84 % |
| Papa | 2.152 | 2.152 | 100,00 % |
| Pimiento | 2.475 | 2.473 | 99,92 % |

El mapa oficial de hojas (40.328 entradas) no tiene **ninguna** clave que mencione
"corn" ni "maize" — la cobertura de 0 % no es un artefacto del cálculo, es que el mapa
oficial simplemente no cubre maíz. Es evidencia objetiva y reproducible para la bitácora;
no reemplaza el motivo original de la decisión del equipo, la complementa.

## 4. Mapeo candidato de clases (tomate, papa, pimiento)

Borrador completo, con las ambigüedades marcadas, en **`docs/mapeo_clases.md`** (no
cerrado). Resumen: tomate y papa mapean casi clase a clase entre los dos datasets. Quedan
5 puntos sin resolver, agenda para la reunión con el docente:

1. `Tomato two spotted spider mites leaf` (PlantDoc) tiene 0 imágenes en test.
2. `Tomato___Target_Spot` (PlantVillage) no tiene equivalente en PlantDoc.
3. `Tomato leaf` (PlantDoc): ¿equivale a `Tomato___healthy` o es una clase genérica
   "sin diagnóstico"?
4. No existe clase de papa sana en PlantDoc.
5. `Bell_pepper leaf` (PlantDoc): mismo problema que el punto 3, para pimiento.

## 5. Insumos para la decisión de partición de PlantDoc

**5a.** Conteos reales train/test oficiales: ya mostrados en la sección 2 y, desglosados
por clase candidata, en el notebook §4.

**5b. ⚠️ NO CONGELADO.** Solo a título ilustrativo, así quedaría una partición propia
estratificada 75/25 sobre el pool (train+test) de las 13 clases candidatas de tomate,
papa y pimiento:

| | Pool | 75 % | 25 % |
|---|---:|---:|---:|
| **Total (13 clases candidatas)** | 1.100 | 824 | 276 |

Tabla completa por clase en el notebook §5b. **No se guardó en `data/splits/`, no es
definitiva, no se usó para evaluar nada** — es solo para comparar números en la reunión
con el docente frente a la opción de usar el split oficial.

## 6. Casi-duplicados PlantVillage ↔ PlantDoc

Hashing perceptual (`phash`, 64 bits), acotado a las clases candidatas para no gastar
cómputo de más: 22.787 imágenes de PlantVillage contra 1.100 de PlantDoc.

- **Umbral estricto (distancia ≤ 6): 1 par sospechoso.**
  - PlantDoc: `train/Tomato leaf late blight/TomatoLateBlightTop.jpg`
  - PlantVillage: `raw/color/Tomato___Late_blight/3de67462-...___RS_Late.B 5148.JPG`
  - Recomendado: inspección visual manual de este par antes de decidir si se excluye.
- Umbral más laxo (≤ 10, solo exploratorio): 43 pares. La mayoría cruza clases sin
  relación temática (ej. "Tomato Septoria leaf spot" con "Pepper healthy"), lo que
  sugiere similitud genérica de forma/color de hoja y no contaminación real. Se listan en
  el notebook §6 igual, por si el equipo quiere revisarlos.

## 7. Tamaño en disco y tiempo de descarga

> **Unidades (aclaración de la sesión 5d):** los tamaños se calcularon dividiendo por 2^20 y 2^30, es decir, son **MiB y GiB** aunque se escriban "MB" y "GB". Verificado para las cifras de la sesión 5 (459 MiB = 481 MB decimales;
> 326 MiB = 341 MB; PlantVillage color 353 MiB y segmented 196 MiB = 370 y 206 MB). De la tabla de abajo (sesión 2) solo se re-midió el clon de PlantVillage (2,56 GiB, coincide con los 2,57); el resto se presume en la misma unidad y no se re-midió.

| | Clon completo (incl. `.git`) | Solo imágenes |
|---|---:|---:|
| PlantVillage | 2,57 GB | 1.301 MB |
| PlantDoc | 2,10 GB | 950 MB |
| **Total** | **4,66 GB** | **~2,2 GB** |

**Alerta operativa:** la máquina usada en esta sesión tenía muy poco espacio libre antes
de empezar (~5 GB sobre un disco de 465 GB al 99 % de uso). Con ese margen, clonar ambos
datasets completos (con historial `.git`) casi agota el espacio disponible. Es una
condición de esta máquina, no del proyecto, pero es evidencia directa para la decisión
Colab-vs-local: **con este margen local, conviene trabajar con al menos uno de los dos
datasets en Colab/almacenamiento externo**, incluso si en teoría los ~2,2 GB de imágenes
puras caben.

**Nota de proceso:** clonar PlantDoc con `--filter=blob:none` y después hacer un checkout
selectivo de ~2.500 archivos fue extremadamente lento (~0,2 MB/s; a ese ritmo habría
tardado horas) porque git pedía cada blob faltante de a uno. Cambiar a un
`git fetch --refetch` (bajar todos los objetos de una sola vez) tardó 65 segundos a
~14,8 MiB/s. Para la próxima descarga: clonar PlantDoc completo desde el principio en vez
de `blob:none` + checkout incremental — a diferencia de PlantVillage, ahí no hay
sparse-checkout que justifique el filtro, porque se necesita casi todo el árbol de todos
modos.

## Qué NO se hizo en esta sesión

- No se entrenó ningún modelo ni se calculó ninguna métrica de desempeño — ver la línea
  correspondiente en `docs/bitacora/accesos_test.md`.
- No se escribió nada en `data/splits/`.
- No se decidió la partición de PlantDoc ni ninguna de las 5 ambigüedades del mapeo de
  clases — quedan para la reunión con el docente.

## Próximos pasos sugeridos

1. Llevar a la reunión con el docente: las 5 ambigüedades de `docs/mapeo_clases.md`, la
   elección entre split oficial vs. partición propia para PlantDoc, y si pimiento entra
   al alcance final.
2. Revisar visualmente el par de casi-duplicado de la sección 6 antes de decidir si se
   excluye alguna imagen.
3. Definir entorno y `requirements.txt` — pendiente desde la sesión 1 (`handoff.md`).
4. Con la alerta de espacio en disco de la sección 7, decidir dónde vive cada dataset de
   acá en adelante (Colab, disco externo, o liberar espacio local).
