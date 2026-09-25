# Mapeo de clases PlantVillage ↔ PlantDoc — borrador, NO cerrado

> Generado en la sesión 2 (exploración de datasets, etapa 0). Es un borrador con las
> ambigüedades marcadas para que el equipo decida; no fija ninguna inclusión ni exclusión
> de clase. Alcance: tomate, papa y pimiento (maíz queda fuera del subconjunto común por
> decisión ya tomada — ver `docs/bitacora/decisiones.md` y `docs/exploracion_datasets.md`
> §3 para la evidencia formal).

Fuente de los conteos:
- PlantVillage: `raw/color/<clase>/`, commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`.
- PlantDoc: árbol oficial `train/` y `test/` tal como lo publicaron los autores, commit
  `5467f6012d78d1c446145d5f582da6096f852ae8` (conteos tomados del árbol de git, no del
  checkout local — ver nota sobre nombres de archivo no válidos en NTFS más abajo, que no
  afecta estos conteos porque se leyeron del árbol, no del disco).

## Tomate

| PlantVillage (color) | n | PlantDoc | train | test |
|---|---:|---|---:|---:|
| `Tomato___Bacterial_spot` | 2127 | `Tomato leaf bacterial spot` | 101 | 9 |
| `Tomato___Early_blight` | 1000 | `Tomato Early blight leaf` | 79 | 9 |
| `Tomato___Late_blight` | 1909 | `Tomato leaf late blight` | 101 | 10 |
| `Tomato___Leaf_Mold` | 952 | `Tomato mold leaf` | 85 | 6 |
| `Tomato___Septoria_leaf_spot` | 1771 | `Tomato Septoria leaf spot` | 140 | 11 |
| `Tomato___Spider_mites Two-spotted_spider_mite` | 1676 | `Tomato two spotted spider mites leaf` | **2** | **0** |
| `Tomato___Target_Spot` | 1404 | — | — | — |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 5357 | `Tomato leaf yellow virus` | 70 | 6 |
| `Tomato___Tomato_mosaic_virus` | 373 | `Tomato leaf mosaic virus` | 44 | 10 |
| `Tomato___healthy` | 1591 | `Tomato leaf` (**ambiguo**, ver abajo) | 55 | 8 |

**Ambigüedades sin resolver (tomate):**

1. **`Tomato two spotted spider mites leaf` tiene 2 imágenes en train y 0 en test.**
   Confirmado contra el árbol oficial del commit fijado. Con 0 en test no se puede evaluar
   OOD para esa clase; queda documentado para que el equipo decida si se descarta del
   subconjunto común o se maneja de otro modo. No se decide acá.
2. **`Tomato___Target_Spot` no tiene equivalente en PlantDoc.** No existe ninguna clase de
   PlantDoc cuyo nombre sugiera "target spot" para tomate. Confirmado por inspección del
   listado completo de 28 clases de PlantDoc (`docs/exploracion_datasets.md` §2).
3. **`Tomato leaf` (PlantDoc) — ¿sana o "sin diagnóstico"?** El nombre no dice explícitamente
   "healthy". Es la única clase de tomate en PlantDoc sin calificador de enfermedad, lo cual
   sugiere que podría corresponder a `Tomato___healthy`, pero también podría ser una clase
   genérica de "hoja de tomate sin enfermedad identificada" (recolectada de búsquedas de
   imágenes de internet, no de un diagnóstico agronómico confirmado — a diferencia de
   PlantVillage, que sí tiene curación experta). El equipo tiene que decidir si la trata como
   equivalente de `Tomato___healthy` o si la excluye por incertidumbre de etiqueta.

## Papa

| PlantVillage (color) | n | PlantDoc | train | test |
|---|---:|---|---:|---:|
| `Potato___Early_blight` | 1000 | `Potato leaf early blight` | 109 | 8 |
| `Potato___Late_blight` | 1000 | `Potato leaf late blight` | 97 | 8 |
| `Potato___healthy` | 152 | — | — | — |

**Ambigüedad sin resolver (papa):**

4. **No hay clase de papa sana en PlantDoc.** Confirmado por inspección del listado completo
   de 28 clases — no existe ninguna variante de "Potato leaf" sin calificador de enfermedad.
   Si el equipo quiere medir la brecha ID→OOD también sobre la clase sana de papa, no hay
   forma de hacerlo con PlantDoc tal como está.

## Pimiento (evaluando si se suma)

| PlantVillage (color) | n | PlantDoc | train | test |
|---|---:|---|---:|---:|
| `Pepper,_bell___Bacterial_spot` | 997 | `Bell_pepper leaf spot` | 62 | 9 |
| `Pepper,_bell___healthy` | 1478 | `Bell_pepper leaf` (**ambiguo**, ver abajo) | 53 | 8 |

**Ambigüedad sin resolver (pimiento):**

5. **`Bell_pepper leaf` (PlantDoc) — mismo problema que `Tomato leaf`.** No dice "healthy"
   explícitamente; podría ser la clase sana o una clase genérica de "hoja sin diagnóstico".
   Misma decisión pendiente que en el punto 3, y probablemente conviene resolverlas juntas
   (criterio consistente entre tomate y pimiento).

## Maíz — fuera del subconjunto común (referencia, no forma parte del mapeo candidato)

PlantDoc sí tiene tres clases de maíz (`Corn Gray leaf spot`, `Corn leaf blight`,
`Corn rust leaf`), pero maíz queda excluido del subconjunto común por decisión ya tomada
del equipo. La verificación formal de esa exclusión (cobertura de agrupamiento de hoja vía
`leaf_grouping/leaf-map.json`) está en `docs/exploracion_datasets.md` §3: maíz tiene 0% de
cobertura en el mapa oficial de hojas (0 de 3852 imágenes de color mapean a un `leaf_id`
real), contra 100% en papa y 99,92% en pimiento. No se vuelve a discutir acá.

## Nota técnica: PlantDoc en un filesystem Windows/NTFS

El checkout directo de PlantDoc falla en Windows por tres motivos de filesystem, ninguno
metodológico. Se resolvieron sin alterar el árbol oficial ni perder imágenes, rescatando
cada archivo afectado con `git cat-file` (por hash de blob, no por ruta) y guardándolo con
un nombre local saneado:

1. **87 archivos con `?` en el nombre** (vienen de URLs de descarga con query string, ej.
   `imagen.jpg?id=123`): carácter no válido en NTFS. Se reemplaza por `_`.
2. **8 archivos con ruta demasiado larga** para el límite de Windows (nombres descriptivos
   largos, típicos de bancos de imágenes): se truncan con un sufijo hash de 8 caracteres
   para evitar colisiones.
3. **6 archivos perdidos por colisión de mayúsculas/minúsculas**: PlantDoc tiene pares de
   archivos cuyo nombre difiere solo en capitalización (ej. `Peach-Leaf.jpg` y
   `peach-leaf.jpg` en `train/Peach leaf/`), válidos como dos archivos distintos en git
   (case-sensitive) pero que NTFS (case-insensitive) colapsa en uno solo al hacer checkout:
   el segundo sobrescribe al primero en silencio, sin error. Se detecta comparando el
   conteo esperado (árbol de git) contra el conteo real en disco por clase, y se rescata el
   que faltaba con un sufijo `__colision_mayusc`.

El mapeo nombre original → nombre local de los 101 archivos afectados (95 de los puntos 1
y 2, 6 del punto 3) está en `docs/bitacora/plantdoc_archivos_renombrados.csv`. Los conteos
de esta tabla (arriba) son los oficiales del árbol de git, no los del disco local, así que
no están afectados por ninguno de estos tres problemas. El detalle completo está en
`docs/exploracion_datasets.md` §2.

## Próximo paso

Estas cinco ambigüedades (puntos 1 a 5) son la agenda para la reunión con el docente sobre
la partición dev/test de PlantDoc. No se resuelven en esta sesión.
