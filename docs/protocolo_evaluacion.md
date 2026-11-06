# Protocolo de evaluación (etapa 3: brecha ID → OOD y CAM)

Fijado por el equipo el 2026-11-04 (sesión 9), **antes** de calcular cualquier número sobre PlantDoc o PV-test. Se registra
también en `docs/bitacora/decisiones.md`. Modelo evaluado: el modelo base de la sesión 8 (run
`02_modelo_base_20261104_201229`, commit `74ff5de`), entrenado con la geometría `aplastar` 224 sobre PV-train.

Invariantes que aplica: 1 (PlantDoc-dev elige la geometría; PlantDoc-test se evalúa una sola vez por variante final), 3
(misma geometría y mismos pesos para PV-test y PlantDoc), 6 (toda métrica OOD con IC bootstrap) y CAM sin backward.

## a. Geometría de evaluación (se elige en PlantDoc-dev)
- Se evalúa el modelo base en PD-dev con `aplastar`, `resize_recorte` y `relleno`, tamaño 224.
- Gana la de mayor macro-F1 en PD-dev, pero **solo reemplaza a `aplastar`** si el IC 95 % bootstrap **pareado** (las mismas
  réplicas de imágenes para las dos geometrías) de la diferencia macro-F1(g) − macro-F1(aplastar) excluye el 0.
- Si ninguna lo logra, queda `aplastar`. Si dos lo logran, gana la de mayor diferencia media.
- La geometría elegida se fija en `configs/03_brecha.yaml` y es la misma para PV-test y PD-test (invariante 3).

**Aclaración 2026-11-04, antes de calcular cualquier número de PlantDoc** (decisión del equipo; el texto de arriba se deja como
estaba). "Excluye el 0" también lo cumple una geometría peor (IC entero debajo de 0), y la redacción dejaba un caso ambiguo
(la de mayor macro-F1 que no supera el criterio). La regla que se aplica es:
- una geometría g **califica** solo si el límite inferior del IC 95 % pareado de macro-F1(g) − macro-F1(`aplastar`) es **> 0**;
- entre las que califican, gana la de **mayor diferencia media** (promedio de las réplicas bootstrap);
- si ninguna califica, queda `aplastar`.

## b. Bootstrap
- 2.000 réplicas, IC 95 % por percentiles (2,5 y 97,5), semilla 42.
- Remuestreo de imágenes con reposición, **estratificado por clase** (cada réplica conserva el número de imágenes de cada clase).
- Pareado (regla a): las mismas réplicas de índices para las dos geometrías; el IC es el de la diferencia por réplica.
- Brecha = macro-F1(PV-test) − macro-F1(PD-test): remuestreo **independiente** de cada test y diferencia por réplica.
- Mismo esquema para el macro-F1 sin mosaic de PV-test (definición `sin_imagenes_de_la_clase`, decisión del 2026-11-04).

## c. Muestra fija para CAM
- PlantDoc-dev, 2 imágenes por clase: 1 acierto y 1 error cuando haya; si no, 2 aciertos o 2 errores. Se eligen con semilla
  42 **después** de predecir con la geometría elegida.
- Más 1 imagen por clase de PV-val, como contraste.
- **Nunca** imágenes de un test.

## d. Lectura de imágenes
- `ImageOps.exif_transpose` y composición del canal alfa sobre blanco, en ambos dominios, dentro de la misma función de
  preproceso (`foliares.data.preproceso`).
- Control: en PV-val no debería cambiar ningún píxel (o se reporta cuántas imágenes cambian) y el macro-F1 de PV-val del
  modelo base no se mueve (0,9559); si se mueve, se frena y se avisa.

## e. Guarda de acceso a los tests
- El script de evaluación sobre cualquier test (PV-test, PD-test) exige `--fila-accesos N` y verifica que esa fila exista en
  `docs/bitacora/accesos_test.md` **commiteada** (en `HEAD`), que el test de la fila sea el que se pide, que mencione la
  config que se usa y que su commit sea un ancestro de `HEAD` sin cambios posteriores en la config, `src/` ni `scripts/`, con
  el árbol limpio. Si algo no se cumple, no evalúa.
- Orden: protocolo + decisiones + código de la guarda se commitean antes de cualquier número de PlantDoc; las filas de
  acceso se commitean antes de cada evaluación de un test.

**Aclaración 2026-11-05** (el texto de arriba se deja como estaba): los flags del script son `--fila-accesos-pv` (fila de PV-test) y `--fila-accesos-pd` (fila de PD-test), no un único `--fila-accesos N`; la verificación de cada fila es la descrita en esta sección.

**Aclaración 2026-11-06** (sesión 10; el texto de arriba se deja como estaba): la guarda además (1) rechaza una fila ya usada: cuenta como usada si algún JSON versionado de `results/` la trae en su bloque `accesos`; y (2) exige que desde el commit de la fila no cambie nada en `configs/` (no solo la config de evaluación), `src/` ni `scripts/`.
