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

## f. Intervención de la etapa 4 (sesión 10, registrada el 2026-11-06 antes de entrenar)
- Una sola variante: fine-tuning parcial + aumentación dirigida, desde los pesos del modelo base. No separa el efecto de
  descongelar del de aumentar (limitación).
- Entrenables: `features[11]`, `features[12]` y `fc`; el resto congelado y en modo `eval` durante el entrenamiento (BatchNorm con las
  estadísticas de ImageNet fijas); solo los bloques entrenables actualizan su BatchNorm. Adam, LR 1e-4, batch 64, máximo 20 épocas,
  paciencia 5, semilla 42 determinista.
- Aumentación solo en PV-train (`RandomResizedCrop(224, scale=(0.6, 1.0))`, `RandomHorizontalFlip`, `RandomRotation(20)`,
  `ColorJitter(0.3, 0.3, 0.3)`). PV-val, PD-dev y los tests: `aplastar` 224 (§a, invariante 3).
- Selección del checkpoint: mejor macro-F1 en PV-val con mosaic. PlantDoc no entrena ni elige (invariante 1).
- PD-dev: solo informativo (macro-F1 con IC y diferencia pareada contra el base) y CAM sobre la misma muestra fija de §c.
- PV-test y PD-test: una pasada, con filas de acceso nuevas commiteadas antes (§e). Se reportan con IC el macro-F1 de cada test
  y la brecha (§b), en una tabla base vs. variante. No se concluye sobre diferencias dentro del IC.

## g. Calibración y abstención (sesión 11, registrada el 2026-11-07 antes de calcular cualquier número nuevo)
1. **Modelo.** Se calibra el de mayor macro-F1 con mosaic en PV-val entre el base (`02_modelo_base_20261104_201229`) y la
   variante (`04_aumentacion_20261106_232146`), leído de los JSON ya versionados en `results/`, sin calcular nada nuevo. Mayor
   estricto; si empatan, el base. No usa PlantDoc ni ningún test. Aplicación: base 0,95589
   (`results/02_modelo_base/02_modelo_base_20261104_201229/resultados.json`), variante 0,96584
   (`results/04_aumentacion/04_aumentacion_20261106_232146/resultados.json`) → se calibra la **variante**.
2. **Temperatura.** Un único escalar T que minimiza la NLL en PD-dev (696 imágenes, `aplastar` 224, §a), búsqueda en grilla de
   0,5 a 10 con paso 0,05. Si el óptimo cae en un borde de la grilla, se frena y se avisa. Dividir los logits por T no cambia el
   argmax: el macro-F1 no se mueve.
3. **ECE.** 15 bins de igual ancho sobre la confianza máxima. En PD-dev se estima sin usar las imágenes que ajustaron T:
   5 particiones estratificadas por clase (semilla 42); en cada una se ajusta T con las otras 4 y se calibran las de esa
   partición; con las 696 predicciones calibradas fuera de partición se calcula un ECE. Se reporta el ECE antes (T = 1) y
   después, con IC bootstrap (§b: 2.000 réplicas, estratificado por clase, semilla 42). La T final (la que usan el umbral,
   PD-test y la app) se ajusta con todo PD-dev.
4. **Umbral.** τ = 0,5 fijo sobre la probabilidad máxima calibrada: responde si p ≥ 0,5; si no, se abstiene. No se busca
   nada en la curva de cobertura.
5. **Métricas del sistema** (PD-dev fuera de partición y PD-test): cobertura (fracción respondida), acierto sobre lo
   respondido, tasa de abstención y errores de alta confianza (respondidas con p ≥ τ y mal clasificadas, como fracción del
   total), todas con IC bootstrap (§b); más los mismos errores de alta confianza sin calibrar (T = 1, p ≥ 0,5).
6. **PD-test.** Una sola pasada con el modelo, T y τ ya fijos, bajo una fila nueva de `accesos_test.md` commiteada antes
   (§e). Si las predicciones por imagen de la sesión 10 ya guardaron logits, se pueden usar sin pasar imágenes, pero la fila
   hace falta igual. No se toca PV-test. Nada se ajusta después de ver estos números.
