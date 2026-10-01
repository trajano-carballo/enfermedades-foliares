# Handoff

> Se actualiza al cierre de cada sesión de trabajo. Lo más reciente arriba.

## 2026-10-01 — Sesión 3: chequeos previos a las decisiones (sin entrenar nada)

**Hecho** (detalle y tablas en `docs/exploracion_sesion3.md`; notebook `notebooks/00b_chequeos_previos.ipynb`)
- Verificado que CLAUDE.md y README.md ya decían "solo CAM, sin Grad-CAM"; en este archivo se
  marcaron como obsoletas dos menciones viejas de Grad-CAM (sesión 1). Hashes de ambos datasets OK.
- Grupos de hoja (PlantVillage): tomate 11.411/18.160 con grupo (coincide con el loader). Sin grupo =
  sesiones enteras (YLCV Lab, Target_Spot por prefijo de CSV distinto, mosaic sin CSV, parte de Late
  blight/Septoria/healthy). Numeración contigua: no viable (bloque de 1.046 con t=10).
- Casi-duplicados: **PlantDoc oficial tiene 6 pares train↔test y 25 train↔train**; uno train↔test con
  etiquetas contradictorias (papa early vs late blight). PlantVillage↔PlantDoc con 8 variantes: sin
  duplicados reales; `TomatoLateBlightTop.jpg` no es la misma foto.
- Muestras visuales (semilla 42) y estadísticas de resolución: las clases "X leaf" de PlantDoc son
  heterogéneas y con ruido de etiqueta; PlantVillage es 256×256 fijo contra miles de resoluciones.
- Nuevo código: `src/foliares/data/grupos_hoja.py`, `visual.py`, y funciones de hashing con 8
  variantes en `duplicados.py`. Config `configs/00b_chequeos_previos.yaml`. Figuras en `docs/figuras_sesion3/`.
- `docs/mapeo_clases.md`: agregada la evidencia del paper para las clases "X leaf" como propuesta
  pendiente de verificación (NO cerrada). Aviso: solo pude confirmar en el abstract "13 especies / 17
  clases de enfermedad / 2.598 imágenes"; el "27 = 17 + 10" viene del equipo, y el árbol local tiene
  28 carpetas (18 + 10) y 2.578 imágenes.
- Registrado en `accesos_test.md` (n=45 imágenes de test vistas + estadísticas de cabecera).
- No se escribió nada en `data/splits/`; no hay modelos ni métricas.

**Próximos pasos**
1. Reunión con el docente con `docs/exploracion_sesion3.md` + `docs/mapeo_clases.md`.
2. Con las decisiones: mapeo definitivo, deduplicado/partición de PlantDoc, regla de agrupamiento por hoja, y recién ahí congelar `data/splits/`.
3. Pendientes viejos: entorno y `requirements.txt`, dónde viven los datasets (disco local casi lleno: 1,6 GB libres al cierre).

**Decisiones pendientes del equipo** (opciones con costo en `docs/exploracion_sesion3.md`, "Decisiones que quedan")
- Cómo agrupar PlantVillage (solo con grupo / sesión como bloque / reglas nuevas para recuperar grupos).
- Duplicados de PlantDoc (quitar y deduplicar / partición propia agrupada por hash / split oficial reportando la fuga).
- Las clases "X leaf" (sana vs. sin diagnóstico), las 5 ambigüedades previas, pimiento sí/no, split de PlantDoc, priorización por plazos.

**Riesgos abiertos**
- Disco local: 1,6 GB libres. Las figuras (7,5 MB) y el notebook (~200 KB) son chicos, pero no hay margen para más datos.
- El hash perceptual (64 bits) da muchos falsos positivos con hojas aisladas; no detecta recortes fuertes.

## 2026-09-25 — Sesión 2: exploración de datasets (inventario, sin entrenar nada)

**Hecho**
- Descargados PlantVillage (commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`, sparse
  `raw/color`+`raw/segmented`+`leaf_grouping`) y PlantDoc (commit
  `5467f6012d78d1c446145d5f582da6096f852ae8`), hashes verificados. **No versionados** (ver
  `.gitignore`); para reproducir la descarga, ver `docs/bitacora/decisiones.md`.
- `notebooks/00_exploracion_datasets.ipynb` ejecutado de punta a punta: inventario y
  chequeo de integridad de ambos datasets, verificación formal de la exclusión de maíz
  (leaf_grouping), mapeo candidato de clases, simulación ilustrativa de partición (NO
  congelada), casi-duplicados por hashing perceptual, tamaño en disco.
- Resumen para el docente: `docs/exploracion_datasets.md`. Borrador de mapeo:
  `docs/mapeo_clases.md` (5 ambigüedades marcadas, sin cerrar).
- Código reutilizable nuevo en `src/foliares/data/`: `taxonomia.py`, `inventario.py`,
  `leaf_grouping.py`, `duplicados.py`. Config: `configs/00_exploracion_datasets.yaml`.
- Actualizados CLAUDE.md y README.md para reflejar: CAM sin Grad-CAM, exclusión de maíz,
  base tomate+papa evaluando sumar pimiento (decisiones ya tomadas por el equipo antes de
  esta sesión).
- Hallazgo técnico (no metodológico): 101 archivos de PlantDoc no son representables con
  su nombre original en NTFS (87 por `?`, 8 por ruta larga, 6 por colisión de
  mayúsculas/minúsculas). Rescatados por hash de blob sin alterar el árbol oficial; mapeo
  en `docs/bitacora/plantdoc_archivos_renombrados.csv`. Script reutilizable:
  `scripts/rescatar_archivos_plantdoc.py`.
- Hallazgo operativo: la máquina de esta sesión quedó con poco margen de disco (~5 GB
  libres antes de empezar; ambos clones completos ocuparon 4,66 GB). Ver
  `docs/exploracion_datasets.md` §7 para la recomendación sobre Colab vs. local.

**Estado del plan** (propuesta §8): Etapa 0 cerrando; insumos para etapa 1 listos.

**Próximos pasos**
1. Reunión con el docente: cerrar las 5 ambigüedades de `docs/mapeo_clases.md` y decidir
   split oficial vs. partición propia de PlantDoc (sección 5 de
   `docs/exploracion_datasets.md` trae los números de ambas opciones).
2. Revisar visualmente el par de casi-duplicado detectado (sección 6 del mismo doc) antes
   de decidir si se excluye alguna imagen.
3. Definir entorno y `requirements.txt` (pendiente desde sesión 1 — ver abajo). Por ahora
   se instalaron sueltos en el Python del sistema: `pillow`, `imagehash`, `pandas`,
   `numpy`, `pyyaml`, `matplotlib`, `nbconvert`/`nbclient`/`ipykernel`.
4. Con el mapeo cerrado: mapeo de clases definitivo, control de casi-duplicados sobre el
   alcance final, y congelamiento de particiones (`data/splits/`) — recién ahí arranca la
   etapa 1 en serio.

**Decisiones pendientes del equipo**
- Las 5 ambigüedades de `docs/mapeo_clases.md`.
- Partición de PlantDoc: split oficial vs. propia (insumos listos, decisión pendiente).
- Si pimiento entra al alcance final.
- Priorización por plazos (pedido docente) — sigue pendiente desde sesión 1, ver abajo.
- Dónde vive cada dataset de acá en adelante (Colab, disco externo, o liberar espacio
  local) — ver hallazgo operativo arriba.

**Riesgos abiertos**
- Espacio en disco local ajustado (ver hallazgo operativo).
- Ver también los 3 pendientes de sesión 1 que siguen sin resolver: `requirements.txt`,
  reparto de roles, tamaño de muestra fija para explicabilidad (ya no aplica a Grad-CAM,
  que se descartó del alcance — sigue aplicando si se quiere una muestra fija para
  inspeccionar CAM cualitativamente).

## 2026-09-24 — Sesión 1: estructura inicial

**Hecho**
- Estructura de carpetas, `README.md`, `CLAUDE.md`, `.gitignore`, bitácoras vacías.
- Propuesta copiada a `docs/propuesta_entregable1.md`.
- Registrada la devolución docente sobre Grad-CAM → CAM en deploy.

**Estado del plan** (propuesta §8): Etapa 0 en curso.

**Próximos pasos (etapa 0 → 1)**
1. Definir entorno: versión de Python, `requirements.txt` (torch, torchvision, gradio,
   imagehash, scikit-learn, pyyaml) y cómo se sincroniza Colab ↔ repo.
2. Cerrar alcance: especies y clases del subconjunto común (tomate y maíz como punto de
   partida). Borrador de mapeo en `docs/mapeo_clases.md`.
3. Reparto de roles con rotación (registrar en bitácora).
4. Scripts de descarga e integridad (`src/foliares/data/`).

**Decisiones pendientes del equipo**
- Mapeo de clases PlantVillage ↔ PlantDoc.
- Priorización por plazos (pedido docente). Propuesta de corte a discutir:
  - *Imprescindible*: modelo base, brecha con bootstrap, matrices de confusión, CAM,
    una intervención, calibración + abstención, app Gradio.
  - *Si hay tiempo*: segunda intervención, combinación de ambas. (Grad-CAM comparado: descartado en la sesión 2, solo CAM.)
  - *Solo si todo lo anterior está*: backbone de contingencia (ResNet-18/EfficientNet-B0).
- ~~Tamaño de la muestra fija para Grad-CAM~~ (obsoleto: Grad-CAM descartado; solo CAM).

**Riesgos abiertos**
- Ninguno nuevo.
