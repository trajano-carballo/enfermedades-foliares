# Handoff

> Se actualiza al cierre de cada sesión de trabajo. Lo más reciente arriba.

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
  - *Si hay tiempo*: segunda intervención, combinación de ambas, Grad-CAM comparado.
  - *Solo si todo lo anterior está*: backbone de contingencia (ResNet-18/EfficientNet-B0).
- Tamaño de la muestra fija para Grad-CAM.

**Riesgos abiertos**
- Ninguno nuevo.
