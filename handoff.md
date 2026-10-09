# Handoff

> Se actualiza al cierre de cada sesión de trabajo. Lo más reciente arriba.

## 2026-10-09 — Sesiones 5c y 5d: PlantDoc en la PC personal, reglas de revisión de un revisor y arreglos de la revisión de 92725bc (sin modelos)

**Sesión 5c (PC personal; lo que consta en los artefactos de `data/interim/ejecuciones_pc_personal/`)**
- PlantDoc materializado en la PC personal con el procedimiento de tres pasos (checkout por defecto falla entero → checkout por pathspec literal de las rutas representables → rescate por hash de blob de las 101 filas del mapeo → verificación sha1 contra el árbol);
  documentado en `docs/exploracion_datasets.md` (§2). Notebooks 01, 01b y 00c re-ejecutados en copias; la planilla no se escribió.
- **Reproducibilidad entre PC:** los sha256 de ambos candidatos son **idénticos en las dos PC** (antes del cambio a LF): PlantVillage `4caedf7a…a4498`, PlantDoc `04a122a3…f01a1554`.

**Sesión 5d — hecho** (decisiones D1–D3 en `decisiones.md`; 94 tests pasan con `python -m pytest -rs`, 0 saltados)
- **D1** un solo revisor para la planilla (`revisor_1`; `revisor_2` se ignora con aviso); **D2** regla mecánica sin excepciones para los pares ≤ 6 dudosos; **D3** los pares 7–10 los revisa un revisor y `misma_foto` se une a los grupos antes del split.
  Reglas configurables en `configs/01b_depuracion_plantdoc.yaml` (`revision_etiquetas`, `pares_7_10`). El resultado es COMPLETO solo con planilla y pares completos; si no, PROVISIONAL.
  Errores claros ante planilla/pares con `;`, columnas inesperadas, rutas que no están en el manifiesto o repetidas. `congelar.py` trata un `.meta.json` sin `revision_estado` como PROVISIONAL (y solo `COMPLETO` habilita; PlantVillage se declara `requiere_revision: false`).
- `.meta.json` de 01b: modo de revisión, qué falta (`revision_pendiente`), exclusiones por estado/clase/motivo, pares aplicados, commit del repo y si había cambios sin commitear; el de 01, el commit.
- Guards: el 00c no pisa la planilla; el 01b no pisa `plantdoc_pares_7_10_revision.csv` ni pierde `decision_equipo`. Corregido el print final del 00c.
- `.gitattributes` (`eol=lf`; png/jpg/tar/npz binarios) y todo CSV/`.meta.json` que escribe el código sale con LF (`foliares.utils.archivos`; en los notebooks, `lineterminator="\n"`).
- Script de rescate reescrito (colisiones, nunca sobrescribe, salidas en `data/interim/`, rutas faltantes por `git ls-tree`, aplica siempre las 101 filas, verificación sha1). `paquete.py` sin archivos parciales. `requires-python >= 3.11.4`; `*.tar` y `*.tar.sha256` en `.gitignore`.
- Docs: 87/8 vs 85/10 aclarado, MiB/GiB, mosaic como excepción de "val/test sin sesiones sin grupo", línea 174 de `exploracion_sesion5.md` (2 de los 3 pares 7–10 cruzan dev y test propuestos), `mapeo_clases.md` ("pendiente de confirmar a mano"), notas en `accesos_test.md`.
- Re-ejecutados 01 y 01b (copias `*.5d.ejecutado.ipynb`) con planilla y pares vacíos: **mismos números que la 5c** (PlantVillage 11.968 / 2.207 / 2.170; PlantDoc 1.010 conservadas → dev 707 / test 303; 60 / 28 / 0 / 2), estado PROVISIONAL, y el **mismo sha256 normalizado** (ordenado por ruta, LF, sin BOM):
  PlantVillage `319739c0a043e1bf00af29f714486f4c02f8f21dcef439a20c3effbb4ed776a6`, PlantDoc `85e89d230e37d5dbb10145472ab7a975f12ac3ceb28458d0310c2dd5a6a8a46e`.
- **sha256 nuevos "tal cual" (ya con LF)**, para comparar entre PC: PlantVillage `1ff04397bed8182b901609f2e0d3e0e8b3ceed3ac4b7739512b64f9b366713dd`, PlantDoc `17f0a7d49a65d83cf77f7820bbace8f99442aeafecf22fe406a30398217da723`.
- Simulación sobre copias temporales de la planilla y de los pares (los archivos reales no se tocaron): con 3 `descartar` y pares `misma_foto` el modo pasa a COMPLETO y solo cambian 4 clases (`Tomato leaf`, `Bell_pepper leaf`, `Tomato Early blight leaf`, `Potato leaf early blight`).

**Qué quedó a medias**
- **Revisión humana pendiente:** el candidato de PlantDoc sigue **PROVISIONAL** (planilla 124 de 124 sin completar; pares 7–10: 3 de 3 sin decisión). `congelar_splits.py` no congela así.
- `data/splits/` sigue vacío; `REPO_URL` y `DRIVE_DIR` del notebook de Colab sin completar; la rama de torch de `fijar_semillas` sin probar.
- Los manifiestos de `data/interim/` generados en la 5d registran `repo_con_cambios_sin_commitear: true` (los cambios de la 5d aún no están commiteados): **re-ejecutar 01 y 01b con el árbol limpio antes de congelar**.
- Sin ejecutar en datos reales: `congelar_splits.py`, `empaquetar_subconjunto.py`, `verificar_paquete.py`. La propuesta (`docs/propuesta_entregable1.md`) sigue sin editar (lista en `exploracion_sesion5.md` §5).

**Próximos pasos**
1. Un integrante completa `revisor_1` (`mantener`/`descartar`, motivo 1/2/3) en `docs/bitacora/revision_etiquetas_planilla.csv` (124 filas) y `decision_equipo` (`misma_foto`/`distinta`) en `docs/bitacora/plantdoc_pares_7_10_revision.csv` (3 pares). Guardar como CSV UTF-8 delimitado por comas (no `;`).
2. Commitear la 5d (ver commits sugeridos en el reporte), re-ejecutar 01 y 01b → debe decir **COMPLETO**; `congelar_splits.py --dry-run`, luego real; commitear `data/splits/`.
3. En la PC con disco: `empaquetar_subconjunto.py` y subir los `.tar` a Drive; completar `REPO_URL`/`DRIVE_DIR` de Colab.
4. Etapa 2: modelo base (solo con splits congelados).

**Riesgos abiertos**
- El número de `grupo_hash` es cosmético y se corre si los pares `misma_foto` crean grupos nuevos (el estado y la partición no dependen de él).
- En el CSV del manifiesto, `revision_estado` conserva la redacción de la 5c ("… sin completar por algún revisor …") para no cambiar el hash; el detalle por planilla y por pares está en el `.meta.json`.
- El test propuesto (303) viene casi todo de la carpeta *train* oficial: no comparable con el test oficial.

## 2026-10-08 — Sesión 5: cierre de datos (candidatos finales; el equipo congela)

**Hecho** (detalle y tablas en `docs/exploracion_sesion5.md`)
- Decisiones del equipo registradas en `decisiones.md` (12 clases, partición propia de PlantDoc 70/30 con exclusión previa, mosaic zona 20, Colab opción A, invariante 1 nuevo). `CLAUDE.md` actualizado
  (Contexto, invariante 1, Colab). `mapeo_clases.md`: fuente del "27 = 17 + 10" (ar5iv) y 29 grupos contradictorios.
- PlantVillage: `clase_eval` (12 clases); train/val/test = 11.968 / 2.207 / 2.170 (coincide). Test de que filtrar no cambia la partición de las demás.
- PlantDoc: `data/interim/manifiesto_plantdoc_depurado.csv` (+ `.meta.json`): 1.010 conservadas → dev 707 / test 303; 60 excl. contradictorias (29 grupos), 28 duplicadas, 0 por revisión, 2 fuera de alcance.
  Sin alertas (dev mín 37, test mín 16). Matriz de contradictorios, ejemplos (`docs/figuras_sesion5/`), 3 pares de 7–10 para revisión (`docs/bitacora/plantdoc_pares_7_10_revision.csv`).
- Scripts (NO ejecutados): `congelar_splits.py`, `empaquetar_subconjunto.py`, `verificar_paquete.py`; `notebooks/colab_arranque.ipynb` sin ejecutar. 43 tests pasan.
- `accesos_test.md`: 1 imagen de test oficial en una figura (ya vista); el test propuesto no se miró.

**Qué quedó a medias**
- **Revisión humana** de `Tomato leaf`/`Bell_pepper leaf`: la planilla está vacía → el candidato de PlantDoc es **PROVISIONAL** y `congelar_splits.py` se niega a congelar. Al completarla, re-ejecutar `01b` (solo cambian esas 2 clases).
- `data/splits/` vacío; `REPO_URL` y `DRIVE_DIR` del notebook de Colab sin completar; la rama de torch de `fijar_semillas` sin probar (torch no instalado, ~2,9 GB libres).
- Las citas del paper se leyeron con una herramienta de lectura: confirmar a mano.

**Próximos pasos**
1. Dos integrantes completan `docs/bitacora/revision_etiquetas_planilla.csv` (`mantener`/`descartar`; criterio 1/2/3 en `motivo`). Re-ejecutar `01b_depuracion_plantdoc.ipynb`.
2. Decidir: los 2–3 casos que a ojo no son duplicados y los 3 pares de 7–10 (ver "Decisiones que quedan" en `exploracion_sesion5.md`).
3. En la PC con disco: ejecutar notebooks 01 y 01b, `empaquetar_subconjunto.py` y subir los `.tar` a Drive; `congelar_splits.py --dry-run` y luego real; commitear `data/splits/`.
4. Actualizar la propuesta con la lista de cambios de `exploracion_sesion5.md` §5 (no se editó).
5. Etapa 2: modelo base (solo con splits congelados).

**Riesgos abiertos**
- El test propuesto de PlantDoc (303) viene casi todo de la carpeta train oficial: no comparable con el test oficial (advertir en el informe).
- Si la revisión humana descarta imágenes después de congelar, habría que re-congelar (contra el invariante): conviene cerrar la revisión antes.

## 2026-10-08 — Sesión 4: entorno, manifiesto de PlantVillage y chequeos de PlantDoc (sin modelos)

**Hecho** (detalle y tablas en `docs/exploracion_sesion4.md`; notebooks `01_preparacion_datos.ipynb` y `00c_chequeos_plantdoc.ipynb`)
- Entorno: `requirements.txt`, `pyproject.toml` (`pip install -e . --no-deps` hecho), `configs/paths.yaml` (`DATA_ROOT`, override por variable de entorno `DATA_ROOT`),
  `src/foliares/utils/{paths,seeds}.py`. Los manifiestos guardan rutas relativas a `DATA_ROOT`. 23 tests pasan (`python -m pytest`).
- Manifiesto PlantVillage (22.787 imágenes, 15 clases, `data/interim/manifiesto_plantvillage.csv`); par color/segmented completo salvo 1 par con nombre irregular.
- Mosaic: zona de descarte validada con simulación (10 → 4,39 %, 15 → 1,77 %, 20 → 0,84 %); por la regla fijada de antemano el candidato usa **zona 20**.
- Candidato de particiones PlantVillage en `data/interim/particion_candidato_plantvillage.csv` (train 14.648 / val 2.483 / test 2.446). **`data/splits/` sigue vacío.**
- PlantDoc: duplicados entre todas las clases (phash, 8 variantes), CSV `docs/bitacora/plantdoc_duplicados.csv` con acción propuesta (no ejecutada),
  `data/interim/manifiesto_plantdoc.csv` con columnas para ambas opciones de split, planilla y hojas de contacto de revisión de `Tomato leaf`/`Bell_pepper leaf`.
- Registrado en `accesos_test.md` (12 imágenes de test miradas en grillas + 1 hoja de contacto; 16 generadas para revisión humana).

**Hallazgo principal:** 12 de 102 imágenes de test de PlantDoc tienen gemelo en train; **6 con etiqueta contradictoria** (p. ej. mismo archivo en `train/Potato leaf early blight` y
`test/Potato leaf late blight`). 29 grupos de duplicados con etiquetas contradictorias (60 imágenes) en total.

**Qué quedó a medias / no verificado**
- La revisión humana de etiquetas (dos revisores, por separado) no se hizo: es de personas.
- La rama de torch de `fijar_semillas` no se probó (torch no instalado; disco local ~2,8 GB libres).
- El candidato de PlantVillage no se congeló ni se copió a `data/splits/` (lo hace el equipo).
- Colab ↔ repo sin decidir; los manifiestos de PlantDoc guardan nombre *local* (NTFS) para 101 archivos: en Linux hay que usar `nombre_original`/el mapeo.

**Próximos pasos**
1. El equipo decide con `docs/exploracion_sesion4.md` ("Opciones con su costo"): cuáles son las 12 clases; si los gemelos con etiqueta contradictoria cuentan como "defecto nuevo en el test";
   criterio de la zona de mosaic; qué hacer con val/test sin sesiones sin grupo; tope por sesión; Colab↔repo.
2. Dos integrantes completan `docs/bitacora/revision_etiquetas_planilla.csv` (hojas de contacto en `data/interim/revision_etiquetas/`, regenerables con el notebook 00c).
3. Con las decisiones: congelar `data/splits/` (PlantVillage desde el candidato; PlantDoc según la opción elegida) y recién ahí empezar el modelo base.

**Decisiones pendientes del equipo:** las del punto 1, más las que siguen de sesiones 2–3 (mapeo de ambigüedades, pimiento sí/no, split de PlantDoc, priorización por plazos).

**Riesgos abiertos**
- El criterio de la zona de mosaic es agregado; por tanda hay casos de 11 % (laboratorio). No se sabe a qué se parece mosaic (es un proxy).
- Disco local ~2,8 GB libres; `data/interim/` pesa ~17 MB más las hojas de contacto (~12 MB).

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
  mayúsculas/minúsculas) [5d: conteo por orden de chequeo; 2 de los 87 con `?` también son de ruta larga, así que la clasificación exclusiva del CSV es 85 / 10 / 6]. Rescatados por hash de blob sin alterar el árbol oficial; mapeo
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
