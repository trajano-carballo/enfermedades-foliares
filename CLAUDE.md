# CLAUDE.md — instrucciones para Claude Code

## Forma de trabajo
- **El equipo decide, Claude Code ejecuta.** Ante cualquier decisión metodológica
  (clases, splits, hiperparámetros, métricas, qué recortar por plazos) proponé opciones con
  su costo y esperá la decisión. No elijas por tu cuenta.
- Al inicio de cada sesión leé `handoff.md`. Al final, actualizalo (estado, qué quedó a
  medias, próximos pasos, decisiones pendientes).
- Cada decisión tomada va a `docs/bitacora/decisiones.md`.
- Git: no hagas `git add`, `commit` ni `push`. Al terminar una tarea, sugerí el mensaje de
  commit (Conventional Commits, en español).
- Idioma: español en docs, comentarios y mensajes. Nombres de código en inglés o español,
  pero consistentes dentro de cada módulo.

## Contexto
Propuesta completa en `docs/propuesta_entregable1.md`. Resumen: MobileNetV3-Small
(ImageNet) entrenada en PlantVillage, evaluada en PlantDoc; se mide la brecha ID→OOD en
macro-F1 y el efecto de aumentación dirigida y supresión de fondo; calibración por
temperatura y umbral de abstención; demo en Gradio.

Alcance de especies (decisión del equipo, sesión 2026-09-25): se excluye maíz del
subconjunto común. Base de trabajo: tomate y papa, evaluando si sumar pimiento según la
cobertura real en ambos datasets (ver `docs/mapeo_clases.md`). La partición dev/test de
PlantDoc no está decidida — se lleva evidencia numérica a la reunión con el docente, no se
resuelve unilateralmente.

## Invariantes (no romper nunca)
1. **PlantDoc no toca entrenamiento ni validación.** Si un cambio lo arriesga, detenete y avisá.
2. **Tests congelados.** `data/splits/` no se regenera tras la etapa 1. Cada evaluación sobre
   un test se agrega a `docs/bitacora/accesos_test.md` antes de reportar el número.
3. **Mismo preproceso y mismos pesos** para evaluar ambos dominios.
4. **Cabezal = GAP → Linear, sin capa oculta.** Es lo que habilita CAM (ver abajo). No
   reutilizar el `classifier` de torchvision (tiene Linear 576→1024 intermedia).
5. **Semillas fijas** y config en `configs/*.yaml`; cada resultado en `results/` debe
   poder rastrearse a una config y un commit.
6. Toda métrica OOD se reporta con intervalo bootstrap.
7. Datos y pesos no se versionan (ver `.gitignore`). Las particiones (CSV) sí.

## Explicabilidad: CAM (sin Grad-CAM)
Decisión del equipo (sesión 2026-09-25, a partir de la devolución docente sobre costo de
Grad-CAM para deploy): se usa **CAM únicamente**, en deploy y en el análisis offline.
Grad-CAM queda descartado del alcance, incluida la comparación de viabilidad.
- **CAM**: con cabezal GAP→Linear, `CAM_c = Σ_k w_ck · A_k` sobre el último mapa de
  features (576×7×7 en MobileNetV3-Small a 224 px). Sale del mismo forward, sin backward.
- No se implementa Grad-CAM en ninguna etapa del proyecto.

## Restricciones de cómputo
Colab gratuito: checkpointing obligatorio, registrar tiempo y memoria junto a cada métrica.
Latencia de la app se mide en CPU.

## Convenciones
- Código reutilizable en `src/foliares/`; notebooks solo orquestan y grafican.
- Notebooks numerados por etapa: `01_preparacion_datos.ipynb`, `02_modelo_base.ipynb`, …
- Scripts en `scripts/` con `argparse` y `--config`.
