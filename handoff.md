# Handoff

> Se actualiza al cierre de cada sesión de trabajo. Lo más reciente arriba.

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
