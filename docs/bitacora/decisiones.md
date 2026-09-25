# Bitácora de decisiones

Una entrada por decisión. La toma el equipo; Claude Code solo la registra.

| Fecha | Decisión | Alternativas descartadas | Motivo | Quién |
|---|---|---|---|---|
| 2026-09-24 | Estructura inicial del repositorio | — | Primer commit | Equipo |
| 2026-09-24 | Explicabilidad en deploy con CAM; Grad-CAM solo sobre una muestra y como comparación | Grad-CAM en todas las inferencias | Devolución docente: costo de gradientes y plazos | Equipo |
| 2026-09-25 | Explicabilidad: CAM únicamente, en deploy y en el análisis; se descarta Grad-CAM por completo (supera la decisión del 2026-09-24) | Grad-CAM offline sobre una muestra fija | El equipo decidió simplificar el alcance dado el costo/tiempo de mantener también Grad-CAM, aunque fuera limitado a una muestra | Equipo |
| 2026-09-25 | Excluir maíz del subconjunto común de clases | Incluir maíz junto con tomate y papa | Foco en las especies con mayor intersección real de clases entre PlantVillage y PlantDoc | Equipo |
| 2026-09-25 | Base de trabajo: tomate y papa; evaluar sumar pimiento según la cobertura real en ambos datasets | Solo tomate; tomate + maíz | Papa y pimiento amplían el subconjunto común; la inclusión de pimiento queda condicionada a los conteos del inventario (sesión 2, exploración) | Equipo |
| 2026-09-25 | Commits fijados para la descarga: PlantVillage `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a` (sparse-checkout `raw/color`, `raw/segmented`, `leaf_grouping`), PlantDoc `5467f6012d78d1c446145d5f582da6096f852ae8`. Tamaño en disco medido: PlantVillage 2,57 GB (clon completo), PlantDoc 2,10 GB (clon completo) | — | Ambos hashes verificados contra HEAD remoto y contra `git log -1` local. Detalle en `docs/exploracion_datasets.md` §0 y §7 | Equipo |
| 2026-09-25 | Exclusión de maíz — evidencia formal agregada (leaf_grouping): 0,00 % de cobertura en `leaf_grouping/leaf-map.json` oficial (0/3852 imágenes de color mapean a un leaf_id real), contra 100,00 % en papa y 99,92 % en pimiento. No es el motivo original de la exclusión, la complementa | — | Verificación formal solicitada para la bitácora, reproduce la lógica exacta de `plant_village.py::_generate_examples`. Detalle en `docs/exploracion_datasets.md` §3 | Equipo |
| 2026-09-25 | Mapeo de clases tomate/papa/pimiento — borrador con 5 ambigüedades marcadas, sin cerrar (ver `docs/mapeo_clases.md`) | — | Insumo para la reunión con el docente sobre la partición de PlantDoc; no se resuelve en esta sesión | Equipo |
