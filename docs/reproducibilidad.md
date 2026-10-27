# Reproducibilidad: qué se garantiza y qué no

Sesión 7 (2026-10-27). `fijar_semillas` ([seeds.py](../src/foliares/utils/seeds.py)) fija python, numpy y torch (y CUDA si hay);
con `determinista=True` pide además algoritmos deterministas. Lo que sigue separa lo **medido** (CPU de la PC personal con los
tests de `tests/test_semillas_torch.py`, y una GPU de Colab con la sonda de `scripts/colab_humo.py`) de lo **documentado por
PyTorch y sin verificar**.

## En CPU (medido)
Torch 2.14.1+cpu, PC personal.
- Misma semilla, mismo torch, misma máquina y **mismos hilos**: logits, pérdidas, gradientes y pesos tras 2 pasos de
  MobileNetV3-Small son **idénticos bit a bit**, en el mismo proceso y entre procesos distintos.
- `determinista=True` no cambia el resultado en CPU.
- Semilla distinta → resultado distinto.
- **Cambiar la cantidad de hilos cambia el resultado** (el orden de las reducciones): con 1 hilo vs 6 hilos la diferencia
  de pesos tras 2 pasos fue de ~1e-7 (redondeo de float32), no igualdad exacta. Por eso cada resultado registra los hilos.

## En GPU (medido en Colab)
Tesla T4, torch 2.11.0+cu130, CUDA 13.0, cuDNN 92700, Python 3.13.16, en dos sesiones del 2026-10-27 (commits `bdd9805` y
`161b553`). La sonda corre dos veces los mismos 2 pasos desde la misma semilla (lote de 32 imágenes de PlantVillage-train,
MobileNetV3-Small sin pesos de ImageNet, SGD) y compara:

| modo | igual bit a bit | máx. dif. de pesos | máx. dif. de pérdidas | avisos de torch |
|---|---|---|---|---|
| `determinista=False`, sesión de `bdd9805` | **no** | 1,0058e-07 | 0,0 | ninguno |
| `determinista=False`, sesión de `161b553` | **no** | 2,98e-08 | 2,38e-07 | ninguno |
| `determinista=True` (las dos sesiones) | **sí** | 0,0 | 0,0 | ninguno |

Sin `determinista`, la diferencia entre corridas es de redondeo (≲ 1e-7) y **su tamaño varía de una sesión a otra**; con
`determinista=True` fue exacta las dos veces.

**Reanudar desde un checkpoint en GPU** (fase 2 de la prueba de humo, sesión de `161b553`, `determinista=False`): `reanudacion_ok:
true`. Las 5 comprobaciones dieron verdadero (sha256 del checkpoint igual al de la fase 1, paso correcto, pesos cargados iguales a
los guardados, mismo lote, pérdidas dentro de la tolerancia de 1e-3) y las 3 pérdidas reanudadas fueron **idénticas** a las de la
corrida continua (diferencia 0,0). Aun así `igual_bit_a_bit` dio `false`: ese campo exige también que el hash de los pesos finales
coincida, y las pérdidas de los 3 pasos no dependen de los pesos que deja el último paso. Es lo esperable con `determinista=False`
(la sonda muestra diferencias de ~1e-8 a 1e-7 entre corridas) y no es un error de la reanudación (los pesos recién cargados son
idénticos a los guardados), pero **no se midió** cuánto difieren los pesos finales ni se probó la reanudación con
`determinista=True`.

Alcance de las mediciones: una GPU (T4), 2 a 3 pasos, lote de 32, sin AMP y sin DataLoader con workers. No prueban que una
corrida larga o otra GPU/versión de torch sea determinista: solo que, en este caso, `determinista=True` alcanzó para la igualdad
exacta de la sonda y que, sin él, la diferencia es de redondeo.

**Corrección a la primera versión de este documento.** Decía que `AdaptiveAvgPool2d` volvía no determinista el backward en CUDA
por los 9 bloques Squeeze-Excitation y el GAP del cabezal (inferido de la documentación de `use_deterministic_algorithms`, sin
medir). No se cumplió: con `output_size=1`, el único que usa MobileNetV3-Small, torch reduce la operación a un promedio (el
grafo de autograd muestra `MeanBackward1`, verificado en CPU) y la T4 no emitió avisos con `determinista=True`. La advertencia
de PyTorch sigue valiendo para `AdaptiveAvgPool2d` con `output_size > 1`, que acá no se usa.

## Lo que sigue sin garantía (documentado por PyTorch, no verificado acá)
- cuDNN elige algoritmos de convolución distintos según el hardware y `benchmark`; `cudnn.deterministic=True` y
  `benchmark=False` (lo que hace `determinista=True`) los fijan en una GPU dada, pero **otro modelo de GPU u otra versión de
  cuDNN puede dar otros resultados**. Colab asigna GPU distintas según la sesión (T4, L4, …).
- cuBLAS necesita `CUBLAS_WORKSPACE_CONFIG=:4096:8` **antes de su primer uso**; `fijar_semillas(determinista=True)` lo fija si
  todavía no se usó CUDA (`scripts/colab_humo.py` lo fija al arrancar). Si se llama después, no tiene efecto.
- Precisión mixta (AMP) y TF32 agregan diferencias; si se usan, registrarlo.
- DataLoader con workers: usar un `torch.Generator` con semilla explícita y `worker_init_fn` para que la aumentación sea reproducible.
- **Los resultados de CPU y GPU no son comparables bit a bit**, ni los de versiones distintas de torch: en la sesión 7 la PC
  usó torch 2.14.1+cpu y Colab torch 2.11.0+cu130 (consistente con la decisión de dejar cotas mínimas en `requirements.txt` y
  registrar las versiones en cada JSON).

## Cómo se documenta en los resultados
- Cada corrida escribe un JSON con semillas, `determinista`, versión de torch/CUDA/cuDNN, dispositivo y nombre de la GPU, hilos
  de CPU, commit, tiempo y memoria pico (ver `foliares.utils.sesion`).
- Para igualdad exacta entre corridas, usar `determinista=True` en la misma GPU y versión (costo en velocidad: no medido). Las
  comparaciones entre variantes **no** se apoyan en igualdad bit a bit: se reportan con varias semillas y con el intervalo
  bootstrap de siempre.
- Reanudar desde un checkpoint guarda también los estados de los generadores (python, numpy, torch, CUDA); en CPU la reanudación
  reproduce la corrida continua bit a bit (test). En GPU, con `determinista=False`, reproduce las pérdidas pero no
  necesariamente los pesos finales bit a bit (ver arriba).
