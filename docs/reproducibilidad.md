# Reproducibilidad: qué se garantiza y qué no

Sesión 7 (2026-10-27). `fijar_semillas` ([seeds.py](../src/foliares/utils/seeds.py)) fija python, numpy y torch (y CUDA si hay);
con `determinista=True` pide además algoritmos deterministas. Lo que sigue separa lo **comprobado** (tests en
`tests/test_semillas_torch.py`, CPU de la PC personal, torch 2.14.1+cpu) de lo **documentado por PyTorch y sin verificar acá**
(no hay GPU en la PC; se comprueba en Colab con la celda de humo, ver `notebooks/colab_arranque.ipynb`).

## En CPU (comprobado)
- Misma semilla, mismo torch, misma máquina y **mismos hilos**: logits, pérdidas, gradientes y pesos tras 2 pasos de
  MobileNetV3-Small son **idénticos bit a bit**, en el mismo proceso y entre procesos distintos.
- `determinista=True` no cambia el resultado en CPU.
- Semilla distinta → resultado distinto.
- **Cambiar la cantidad de hilos cambia el resultado** (el orden de las reducciones): con 1 hilo vs 6 hilos la diferencia
  de pesos tras 2 pasos fue de ~1e-7 (redondeo de float32), no igualdad exacta. Por eso cada resultado registra los hilos.

## En GPU (documentado por PyTorch; sin verificar acá)
No se puede garantizar igualdad bit a bit entre corridas ni entre GPU distintas:
- `AdaptiveAvgPool2d` no tiene backward determinista en CUDA: con `use_deterministic_algorithms(True)` lanza error (con
  `warn_only=True`, que es lo que usamos, solo avisa y sigue siendo no determinista). MobileNetV3-Small tiene **9 bloques
  Squeeze-Excitation** que lo usan, además del GAP del cabezal, así que el backward en GPU no es determinista aunque se pida.
- cuDNN elige algoritmos de convolución distintos según el hardware y `benchmark`; `cudnn.deterministic=True` y
  `benchmark=False` (lo que hace `determinista=True`) reducen la variación pero no la eliminan entre modelos de GPU.
- cuBLAS necesita `CUBLAS_WORKSPACE_CONFIG=:4096:8` **antes de su primer uso**; `fijar_semillas(determinista=True)` lo fija si
  todavía no se usó CUDA. Si se llama después, no tiene efecto.
- Colab asigna GPU distintas según la sesión (T4, L4, …): el mismo código puede dar resultados distintos entre sesiones.
- Precisión mixta (AMP) y TF32 agregan diferencias; si se usan, registrarlo.
- DataLoader con workers: usar un `torch.Generator` con semilla explícita y `worker_init_fn` para que la aumentación sea reproducible.

## Cómo se documenta en los resultados
- Cada corrida escribe un JSON con semillas, `determinista`, versión de torch/CUDA/cuDNN, dispositivo y nombre de la GPU, hilos
  de CPU, commit, tiempo y memoria pico (ver `foliares.utils.sesion`).
- Las comparaciones entre variantes en GPU **no** se apoyan en igualdad bit a bit: se reportan con varias semillas y con el
  intervalo bootstrap de siempre. La semilla fija da repetibilidad aproximada, no exacta.
- Reanudar desde un checkpoint guarda también los estados de los generadores (python, numpy, torch, CUDA); en CPU la reanudación
  reproduce la corrida continua bit a bit (test), en GPU es aproximada por lo anterior.
