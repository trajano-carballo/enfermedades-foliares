# Clasificador de enfermedades foliares: brecha laboratorio → campo

Proyecto Integrador AIAI — Aplicaciones de Aprendizaje Automático (LIDIA, UTEC ITRN).
César Carballo · Juan Lucas Pimentel · Victor Uría — Prof. Juan Pedro de León.

MobileNetV3-Small (ImageNet) con cabezal GAP → Linear, entrenada en **PlantVillage** (laboratorio) y evaluada en **PlantDoc**
(campo), en 12 clases de tomate, papa y pimiento. El resultado principal es la **brecha de macro-F1** entre los dos dominios, con
IC bootstrap; después vienen una variante con fine-tuning parcial y aumentación, la calibración por temperatura con umbral de
abstención, y una app en Gradio con CAM. Informe: [`docs/informe_final.md`](docs/informe_final.md). Estado al día:
[`handoff.md`](handoff.md).

Este README explica cómo reproducir el proyecto, desde clonar el repo hasta levantar la app. Cada paso dice qué escribe y
dónde.

## 0. Lo que se reproduce sin datos

Todos los números y figuras del informe salen de JSON versionados en `results/`. Sin descargar nada:

```
pip install -r requirements.txt && pip install -e . --no-deps
python scripts/figuras_finales.py --config configs/07_figuras_finales.yaml   # docs/figuras_finales/ (figuras y tablas.md)
python -m pytest tests/test_trazabilidad.py tests/test_figuras_finales.py    # cada número del informe contra su fuente
```

`docs/trazabilidad.md` lleva cada número del informe a su archivo, config, commit y fila de acceso a test.

## 1. Entorno

- **Python ≥ 3.11.4** (probado con 3.13.7 en Windows y 3.13.16 en Colab).
- `pip install -r requirements.txt` y `pip install -e . --no-deps`. `requirements.txt` fija cotas mínimas; las versiones exactas
  de la PC de trabajo están en `docs/entorno_referencia.txt`, y cada JSON de `results/` registra torch, CUDA, GPU e hilos.
- **Sin GPU** alcanza para todo menos el entrenamiento de la variante (Colab, T4). El modelo base entrena en CPU en minutos.
- Tests: `python -m pytest -rs`. Con los datos y los pesos en su lugar pasa la suite completa. Sin ellos, algunos tests se saltan
  (`-rs` dice cuáles) y no se verificó qué pasa con el resto; sin datos, correr los de §0.
- Windows: activar rutas largas (`LongPathsEnabled=1`) para PlantDoc (§2).

Todas las rutas salen de `configs/paths.yaml`. Los datos van en `data/` (o en la carpeta que diga la variable `DATA_ROOT`). Los
pesos y las salidas pesadas van en una carpeta **fuera del repo**, que acá se llama `SALIDA` (por ejemplo `../foliares_salidas`):
los scripts se niegan a escribir pesos adentro del repo.

## 2. Datos

Ninguno de los dos conjuntos se redistribuye (uso académico): se clonan de sus repositorios, en el commit fijado.

```
# PlantVillage (solo color, segmented y leaf_grouping)
git clone --filter=blob:none --no-checkout https://github.com/spMohanty/PlantVillage-Dataset data/raw/plantvillage
git -C data/raw/plantvillage sparse-checkout set raw/color raw/segmented leaf_grouping
git -C data/raw/plantvillage checkout 7f7ecc7e1eaca78107e3affe7cb5abd9427e139a

# PlantDoc
git clone --no-checkout https://github.com/pratikkayal/PlantDoc-Dataset data/raw/plantdoc
git -C data/raw/plantdoc checkout 5467f6012d78d1c446145d5f582da6096f852ae8
python scripts/rescatar_archivos_plantdoc.py --rev 5467f6012d78d1c446145d5f582da6096f852ae8 --mover-originales
```

El rescate aplica el mapeo versionado (`docs/bitacora/plantdoc_archivos_renombrados.csv`, 101 archivos) para que los nombres
coincidan con los de las particiones en cualquier sistema, y verifica el sha1 de cada archivo contra el árbol de git. **En Windows**
el `checkout` de PlantDoc falla (nombres con `?` y colisiones de mayúsculas): seguir el procedimiento de tres pasos de
[`docs/exploracion_datasets.md`](docs/exploracion_datasets.md) §2 (checkout selectivo y después el rescate).

## 3. Particiones congeladas (no se regeneran)

`data/splits/` tiene las particiones de la etapa 1 (congeladas el 2026-10-23, commit `b79d3a2`). Nunca se regeneran
(invariante 2 de [`CLAUDE.md`](CLAUDE.md)). Para verificarlas:

```
cd data/splits && sha256sum -c SHA256SUMS.txt      # PowerShell: Get-FileHash <archivo> -Algorithm SHA256
```

Cada script que las lee registra su sha256 en el JSON de salida. Cómo se armaron (partición por hoja, depuración de PlantDoc,
revisión humana): `docs/exploracion_sesion5.md`, `data/splits/README_VERSION.md` y los notebooks `01_preparacion_datos.ipynb` y
`01b_depuracion_plantdoc.ipynb`.

## 4. Entrenamiento

**Modelo base** (CPU, unos 6 minutos: features GAP precalculadas y un `Linear(576, 12)`; solo PV-train y PV-val):

```
python scripts/entrenar.py --config configs/02_modelo_base.yaml --salida SALIDA
```

**Variante** (fine-tuning parcial y aumentación; Colab con GPU T4, unos 25 minutos). Usa el checkpoint del base:

1. Empaquetar el subconjunto de 12 clases (en la PC con los datos; antes, correr los notebooks `01` y `01b`, que escriben los
   manifiestos en `data/interim/`):
   `python scripts/empaquetar_subconjunto.py --config configs/paquete_subconjunto.yaml --salida <carpeta>`, y subir el `.tar`
   de PlantVillage y su `.sha256` a `foliares/paquetes/` en Drive. PlantDoc no va a Colab: no entrena ni elige nada.
2. Subir el checkpoint del base a `foliares/checkpoints/02_modelo_base/<run_id>/mejor_checkpoint.pt`.
3. Abrir `notebooks/04_aumentacion_colab.ipynb` en Colab y correr las celdas en orden: clona el repo, descomprime y verifica el
   paquete (`scripts/verificar_paquete.py`), y entrena con `scripts/entrenar_finetune.py --config configs/04_aumentacion.yaml`,
   con un checkpoint por época y reanudación si se corta la sesión.
4. Bajar `mejor.pt` y `resultados.json` a `SALIDA/04_aumentacion/<run_id>/`.

Cada corrida escribe su JSON completo en `SALIDA/<experimento>/<run_id>/` y copia los JSON chicos a `results/<experimento>/<run_id>/`.

**Sobre la igualdad exacta.** Con `determinista=True` y la misma máquina, el resultado se repite bit a bit. En otra GPU u otra
versión de torch puede diferir por redondeo (ver [`docs/reproducibilidad.md`](docs/reproducibilidad.md)). Las evaluaciones y la app
verifican el **sha256 del checkpoint** que fija cada config: con pesos reentrenados, el sha256 no va a coincidir y los scripts no
cargan el modelo. Para usarlos, hay que poner su `run_id`, ruta y sha256 en la config, lo que es una corrida nueva y no la del
informe. Los pesos del informe no se versionan ni se publican: **se piden al equipo** (los dos checkpoints van en
`SALIDA/02_modelo_base/<run_id>/` y `SALIDA/04_aumentacion/<run_id>/`, con los `run_id` y sha256 de las configs).

## 5. Evaluación

```
# PlantDoc-dev: compara las tres geometrías de preproceso y aplica la regla fijada (no toca ningún test)
python scripts/evaluar.py --config configs/03_brecha.yaml --salida SALIDA --fase dev
# PlantDoc-dev, variante (informativo, con la diferencia pareada contra el base)
python scripts/evaluar.py --config configs/04_evaluacion.yaml --salida SALIDA --fase dev_informativo
```

**Los tests ya se evaluaron y no se vuelven a evaluar.** `--fase tests` solo corre con una fila nueva de
[`docs/bitacora/accesos_test.md`](docs/bitacora/accesos_test.md), commiteada antes y no usada; la guarda rechaza las filas 1–5, que
ya están gastadas. Los números de los tests están en `results/` (`tests.json` de cada modelo), y lo que se deriva de ellos sale de las
predicciones guardadas, sin volver a pasar ningún test por un modelo:

```
python scripts/ic_por_clase.py --config configs/03_brecha.yaml --salida SALIDA --run-dev <run dev> --run-tests <run tests>
python scripts/comparar_con_base.py --config configs/04_evaluacion.yaml --salida SALIDA --run-tests <run tests de la variante>
```

Protocolo completo, con las reglas fijadas antes de cada número: [`docs/protocolo_evaluacion.md`](docs/protocolo_evaluacion.md).

## 6. Calibración

```
python scripts/calibrar.py --config configs/05_calibracion.yaml --salida SALIDA --fase dev
```

Elige el modelo por PV-val (regla 1), ajusta T en PlantDoc-dev con la grilla de la config y estima el ECE fuera de partición. La
fase `test` exige una fila de acceso, igual que en §5 (la del informe es la fila 5, ya usada). La T y τ que usa la app están en
`configs/05_calibracion.yaml` (`temperatura`, `tau`).

## 7. App

```
python app/app.py --config configs/06_app.yaml --salida SALIDA        # http://127.0.0.1:7860
```

Carga el checkpoint, T y τ de `configs/05_calibracion.yaml` y verifica el sha256. Muestra el top-3 calibrado, el CAM y, si la
probabilidad máxima calibrada es menor que τ, el cartel "Confianza baja". Con la T del informe, ese cartel aparece en todas las
fotos (ver el informe, §7–8). Para probarla, usar imágenes de PV-val o de PD-dev, nunca de un test (la lista está en
`results/06_app/06_app_latencia_20261110_150137/latencia.json`).

Mediciones de la app (no tocan ningún test):

```
python scripts/medir_pv_val.py --config configs/06_app.yaml --salida SALIDA        # cobertura informativa en PV-val
python scripts/verificar_externas.py --config configs/06_app.yaml --cache <carpeta>  # duplicados de las fotos de Commons
python scripts/medir_latencia.py --config configs/06_app.yaml --salida SALIDA      # latencia y tamaño, en CPU
```

Las fotos de Wikimedia Commons no se versionan: autor, licencia, URL y sha256 están en `docs/imagenes_externas.csv`.

## Estructura

```
configs/           un YAML por experimento (02 base, 03 evaluación, 04 variante, 05 calibración, 06 app, 07 figuras)
data/splits/       particiones congeladas (versionadas); data/raw, interim, processed no se versionan
src/foliares/      data/ (preproceso, particiones, duplicados), models/, training/, eval/ (métricas, bootstrap,
                   calibración, guarda de accesos, figuras), explain/ (CAM), inferencia.py (lo que usa la app)
scripts/           puntos de entrada con --config
notebooks/         numerados por etapa; solo orquestan y grafican
app/app.py         la app en Gradio
results/           JSON chicos de cada corrida (versionados); los pesos van en SALIDA
docs/              informe, protocolo, trazabilidad, arquitectura, figuras; bitacora/ (decisiones y accesos a test)
tests/             tests
```

## Reglas del experimento

1. PlantDoc no entra al entrenamiento ni a la selección del checkpoint. PlantDoc-dev elige geometría, temperatura y umbral;
   PlantDoc-test se evalúa una vez por modelo.
2. `data/splits/` no se regenera, y cada evaluación de un test se registra en `docs/bitacora/accesos_test.md` antes de calcular.
3. Mismo preproceso y mismos pesos para los dos dominios; cabezal GAP → Linear (habilita CAM); semillas fijas.
4. Toda métrica fuera de distribución lleva IC bootstrap, y toda decisión va a `docs/bitacora/decisiones.md`.
5. La exactitud en PlantVillage no se reporta como resultado titular.

## Datos y licencias

- PlantVillage (Hughes y Salathé, 2015): https://github.com/spMohanty/PlantVillage-Dataset
- PlantDoc (Singh et al., 2020): https://github.com/pratikkayal/PlantDoc-Dataset

Uso académico, sin redistribución: los datos y los pesos no se suben al repositorio.
