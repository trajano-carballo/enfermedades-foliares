# LICENCIATURA EN INGENIERÍA DE DATOS E INTELIGENCIA UNIVERSIDAD TECNOLÓGICA (UTEC) ARTIFICIAL ITRN

# Clasificador de enfermedades foliares e investigación de degradación de métricas

César Trajano Carballo Asis - 55188291 Juan Lucas Pimentel Barreto - 54663222 Victor Samuel Uría Padilla - 54965630

Prof:

Juan Pedro de León Sum

17/09/2026


## 1. Introducción

La literatura sobre clasificación de enfermedades foliares reporta accuracies cercanas al 99%. Esas cifras se obtienen entrenando y evaluando sobre imágenes de laboratorio: hoja arrancada, fondo uniforme, iluminación controlada. Mohanty, Hughes y Salathé (2016) documentaron que un modelo con 99,35% de accuracy en esas condiciones cae a 31,4% al enfrentar imágenes tomadas en otras.

Este trabajo toma esa brecha como objeto de estudio. Entrenamos un clasificador sobre PlantVillage, lo evaluamos sobre PlantDoc —fotos de campo que el modelo nunca vio durante el entrenamiento— y medimos cuánto pierde, por qué lo pierde y cuánto se recupera con dos intervenciones acotadas.

*Ilustración 1 – Caída de accuracy en laboratorio vs campo*

Fuente: Elaboración propia

El resultado que este proyecto se propone reproducir y explicar. Mohanty, Hughes y Salathé (2016) entrenaron una red convolucional sobre PlantVillage y obtuvieron 99,35% de accuracy en un conjunto de prueba retenido. Los mismos autores reportan que, al evaluar sobre imágenes tomadas en condiciones distintas a las de entrenamiento, la accuracy se reduce a 31,4%. Sigue siendo muy superior al 2,6% que daría elegir al azar entre 38 clases, y esa doble lectura es parte de lo que hay que explicar.

## Problema

La detección temprana de enfermedades foliares decide si una intervención llega a tiempo. La respuesta obvia es una aplicación que clasifique la enfermedad a partir de una foto, y la literatura reporta accuracies cercanas al 99% para esa tarea.

El problema es que esas cifras se obtienen entrenando y evaluando sobre el mismo tipo de imagen: hoja arrancada, fondo uniforme, iluminación controlada. Cuando el modelo recibe una foto tomada en condiciones reales —hoja en la planta, sol directo o sombra, tierra y malezas de fondo, encuadre torcido— el rendimiento se desploma.


## Pregunta del proyecto

¿Cuánto rendimiento pierde un clasificador de enfermedades foliares al pasar de imágenes de laboratorio a imágenes de campo, qué está mirando el modelo cuando se equivoca, y cuánto de esa brecha se recupera con dos intervenciones acotadas: aumentación dirigida y supresión del fondo?

El entregable no es solo un modelo. Es un modelo más una medición de su condición de despliegue: un sistema que, cuando no está en condiciones de opinar, lo dice.

## Por qué es relevante

El fenómeno que se estudia —un modelo que aprende una correlación espuria del conjunto de entrenamiento y colapsa al cambiar el dominio— es uno de los modos de falla más comunes y más caros de los sistemas de aprendizaje automático en producción. Estudiarlo con datasets públicos y bien documentados es una forma barata de aprender algo que en un despliegue real se paga caro.

## Origen de la pregunta

El equipo desarrolla en paralelo un sistema de control de calidad para el vivero de una empresa forestal. Al evaluar si convenía automatizar parte de ese control con visión por computadora, surgió la duda de cuánto vale realmente un modelo de visión fuera de sus datos de entrenamiento. Este proyecto responde esa pregunta con datos públicos, sin comprometer datos ni plazos de un tercero. La aplicación al caso forestal queda explícitamente como trabajo futuro, condicionada al resultado de esta medición.

## 2. Usuario o Beneficiario

| Actor | Qué hace con el sistema | Qué gana |
| --- | --- | --- |
| Productor o técnico | Sube una foto de una hoja | Una orientación inmediata y un |
| agrónomo | sospechosa y recibe una sugerencia | criterio para saber cuándo hace |
| Usuario directo | con nivel de confianza, o un «no | falta una opinión calificada. |
|   | puedo decidir». |   |
| Equipo técnico que | Lee el informe de la brecha entre | Un número concreto sobre cuánto |
| evalúa desplegar visión | dominios. | descontar a las métricas publicadas |
| por computadora |   | antes de invertir. |
| Beneficiario del hallazgo |   |   |
| El propio equipo | Primera experiencia en visión por | Criterio para juzgar sistemas de |
| Beneficiario formativo | computadora sobre un problema | visión, más allá de este caso. |
|   | acotado y medible. |   |

Fuente: Elaboración propia

## Límite declarado del alcance

El sistema se presenta como asistente de orientación, nunca como diagnóstico. La interfaz lo declara de forma visible y la evaluación asume que siempre hay una persona en el circuito.


## 3. Datos

El diseño experimental se apoya en usar dos datasets públicos que difieren exactamente en la variable de interés: las condiciones de captura. Uno se usa para entrenar, el otro solo para evaluar.

| Conjunto |   | Rol | Contenido | Acceso |
| --- | --- | --- | --- | --- |
| PlantVillage |   | Entrenamient | 54.306 imágenes RGB de hojas | Kaggle |
| Hughes | y | o y evaluación | sanas e infectadas, 14 especies de | (emmarex/plantdisease), |
| Salathé, 2015 |   | in-distribution | cultivo, 38 clases. Hoja individual, | IEEE DataPort y el |
|   |   |   | fondo uniforme, iluminación | repositorio original en |
|   |   |   | controlada. Hay versiones color, | GitHub. Uso académico. |
|   |   |   | segmentada y escala de grises. |   |
| PlantDoc |   | Evaluación | 2.598 imágenes RGB de 13 | Repositorio |
| Singh et al., |   | out-of-distribu | especies y 17 clases de | pratikkayal/PlantDoc-Data |
| 2020 |   | tion | enfermedad, que con las clases | set en GitHub y espejo en |
|   |   |   | sanas dan 27 categorías. Entornos | Kaggle. Uso académico, sin |
|   |   |   | naturales: iluminación variable, | redistribución. |
|   |   |   | fondos complejos, oclusiones. El |   |
|   |   |   | conjunto de prueba tiene entre 8 y |   |
|   |   |   | 12 imágenes por clase. |   |

## Variables relevantes

- Imagen RGB redimensionada a 224×224 y normalizada con las estadísticas de ImageNet, para ser compatible con los backbones preentrenados.

- Etiqueta de clase planta–enfermedad, en el subconjunto de clases común a ambos datasets.

- Dominio de origen (laboratorio o campo). Es la variable central del experimento: no se usa como feature, se usa como criterio de partición y de reporte.

## Preparación necesaria

Los dos datasets no son compatibles tal como vienen, y esa incompatibilidad es una decisión de diseño que hay que justificar:

- Mapeo a un subconjunto común de clases. Las taxonomías no coinciden uno a uno; la intersección más rica está en tomate y maíz. Se documenta el mapeo, las clases descartadas y por qué.

- Detección automática de casi-duplicados entre conjuntos mediante hashing perceptual. PlantDoc se recolectó de internet y podría contener imágenes también presentes en PlantVillage, que contaminarían la evaluación.


- Control automático de integridad: archivos ilegibles, imágenes en escala de grises dentro del conjunto color, resoluciones anómalas y clases vacías tras el mapeo. Se resuelve con un script que reporta conteos, sin inspección manual exhaustiva.

## Datos que se evitan deliberadamente

- Scraping de imágenes de internet para ampliar el entrenamiento. Aumentaría el volumen, pero introduce etiquetas de calidad desconocida y derechos de uso poco claros.

- La versión segmentada de PlantVillage como entrenamiento base. Viene con el fondo removido, lo que enmascararía justamente el efecto que se quiere medir. Se reserva para la intervención de la etapa 4.

- Imágenes con personas, matrículas o datos identificables. El proyecto trabaja solo con los dos conjuntos públicos, lo que elimina la exposición a datos personales.

Ninguno de los dos datasets se construyó en Uruguay ni con las patologías prevalentes localmente, y ninguno incluye especies forestales. El proyecto es una demostración metodológica sobre un subconjunto de clases hortícolas, y así se enuncia en el documento final.

## Limitación de origen asumida desde el inicio


## 4. Arquitectura inicial

El sistema tiene dos circuitos. El superior es el de producción: de la foto a la sugerencia. El inferior es el de evaluación, y es donde vive el aporte del proyecto: el mismo modelo se mide contra dos dominios distintos y la diferencia entre ambas mediciones es el resultado principal.

## Ilustración 2 – Diagrama de arquitectura

Fuente: elaboración propia

## Cómo leer el diagrama

Azul: condiciones de laboratorio. Ocre: condiciones de campo. Rojo: resultado principal. Línea punteada: componente opcional o lazo de realimentación.

- PlantVillage entra únicamente por la rama de entrenamiento. PlantDoc nunca toca el entrenamiento. Si un solo lote de campo se filtrara, la medición de la brecha perdería sentido.

- El circuito de evaluación se bifurca después del modelo, no antes. Es el mismo modelo, con los mismos pesos y el mismo preproceso, el que se mide contra los dos dominios. Cualquier diferencia entre las ramas es atribuible al dominio y no al procedimiento.

- El umbral de abstención se fija con el resultado del circuito de evaluación, no antes. Es el punto donde el hallazgo se convierte en una decisión de producto: si la brecha es grande, el sistema se vuelve más conservador.


## 5. Modelos candidatos

Los dos candidatos comparten backbone y difieren en qué se entrena y con qué datos aumentados. Esa simetría es deliberada: hace que la comparación sea atribuible a la

intervención y no a un cambio de arquitectura.

## Referencia previa: clasificador trivial

Antes de los dos candidatos se reporta el desempeño de predecir siempre la clase mayoritaria. Deja escrito el piso y evita que un macro-F1 mediocre parezca un logro.

## Modelo base: backbone congelado más cabezal lineal

MobileNetV3-Small preentrenada en ImageNet, con los pesos convolucionales congelados y una única capa densa entrenada sobre las características extraídas.

- Entrena en pocos minutos en Colab gratuito, lo que permite iterar sobre el diseño experimental en lugar de esperar épocas.

- Al congelar el backbone se reduce la cantidad de parámetros entrenables, lo que limita el sobreajuste con un conjunto visualmente muy homogéneo.

- MobileNetV3 fue diseñada para inferencia en dispositivos móviles, lo que coincide con el escenario de uso y permite reportar tamaño y latencia como métricas reales.

- Da un piso limpio contra el cual medir si las intervenciones aportan algo.

## Modelo alternativo: fine-tuning parcial con aumentación dirigida

El mismo backbone, con los últimos bloques convolucionales descongelados y tasa de aprendizaje diferenciada, entrenado con una política de aumentación diseñada contra la hipótesis del proyecto: recortes aleatorios, variación de color y brillo, desenfoque, rotaciones y, en una variante, supresión del fondo usando la versión segmentada de PlantVillage.

- La hipótesis a testear es concreta: el modelo se apoya en el fondo uniforme y en la iluminación constante del laboratorio en lugar de la lesión. La aumentación de color y luz ataca la segunda parte; la supresión de fondo ataca la primera. Cada intervención se evalúa por separado.

- Descongelar las últimas capas permite reajustar las representaciones de alto nivel, que son las más específicas del dominio de entrenamiento y las más responsables de la caída.

- El costo adicional en GPU es acotado y medible, lo que convierte la comparación en una decisión de ingeniería reportable.

Los dos modelos no compiten por accuracy en PlantVillage. Compiten por brecha entre dominios. Un modelo que baja tres puntos en laboratorio pero sube doce en campo gana.

## Criterio de comparación


## Variante de contingencia

Si el fine-tuning no aporta mejora fuera de dominio, se prueba un backbone alternativo (ResNet-18 o EfficientNet-B0) bajo el mismo protocolo, para separar el efecto de la arquitectura del efecto de los datos. Si tampoco aporta, ese es el resultado y se reporta como tal.

## 6. Métricas

## Métricas de ML

- Matriz de confusión por dominio. Es la métrica de diagnóstico central: no dice solo cuánto cae el modelo, sino cómo cae. Interesa distinguir tres patrones: confusión entre enfermedades de la misma especie, confusión entre especies, y colapso hacia una clase dominante. Un macro-F1 de 0,30 producido por confusiones vecinas y uno producido por colapso son fallas distintas y exigen intervenciones distintas.

- Macro-F1 como métrica resumen, calculada por separado en cada dominio. Se prefiere sobre accuracy porque el subconjunto común queda desbalanceado tras el mapeo.

- Brecha ID→OOD: la diferencia de macro-F1 entre el test de PlantVillage y PlantDoc. Es el número que resume el proyecto.

- Recall por clase, leído junto con la matriz de confusión.

- Intervalos de confianza por bootstrap sobre las métricas OOD. Con 8 a 12 imágenes por clase, una diferencia de cinco puntos puede ser ruido.

- Error de calibración esperado y diagrama de fiabilidad, antes y después del ajuste de temperatura (Guo et al., 2017). Si la app muestra un número de confianza, ese número tiene que significar algo.

- Grad-CAM (Selvaraju et al., 2017) como evidencia cualitativa de dónde mira el modelo.

## Métricas de aplicación

- Cobertura útil: porcentaje de imágenes de campo en las que el sistema responde manteniendo una precisión igual o superior a 0,80. Es la métrica que decide si la herramienta sirve: un sistema que responde en el 35% de los casos y acierta puede ser más valioso que uno que responde siempre y falla la mitad de las veces.

- Tasa de abstención y su reverso operativo: cuántas consultas quedan derivadas a una persona.

- Latencia de inferencia medida en CPU, de la carga de la imagen a la respuesta.

- Tamaño del modelo en MB, que determina si es distribuible en un dispositivo con conectividad intermitente.


- Errores de alta confianza: proporción de predicciones incorrectas emitidas por encima del umbral de abstención. Es el error caro y se cuenta aparte.

## Métrica que no se reporta como titular

La accuracy sobre el test de PlantVillage. Va a dar alta, no informa sobre nada relevante para el usuario, y presentarla como resultado sería el maquillaje que la consigna pide evitar. Aparece solo como término izquierdo de la resta que define la brecha.

## 7. Riesgos y limitaciones

| Riesgo | Por qué puede ocurrir | Mitigación |
| --- | --- | --- |
| Baja generalización | El entrenamiento es visualmente | No se mitiga, se mide. El riesgo |
| Riesgo principal | homogéneo y el despliegue no lo | real sería no medirlo y presentar |
|   | es. Es el fenómeno que el | solo la métrica de laboratorio. |
|   | proyecto estudia. |   |
| Atajo por fondo | El fondo uniforme correlaciona | Grad-CAM y un experimento de |
|   | con la clase en PlantVillage; la | control con fondos sustituidos. Si |
|   | red puede clasificar por textura | la accuracy no cae al cambiar el |
|   | de fondo sin mirar la lesión. | fondo, el modelo no dependía de |
|   |   | él. |
| Test OOD pequeño Entre 8 y 12 imágenes por clase |   | Intervalos de confianza bootstrap |
|   | hacen inestable cualquier | y prohibición explícita de concluir |
|   | estimación puntual. | sobre diferencias dentro del |
|   |   | intervalo. |
| Desbalance | de Tras el mapeo al subconjunto | Macro-F1 en lugar de accuracy, |
| clases | común, algunas clases quedan | ponderación de clases en la |
|   | con muchas menos imágenes. | pérdida y lectura de la matriz de |
|   |   | confusión. |
| Sobreajuste al test | Probar variantes hasta que el | Tests congelados desde la etapa 1. |
|   | número OOD mejore convierte el | Bitácora que registra cuántas |
|   | test en validación encubierta. | veces se tocó cada test; el número |
|   |   | se publica. |
| Contaminación | PlantDoc se recolectó de internet | Detección automática de |
| entre datasets | y podría contener imágenes | casi-duplicados por hashing |
|   | también presentes | en perceptual antes de fijar las |
|   | PlantVillage. | particiones. |
| Costo | El fine-tuning con aumentación | Presupuesto de cómputo fijado |
| computacional | fuerte consume GPU y Colab | por adelantado, checkpointing y |
|   | gratuito corta sesiones. | reporte de tiempo y memoria |
|   |   | junto a cada métrica. |
| Sesgo geográfico y | Los datasets no reflejan las | Declarado como limitación de |
| de especie | especies ni las patologías | alcance desde la propuesta. No se |


| Riesgo | Por qué puede ocurrir | Mitigación |
| --- | --- | --- |
|   | prevalentes en Uruguay, y no | promete transferencia a otro |
|   | incluyen especies forestales. | contexto. |
| Integración | Una demo en notebook no | App en Gradio con el modelo |
|   | evidencia el flujo completo que | servido desde disco, probada con |
|   | exige la defensa final. | imágenes externas a todos los |
|   |   | conjuntos. |
| Inexperiencia | en Es la primera experiencia del | Alcance deliberadamente acotado, |
| visión | por equipo en el área; puede haber | backbones preentrenados estándar |
| computadora | errores de protocolo no | y consulta temprana en la |
|   | evidentes. | instancia de devolución de la |
|   |   | propuesta. |

## 8. Plan de trabajo

Siete etapas hasta la defensa de noviembre. Las semanas son orientativas y se ajustan a las fechas que se confirmen en Moodle. Las etapas 1 a 3 sostienen la presentación de la propuesta; el resto llega a la entrega final.

| Etapa | Semanas Qué se hace |   | Evidencia |
| --- | --- | --- | --- |
| 0 | S1 | Cierre de alcance: especies y clases del | Repositorio con estructura y |
|   |   | subconjunto común, reparto de roles | bitácora iniciada. |
|   |   | con rotación, apertura de la bitácora |   |
|   |   | de decisiones y del repositorio. |   |
| 1 | S1–S2 | Adquisición de datos y controles | Notebook de preparación con |
|   |   | automáticos: descarga, verificación de | conteos y el mapeo |
|   |   | integridad, mapeo de clases, detección | justificado. |
|   |   | de casi-duplicados, definición y |   |
|   |   | congelamiento de las particiones. |   |
| 2 | S2–S3 Modelo base: backbone congelado más |   | Primer número |
|   |   | cabezal lineal, entrenado solo en | in-distribution y matriz de |
|   |   | PlantVillage. Medición en el test de | confusión. |
|   |   | PlantVillage. |   |
| 3 | S3–S4 | Primera medición de la brecha: | Resultado que se lleva a la |
|   |   | evaluación en PlantDoc, macro-F1 con | presentación de la propuesta. |
|   |   | intervalos bootstrap, matriz de |   |
|   |   | confusión comparada entre dominios, |   |
|   |   | Grad-CAM sobre una muestra. |   |
| 4 | S5–S7 | Intervenciones: aumentación dirigida | Tabla comparativa de |
|   |   | y supresión de fondo, evaluadas por | brechas por variante. |
|   |   | separado y en combinación, bajo |   |
|   |   | protocolo idéntico al del modelo base. |   |


| Etapa | Semanas Qué se hace |   | Evidencia |
| --- | --- | --- | --- |
| 5 | S7–S8 | Calibración y abstención: ajuste de | Curva de cobertura-precisión |
|   |   | temperatura en validación, curva de | y umbral justificado. |
|   |   | cobertura contra precisión, elección |   |
|   |   | del umbral operativo. |   |
| 6 | S8–S9 | Aplicación demostrable en Gradio: | App funcional probada con |
|   |   | carga de imagen, sugerencia con | imágenes externas a los |
|   |   | confianza, mapa de atención, | conjuntos. |
|   |   | abstención visible, aviso de alcance. |   |
| 7 | S9–S10 Documento final, ensayo de la |   | Documento, demo ensayada |
|   |   | demostración cronometrada y | y registro del simulacro. |
|   |   | simulacro de defensa con preguntas |   |
|   |   | cruzadas entre integrantes. |   |


## Registro de uso de inteligencia artificial

Durante la elaboración de esta propuesta se utilizó Claude (Anthropic) como herramienta de

apoyo, en las siguientes instancias:

- Exploración y comparación de alternativas de proyecto en distintos dominios, previo a la elección del tema.

- Localización y contraste de las cifras citadas de PlantVillage, PlantDoc y Mohanty et al. (2016) contra sus fuentes primarias.

- Organización de la estructura del documento y redacción de versiones preliminares de los apartados.

- Elaboración del diagrama de arquitectura y de las figuras.

La definición del problema, la selección de los conjuntos de datos, el diseño experimental y las decisiones metodológicas fueron tomadas por el equipo. Todas las cifras y referencias incluidas fueron verificadas por los autores contra las publicaciones originales. El equipo

asume la responsabilidad por el contenido final del documento.


## Referencias

Anthropic. (2026). Claude (Opus 5) [Modelo de lenguaje grande]. https://claude.ai Guo, C., Pleiss, G., Sun, Y. y Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML. Hughes, D. P. y Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. Dataset PlantVillage. Mohanty, S. P., Hughes, D. P. y Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science, 7:1419. Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S. y Batra, N. (2020). PlantDoc: A Dataset for Visual Plant Disease Detection. CoDS-COMAD. arXiv:1911.10317. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. y Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization.

Proceedings of the IEEE International Conference on Computer Vision, 618–626.
