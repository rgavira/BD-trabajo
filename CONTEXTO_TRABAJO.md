# Contexto Tecnico - Pipeline Big Data para Deteccion de Letras Explicitas

## Idea central

Este trabajo no busca simplemente "el mejor modelo".
Busca responder una pregunta mas importante:

**hasta donde podemos llegar con machine learning distribuido real en un entorno Big Data CPU puro, y donde empieza la frontera del deep learning que ya exige otra infraestructura**

La historia tecnica correcta del proyecto es:

1. construir una baseline seria dentro de `Spark MLlib`
2. mejorar la representacion con `Spark NLP` sin perder distribucion real
3. intentar subir al deep learning y encontrar sus limites practicos
4. reformular ese limite como separacion entre entrenamiento e inferencia
5. dejar `TensorFlow + Horovod + GPU` como extension natural en un entorno real

---

## Punto de partida

Venimos de trabajos anteriores donde el problema se abordaba en entorno clasico:

- notebooks o scripts
- una sola maquina
- TensorFlow/Keras directo
- en algun caso con GPU
- dataset muestreado para que el experimento fuese manejable

Aqui cambiamos de plano:

- Airflow para orquestacion
- HDFS para persistencia distribuida
- Spark para ETL y modelado
- dos workers CPU
- ejecucion dentro de Docker

Ese ultimo punto es crucial:

**el entorno del proyecto es CPU puro**

Y eso condiciona totalmente hasta donde tiene sentido llevar el deep learning.

---

## Pregunta de investigacion

La pregunta central del trabajo puede formularse asi:

1. Se puede reproducir una baseline razonable de deteccion de contenido explicito en un stack Big Data.
2. Que partes del pipeline se benefician de Spark de verdad.
3. Donde deja de tener sentido forzar Spark para deep learning en CPU.
4. Que infraestructura haria falta para cruzar esa frontera con honestidad.

---

## Dataset y ETL

- Origen: `spotify_dataset.csv`
- Tras limpieza: alrededor de `496k` canciones validas
- Desbalanceo: aproximadamente `72%` no explicitas y `28%` explicitas
- Split final: `train/test` 80/20 con `seed=42`

Persistencias:

- `hdfs:///data/raw/` para el dataset crudo
- `hdfs:///data/processed/` para `text` + `label` ya listos en `train/` y `test/`

El ETL crea una base comun y desacopla preparacion y entrenamiento.

---

## Como se ejecuta realmente

### Airflow

Airflow orquesta y lanza tareas.
No convierte por si solo nada en distribuido.

### Driver y workers

Los jobs Spark se lanzan con `--deploy-mode client`.

Eso implica:

- el `driver` vive en el contenedor `airflow`
- los `executors` corren en `spark-worker-1` y `spark-worker-2`

### Regla mental util

- si el job usa `Spark DataFrame` y `fit/transform` de Spark, el trabajo se reparte
- si el job hace `toPandas()` o trae tensores/arrays al proceso Python, el trabajo pasa al driver

Esta distincion es la clave de todo el trabajo.

---

## Nivel 0 - ETL y persistencia

Archivo principal: [02_etl.py](spark-jobs/02_etl.py)

Secuencia:

1. `ingesta_csv_pandas`
2. `spark_etl`

Objetivo:

- preparar el dato
- persistir en HDFS
- dejar una entrada comun para todos los experimentos

Todavia no se compara modelado.

---

## Nivel 1 - Spark MLlib como baseline distribuida real

Archivo principal: [03_word2vec_mlp.py](spark-jobs/03_word2vec_mlp.py)

Pipeline:

```text
text -> Tokenizer -> StopWordsRemover -> Word2Vec -> MLP
```

Este nivel es el corazon del trabajo.

Por que:

- usa solo componentes nativos de Spark
- entrena realmente distribuido
- aprovecha HDFS y workers
- encaja con la infraestructura CPU disponible

Lectura conceptual:

- `Word2Vec` genera un embedding promedio del documento
- eso hace el papel de `GlobalAveragePooling`
- el `MLP` cierra la clasificacion

Lectura arquitectonica:

- aqui si estamos haciendo **machine learning distribuido de verdad**

Esta es la pieza central que el proyecto defiende.

---

## Nivel 2 - Spark NLP

Spark NLP entra en el proyecto por una razon concreta:

- enriquecer la representacion del texto

No porque garantice por si mismo entrenamiento deep learning distribuido real.

Hay dos subniveles relevantes.

---

### Nivel 2a - Spark NLP embeddings + MLlib MLP

Archivo principal: [04_spark_nlp_use_mllib_mlp.py](spark-jobs/04_spark_nlp_use_mllib_mlp.py)

Pipeline:

```text
text -> DocumentAssembler -> UniversalSentenceEncoder -> Parquet intermedio -> MLP MLlib
```

Que aporta:

- embeddings de frase preentrenados
- mejor semantica que el promedio de tokens
- clasificador final todavia en `MLlib`

Por que el nivel funciona bien:

- la mejora viene de la representacion
- el entrenamiento sigue en `MLlib`
- por tanto sigue siendo distribuido de verdad

El Parquet intermedio actua como cortafuegos entre el lineage de Spark NLP y la parte MLlib.

Lectura correcta:

- `Spark NLP` si aporta valor fuerte en la **representacion textual**
- `MLlib` sigue siendo la pieza que garantiza el entrenamiento distribuido real

---

### Nivel 2b - La barrera del deep learning en CPU y el giro hacia inferencia

Archivo activo: [05b_distilbart_zeroshot.py](spark-jobs/05b_distilbart_zeroshot.py)

Archivo historico: [05_spark_nlp_classifierdl.py](spark-jobs/05_spark_nlp_classifierdl.py)

Este es el nivel donde mas se aprende.

#### Intencion original

La idea inicial era usar `ClassifierDL` de Spark NLP para entrenar un clasificador profundo dentro de Spark.

Eso habria parecido el paso natural:

- mejores embeddings
- clasificador deep learning
- mismo ecosistema Spark

#### Lo que paso en realidad

Aparecieron varios problemas practicos:

- `java.io.EOFException at KryoDeserializationStream`
- problemas de serializacion `Kryo`
- buffers grandes necesarios para objetos `Annotation`
- texto completo de canciones viajando en estructuras pesadas
- coste alto en CPU
- ejecucion poco estable dentro de Docker

Pero el punto mas importante fue este:

`ClassifierDLApproach.fit()` llama internamente a `Dataset.collect()`

Eso significa:

- los embeddings se llevan al `driver`
- se construyen tensores localmente
- el entrenamiento deja de estar distribuido
- los workers quedan practicamente ociosos durante el `fit`

En consecuencia:

**la distribucion era aparente, no real**

#### Donde echamos claramente de menos GPU

Este choque deja algo muy visible:

- un modelo deep learning tiene mas sentido cuando hay GPU
- aqui no la hay
- el cluster es CPU puro y corre en Docker

Por eso el problema no era solo de Spark NLP.
Era tambien de infraestructura.

Sin GPU:

- entrenar DL es costoso
- el rendimiento no compensa
- no estamos explotando hardware especializado

#### La reformulacion: separar entrenamiento e inferencia

Ese limite lleva a una decision de diseno muy importante:

**desacoplar entrenamiento e inferencia**

En produccion real esto es normal:

- el entrenamiento se hace una vez, con los recursos adecuados
- la inferencia se despliega y escala sobre nuevos datos

Aplicado aqui:

- dejamos de insistir en entrenar DL dentro de este entorno
- usamos un modelo ya preentrenado del hub de Spark NLP
- distribuimos solo la parte que si tiene sentido repartir: la inferencia

#### Implementacion activa

Pipeline:

```text
text -> DocumentAssembler -> Tokenizer -> BartForZeroShotClassification
```

Modelo:

- `distilbart_mnli_12_3`
- etiquetas candidatas: `["explicit", "clean"]`

Lectura correcta:

- no esta fine-tuneado sobre nuestro dataset
- no compite como modelo entrenado especificamente para Spotify
- ilustra un patron arquitectonico valido en Big Data: usar modelos preentrenados y repartir la inferencia

#### Incidencia tecnica adicional

El primer intento fue con `RoBertaForZeroShotClassification` y `roberta_classifier_large_mnli`.

Fallo con `ClassCastException` porque:

- ese artefacto estaba registrado como `RoBertaForSequenceClassification`
- no como `RoBertaForZeroShotClassification`

La solucion practica fue `BartForZeroShotClassification`.

#### Que significa este nivel para la memoria

Este nivel no demuestra entrenamiento deep learning distribuido.
Demuestra tres cosas mas valiosas:

1. el limite real de `ClassifierDL` en este entorno
2. la necesidad de GPU para entrenar DL con sentido
3. la aparicion del desacoplamiento entrenamiento-inferencia como solucion honesta

---

## Deep learning y limite de infraestructura

Este es probablemente el mensaje mas importante del proyecto:

**el problema no ha sido solo de software; ha sido de infraestructura**

Intentar hacer deep learning a gran escala aqui significa trabajar con:

- CPU pura
- Docker
- sin GPUs reales
- sin una topologia pensada para entrenamiento distribuido

Por eso el cierre natural del trabajo no esta en "forzar TensorFlow".
Esta en reconocer que:

- `MLlib` encaja con la infraestructura disponible
- `Spark NLP` ayuda en embeddings o inferencia
- el entrenamiento DL serio ya pide otro entorno

---

## TensorFlow como frontera cercana

Este nivel existe como frontera tecnica, no como centro del cierre practico.

Que representa:

- Spark sirve los datos
- `TensorFlow` entrena en el driver
- el `fit` no se paraleliza entre workers

Aunque el driver tuviera una GPU buena, seguiria existiendo un problema:

- el entrenamiento seguiria concentrado en un unico nodo si no anades otra capa de distribucion

Por eso este nivel sirve sobre todo para decir:

- "si quiero deep learning clasico, Spark ya no me basta como motor de entrenamiento"

---

## Horovod como frontera real del siguiente paso

Si existiera una infraestructura real con:

- driver con buena GPU
- workers con GPU
- entorno pensado para entrenamiento distribuido

entonces el siguiente paso natural seria:

- `TensorFlow` para el modelo
- `Horovod` para paralelizar entre distintas GPUs o nodos

Que aportaria `Horovod`:

- data parallelism real
- sincronizacion de gradientes
- entrenamiento distribuido de TensorFlow en serio

Por que aqui queda como extension teorica:

- no tenemos ese hardware
- no tenemos ese despliegue
- se sale del foco practico del trabajo actual

Por tanto:

- `TensorFlow` en driver marca la frontera inmediata
- `Horovod + TensorFlow + GPU` marca la frontera real del siguiente proyecto

---

## DAG activo y lectura recomendada

El DAG actual encadena:

1. `ingesta_csv_pandas`
2. `spark_etl`
3. `spark_word2vec_mlp`
4. `spark_use_mllib_mlp`
5. `spark_distilbart_zeroshot`

Pero la lectura recomendada para el cierre del trabajo es:

- el bloque principal termina en `03` y `04`
- `05b` funciona como evidencia del desacoplamiento entrenamiento-inferencia
- `TensorFlow + Horovod` quedan como frontera y trabajo futuro, fuera del codigo activo

---

## Resumen de niveles

| Nivel | Script | Tipo real de ejecucion | Papel en la memoria |
|---|---|---|---|
| 0 | `02_etl.py` | Spark distribuido | base comun reproducible |
| 1 | `03_word2vec_mlp.py` | distribuido real | baseline central del proyecto |
| 2a | `04_spark_nlp_use_mllib_mlp.py` | distribuido real | mejora semantica manteniendo ML distribuido |
| 2b | `05b_distilbart_zeroshot.py` | inferencia zero-shot | muestra el desacoplamiento entrenamiento-inferencia |
| 3 teorico | Horovod | distribuido real con GPU | extension natural en infraestructura real |

---

## Que estamos aprendiendo realmente

La aportacion del trabajo no es solo una tabla de metricas.

Estamos aprendiendo a distinguir entre:

- distribuido real
- distribuido aparente
- mejora de representacion
- entrenamiento local disfrazado de job Spark
- necesidad de infraestructura especializada

La conclusion fuerte es:

- `Spark MLlib` si ha permitido machine learning distribuido real
- `Spark NLP` ha sido valioso sobre todo en embeddings e inferencia
- el deep learning entrenado de verdad no era viable aqui sin GPU

---

## Cierre final

La lectura final que deja el proyecto es esta:

- hemos llevado el problema a un pipeline Big Data reproducible
- hemos demostrado una baseline distribuida seria
- hemos mejorado semanticamente la representacion del texto
- hemos encontrado, no supuesto, el limite del deep learning en CPU pura
- hemos formulado una salida arquitectonica valida: separar entrenamiento e inferencia
- hemos identificado con claridad la infraestructura necesaria para el siguiente salto: `GPU + TensorFlow + Horovod`

El problema no era falta de idea.
Era falta de infraestructura real para ese siguiente nivel.

Y precisamente por eso el aprendizaje tecnico del trabajo es muy fuerte.
