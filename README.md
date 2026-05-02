# Pipeline Big Data - Deteccion de Contenido Explicito en Canciones

## Vision general

Este proyecto traslada el problema de clasificacion de letras explicitas en Spotify a un entorno Big Data real, pero con un foco muy concreto:

- estudiar **machine learning distribuido de verdad**
- medir hasta donde llega Spark de forma nativa
- identificar con honestidad donde empieza la frontera del deep learning en un cluster CPU puro

La pregunta principal ya no es solo "que modelo saca mejor F1", sino:

- que parte del pipeline corre realmente distribuida
- que parte solo parece distribuida porque se lanza con `spark-submit`
- que ocurre cuando queremos pasar de ML distribuido a deep learning sin tener infraestructura real con GPU

La conclusion practica del trabajo es clara:

- `Spark MLlib` es el centro del proyecto porque si entrena distribuido de verdad
- `Spark NLP` aporta valor real cuando mejora la representacion textual sin salir del ecosistema Spark
- el salto a deep learning entrenado en este entorno choca con limites tecnicos e infraestructurales

---

## Tesis del proyecto

El trabajo se cierra con una idea fuerte:

**hemos podido construir una baseline seria y una mejora semantica real dentro de Spark, pero no tenemos la infraestructura adecuada para entrenar deep learning distribuido de forma honesta**

Por eso el arco narrativo queda asi:

1. `Spark MLlib` como baseline distribuida central.
2. `Spark NLP + embeddings` como mejora semantica manteniendo entrenamiento distribuido.
3. barrera practica al intentar entrenar deep learning en un cluster CPU puro.
4. separacion de `entrenamiento` e `inferencia` como salida arquitectonica razonable.
5. `TensorFlow + Horovod` como frontera teorica para un entorno real con GPU.

---

## Arquitectura general

```text
CSV (spotify_dataset.csv)
    |
    v
Task 1 - Airflow + pandas
Parquet crudo local (/opt/data/songs_raw.parquet)
    |
    v
Task 2 - Spark ETL distribuido
HDFS /data/raw/
HDFS /data/processed/
    |
    v
Task 3 - Spark MLlib
Word2Vec + MLP
    |
    v
Task 4 - Spark NLP embeddings + MLlib MLP
USE + MLP MLlib
    |
    v
Task 5 - Spark NLP zero-shot
modelo preentrenado + inferencia distribuida
    |
    v
Frontera teorica
TensorFlow en driver / Horovod + GPU
```

---

## Como se ejecuta realmente

### Airflow

Airflow orquesta tareas, lanza procesos y guarda logs.
No convierte por si solo un script en distribuido.

### Spark jobs

Los jobs se lanzan con `spark-submit` y `--deploy-mode client`.

Eso significa:

- el `driver` vive en el contenedor `airflow`
- los `executors` corren en `spark-worker-1` y `spark-worker-2`

### Regla mental util

- si el script mantiene datos en `Spark DataFrame` y usa `fit/transform` de Spark, el trabajo se reparte entre workers
- si el script hace `toPandas()` o trae datos al proceso Python, a partir de ahi el trabajo es local al driver

---

## Niveles del proyecto

### Nivel 0 - ETL y persistencia

Archivo principal: [02_etl.py](spark-jobs/02_etl.py)

Prepara la base comun del proyecto:

- lectura del Parquet crudo
- limpieza del texto
- normalizacion de etiquetas
- deduplicacion
- `train/test split`
- persistencia en HDFS

No responde a una pregunta de modelado, sino a una necesidad de infraestructura reproducible.

---

### Nivel 1 - Spark MLlib como baseline distribuida

Archivo principal: [03_word2vec_mlp.py](spark-jobs/03_word2vec_mlp.py)

Este es el nivel central del trabajo.

Pipeline:

```text
text -> Tokenizer -> StopWordsRemover -> Word2Vec -> MLP
```

Por que es el centro:

- esta completamente dentro de Spark
- entrena distribuido de verdad
- usa HDFS, driver y workers de forma natural
- reproduce una baseline razonable inspirada en AINE/MLE sin salir del ecosistema Big Data

Lectura correcta:

- aqui si estamos haciendo **machine learning distribuido real**
- aqui si estamos aprovechando el cluster con honestidad

---

### Nivel 2a - Spark NLP embeddings + MLlib MLP

Archivo principal: [04_spark_nlp_use_mllib_mlp.py](spark-jobs/04_spark_nlp_use_mllib_mlp.py)

Pipeline:

```text
text -> DocumentAssembler -> UniversalSentenceEncoder -> Parquet intermedio -> MLP MLlib
```

Que aporta:

- embeddings preentrenados de frase
- mejor representacion semantica que el promedio clasico de tokens
- clasificador final todavia en `MLlib`, por tanto el entrenamiento sigue siendo distribuido

Por que es importante:

- demuestra que `Spark NLP` si aporta valor en este proyecto
- pero ese valor aparece sobre todo en la **representacion**
- seguimos apoyandonos en `MLlib` para mantener el entrenamiento realmente distribuido

---

### Nivel 2b - Barrera del deep learning y separacion entrenamiento-inferencia

Archivo activo: [05b_distilbart_zeroshot.py](spark-jobs/05b_distilbart_zeroshot.py)

Archivo historico: [05_spark_nlp_classifierdl.py](spark-jobs/05_spark_nlp_classifierdl.py)

Este nivel nace del principal aprendizaje del proyecto.

#### Lo que intentamos

La idea original era usar `ClassifierDL` de Spark NLP para subir un escalon y entrenar un clasificador profundo dentro del propio ecosistema Spark.

#### Lo que encontramos

Al intentarlo aparecieron varios limites reales:

- `java.io.EOFException at KryoDeserializationStream`
- problemas de serializacion `Kryo` con objetos `Annotation`
- necesidad de buffers grandes de Kryo para manejar el lineage NLP
- coste computacional alto en CPU
- tiempos y consumo poco razonables dentro de Docker
- ausencia total de GPU, justo donde un modelo DL mas sentido tendria

Pero el hallazgo mas importante fue arquitectonico:

- `ClassifierDLApproach.fit()` llama internamente a `Dataset.collect()`
- los embeddings se llevan al `driver`
- el entrenamiento deja de estar distribuido
- los workers quedan ociosos durante el `fit`

En otras palabras:

**el deep learning dentro de Spark NLP no nos estaba dando entrenamiento distribuido real en este entorno**

#### Por que la CPU pura cambia todo

Nuestro cluster es CPU puro y corre en Docker.
Sin GPU:

- entrenar deep learning deja de ser atractivo
- el coste es alto para el retorno esperado
- no estamos aprovechando una infraestructura especializada

Por eso aqui aparece la idea de:

**separar entrenamiento e inferencia**

#### La solucion que si encaja

En lugar de insistir en entrenar DL en este entorno, usamos un modelo ya preentrenado del hub de Spark NLP y distribuimos solo la inferencia.

Pipeline activo:

```text
text -> DocumentAssembler -> Tokenizer -> BartForZeroShotClassification
```

Lectura correcta:

- no entrenamos sobre nuestros datos
- no hacemos fine-tuning
- no intentamos vender este nivel como mejor modelo entrenado
- lo usamos para ilustrar un patron real de produccion: **entrenar fuera, inferir dentro del cluster**

#### Nota tecnica

El primer intento fue con `RoBertaForZeroShotClassification` y `roberta_classifier_large_mnli`, pero aparecio un `ClassCastException`.
Ese modelo estaba registrado como `SequenceClassification`, no como `ZeroShotClassification`.
La alternativa funcional fue `BartForZeroShotClassification` con `distilbart_mnli_12_3`.

#### Conclusión de este nivel

Este nivel no demuestra entrenamiento distribuido de deep learning.
Demuestra algo mas honesto y mas util:

- donde esta el limite del entorno actual
- por que echamos de menos GPU
- por que entrenamiento e inferencia deben desacoplarse

---

## Lo que NO estamos vendiendo

No estamos diciendo que el deep learning no valga.
Estamos diciendo algo mas preciso:

- en este entorno concreto, **CPU puro + Docker + sin GPUs**, entrenar deep learning a escala no tiene sentido practico
- `Spark MLlib` si encaja con la infraestructura disponible
- `Spark NLP` es util para enriquecer representaciones o para servir modelos preentrenados
- pero el entrenamiento DL serio queda fuera de las capacidades reales del laboratorio

---

## TensorFlow y Horovod como frontera

El siguiente paso natural seria:

1. usar `TensorFlow` con GPU
2. evitar que el entrenamiento quede atrapado en el driver
3. paralelizar entre varias GPUs o nodos

Y ahi aparece `Horovod`.

### Que significaria un entorno real

Si tuvieramos:

- un driver con buena GPU
- varios workers con GPU
- una red y un despliegue pensados para entrenamiento

entonces si tendria sentido plantear:

- `TensorFlow` para el modelo deep learning
- `Horovod` para paralelizar el entrenamiento entre GPUs
- sincronizacion de gradientes entre workers

### Por que aqui queda como frontera teorica

Porque ese no es nuestro entorno.

Nuestro limite no ha sido conceptual, sino de infraestructura:

- cluster CPU puro
- Docker
- sin GPUs reales
- sin despliegue de entrenamiento distribuido especializado

Por eso `TensorFlow` y `Horovod` quedan documentados como:

- frontera tecnica
- trabajo futuro
- extension natural si existiera infraestructura real

---

## Lectura final del proyecto

La historia correcta de este trabajo es esta:

- primero demostramos que el problema puede entrar en un pipeline Big Data reproducible
- despues construimos una baseline distribuida seria con `Spark MLlib`
- luego mejoramos la representacion semantica con `Spark NLP`, sin perder entrenamiento distribuido
- al intentar pasar a deep learning entrenado, encontramos limites reales de arquitectura e infraestructura
- eso nos lleva a formular la separacion entre entrenamiento e inferencia
- finalmente dejamos `TensorFlow + Horovod` como la frontera que exigiria un entorno real con GPU

En resumen:

**el proyecto cierra con machine learning distribuido real y con una discusion honesta de por que el deep learning distribuido no era viable en la infraestructura disponible**

---

## DAG activo

Archivo: [explicit_lyrics_pipeline.py](dags/explicit_lyrics_pipeline.py)

Secuencia actual:

1. `ingesta_csv_pandas`
2. `spark_etl`
3. `spark_word2vec_mlp`
4. `spark_use_mllib_mlp`
5. `spark_distilbart_zeroshot`

Lectura recomendada para la memoria:

- el corazon practico del proyecto esta en los niveles `1`, `2a` y `2b`
- el DAG activo se cierra en `DistilBART`, y `TensorFlow + Horovod` quedan como frontera teorica

---

## Estructura principal

```text
BD-trabajo/
|-- dags/
|   `-- explicit_lyrics_pipeline.py
|-- spark-jobs/
|   |-- 02_etl.py
|   |-- 03_word2vec_mlp.py
|   |-- 04_spark_nlp_use_mllib_mlp.py
|   |-- 05_spark_nlp_classifierdl.py
|   |-- 05b_distilbart_zeroshot.py
|   `-- 99_comparative_study_unused.py
|-- notebooks/
|-- data/
|-- baseline-notebooks/
|-- Dockerfile.airflow-trabajo
|-- Dockerfile.spark-trabajo
|-- docker-compose-trabajo.yml
|-- init-airflow-trabajo.sh
|-- CONTEXTO_TRABAJO.md
`-- README.md
```

---

## Cierre

El problema no ha sido "no saber hacer deep learning".
El problema ha sido intentar llevarlo a gran escala sin la infraestructura adecuada.

Y precisamente por eso el aprendizaje del trabajo es muy fuerte:

- hemos distinguido entre distribuido real y distribuido aparente
- hemos visto hasta donde llega Spark con honestidad
- hemos identificado donde empieza la necesidad de GPU
- y hemos dejado una frontera futura clara y bien justificada
