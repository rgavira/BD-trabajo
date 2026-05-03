"""
DAG: explicit_lyrics_pipeline
Pipeline completo para deteccion de contenido explicito en canciones.

Tareas activas:
  1. ingesta_csv_pandas         -> Lee el CSV y escribe Parquet en volumen compartido
  2. spark_etl                  -> ETL distribuido: limpieza, HDFS raw + processed
  3. spark_word2vec_mlp         -> Baseline distribuida con Spark MLlib
  4. spark_use_mllib_mlp        -> USE + MLP MLlib, 100% distribuido
  5. spark_distilbart_zeroshot  -> Inferencia zero-shot con DistilBART como frontera del proyecto
  6. spark_distilbert_toxicity  -> Clasificador binario toxic/neutral con RoBERTa
  7. spark_bert_toxic_threshold -> BERT toxico con umbral 0.60 y trazabilidad por etiqueta
  8. spark_sent_bert_mllib_mlp  -> BertSentenceEmbeddings (sent_small_bert_L4_512) + MLP MLlib, prueba 1000 filas
  9. spark_minilm_mllib_mlp     -> MiniLM + MLP MLlib, embedding ligero de frase

Nota sobre la tarea 5:
  El script original (05_spark_nlp_classifierdl.py) intentaba entrenar ClassifierDL
  de Spark NLP. Se descubrio que ClassifierDL llama Dataset.collect() internamente
  y entrena con TF en el driver, no en los workers. Ese script se conserva en
  spark-jobs/ como referencia historica del limite encontrado.

  La tarea 5 usa ahora 05b_distilbart_zeroshot.py: inferencia zero-shot con
  DistilBART-MNLI. El .transform() corre desde Spark, pero el coste en CPU y
  el peso del driver marcan la frontera practica del entorno.
  Esto ilustra el patron de Desacoplamiento de Entrenamiento e Inferencia.

  La tarea 6 prueba un clasificador binario de toxicidad ya entrenado. No se
  plantea como modelo final todavia, sino como contraste frente al enfoque
  zero-shot para ver si un modelo especializado separa mejor el texto.

  La tarea 8 prueba MiniLM como embedding congelado mas ligero que MPNet y mas
  moderno que USE. La idea es comprobar si una representacion contextual mejor
  mantiene tiempos razonables y mejora frente a Word2Vec y USE.

  La tarea 7 usa un modelo multi-clase de toxicidad y aplica una regla de
  decision interpretable: si alguna dimension toxica supera 0.60, la cancion
  se marca como explicit. Tambien guarda que etiqueta disparo el positivo.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


SPARK_SUBMIT = "/opt/spark/bin/spark-submit"
SPARK_MASTER = "spark://spark-master:7077"
HDFS_DEFAULT = "hdfs://namenode:9000"
SPARK_NLP_PACKAGE = "com.johnsnowlabs.nlp:spark-nlp_2.12:6.4.0"

SPARK_BASE_CONF = (
    f"--conf spark.hadoop.fs.defaultFS={HDFS_DEFAULT} "
    f"--conf spark.hadoop.dfs.client.use.datanode.hostname=true "
    f"--conf spark.eventLog.enabled=true "
    f"--conf spark.eventLog.dir=/tmp/spark-events "
    f"--conf spark.pyspark.python=python3.11 "
    f"--conf spark.pyspark.driver.python=python3.11"
)
SPARK_NLP_CONF = (
    f"--packages {SPARK_NLP_PACKAGE} "
    f"--conf spark.serializer=org.apache.spark.serializer.KryoSerializer "
    f"--conf spark.kryoserializer.buffer=512m "
    f"--conf spark.kryoserializer.buffer.max=2000M "
    f"--conf spark.sql.shuffle.partitions=4 "
    f"--conf spark.executor.memoryOverhead=1536m"
)
SPARK_RESOURCES = "--executor-memory 3g --executor-cores 2 --num-executors 2"
SPARK_NLP_RESOURCES = "--executor-memory 4g --executor-cores 2 --num-executors 2"

default_args = {
    "owner": "bd-trabajo",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "explicit_lyrics_pipeline",
    default_args=default_args,
    description="Pipeline Big Data: CSV -> HDFS -> MLlib -> USE/MPNet + MLlib -> Transformers",
    schedule_interval=None,
    catchup=False,
    tags=["explicit-lyrics", "nlp", "spark", "hdfs", "mllib", "distilbart", "tox", "threshold", "minilm"],
)


def ingesta_csv_pandas(**context):
    import os

    import pandas as pd

    csv_path = "/opt/data/spotify_dataset.csv"
    output_path = "/opt/data/songs_raw.parquet"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n\n  CSV no encontrado en {csv_path}\n"
            "  Copia el archivo spotify_dataset.csv en la carpeta ./data/\n"
        )

    print(f"Cargando CSV desde {csv_path}...")
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    print(f"  Filas cargadas : {len(df):,}")
    print(f"  Columnas       : {df.columns.tolist()}")

    df.to_parquet(output_path, index=False)
    print(f"Parquet guardado en {output_path}")


t1_ingesta = PythonOperator(
    task_id="ingesta_csv_pandas",
    python_callable=ingesta_csv_pandas,
    dag=dag,
)


t2_etl = BashOperator(
    task_id="spark_etl",
    bash_command=(
        f"{SPARK_SUBMIT} "
        f"--master {SPARK_MASTER} "
        f"--deploy-mode client "
        f"{SPARK_BASE_CONF} "
        f"{SPARK_RESOURCES} "
        f"/opt/spark-jobs/02_etl.py"
    ),
    dag=dag,
)


t3_word2vec_mlp = BashOperator(
    task_id="spark_word2vec_mlp",
    bash_command=(
        f"{SPARK_SUBMIT} "
        f"--master {SPARK_MASTER} "
        f"--deploy-mode client "
        f"{SPARK_BASE_CONF} "
        f"{SPARK_RESOURCES} "
        f"--driver-memory 2g "
        f"/opt/spark-jobs/03_word2vec_mlp.py"
    ),
    dag=dag,
)


t4_use_mllib_mlp = BashOperator(
    task_id="spark_use_mllib_mlp",
    bash_command=(
        f"TF_CPP_MIN_LOG_LEVEL=2 "
        f"TF_FORCE_GPU_ALLOW_GROWTH=true "
        f"{SPARK_SUBMIT} "
        f"--master {SPARK_MASTER} "
        f"--deploy-mode client "
        f"{SPARK_BASE_CONF} "
        f"{SPARK_NLP_CONF} "
        f"--num-executors 1 --executor-cores 4 --executor-memory 4g "
        f"--driver-memory 8g "
        f"--conf spark.driver.maxResultSize=1g "
        f"/opt/spark-jobs/04_spark_nlp_use_mllib_mlp.py"
    ),
    dag=dag,
)


t5_minilm_mllib_mlp = BashOperator(
    task_id="spark_minilm_mllib_mlp",
    bash_command=(
        f"TF_CPP_MIN_LOG_LEVEL=2 "
        f"TF_FORCE_GPU_ALLOW_GROWTH=true "
        f"{SPARK_SUBMIT} "
        f"--master {SPARK_MASTER} "
        f"--deploy-mode client "
        f"{SPARK_BASE_CONF} "
        f"{SPARK_NLP_CONF} "
        f"--num-executors 1 --executor-cores 4 --executor-memory 4g "
        f"--driver-memory 8g "
        f"--conf spark.driver.maxResultSize=1g "
        f"/opt/spark-jobs/04b_spark_nlp_minilm_mllib_mlp.py"
    ),
    dag=dag,
)


t6_distilbart_zeroshot = BashOperator(
    task_id="spark_distilbart_zeroshot",
    bash_command=(
        f"TF_CPP_MIN_LOG_LEVEL=2 "
        f"TF_FORCE_GPU_ALLOW_GROWTH=true "
        f"{SPARK_SUBMIT} "
        f"--master {SPARK_MASTER} "
        f"--deploy-mode client "
        f"{SPARK_BASE_CONF} "
        f"{SPARK_NLP_CONF} "
        f"--executor-memory 4g --executor-cores 2 --num-executors 2 "
        f"--driver-memory 8g "
        f"--conf spark.driver.maxResultSize=2g "
        f"/opt/spark-jobs/05b_distilbart_zeroshot.py"
    ),
    dag=dag,
)


t7_distilbert_toxicity = BashOperator(
    task_id="spark_distilbert_toxicity",
    bash_command=(
        f"TF_CPP_MIN_LOG_LEVEL=2 "
        f"TF_FORCE_GPU_ALLOW_GROWTH=true "
        f"{SPARK_SUBMIT} "
        f"--master {SPARK_MASTER} "
        f"--deploy-mode client "
        f"{SPARK_BASE_CONF} "
        f"{SPARK_NLP_CONF} "
        f"--executor-memory 4g --executor-cores 2 --num-executors 2 "
        f"--driver-memory 8g "
        f"--conf spark.driver.maxResultSize=2g "
        f"/opt/spark-jobs/05c_distilbert_toxicity.py"
    ),
    dag=dag,
)


t8_bert_toxic_threshold = BashOperator(
    task_id="spark_bert_toxic_threshold",
    bash_command=(
        f"TF_CPP_MIN_LOG_LEVEL=2 "
        f"TF_FORCE_GPU_ALLOW_GROWTH=true "
        f"{SPARK_SUBMIT} "
        f"--master {SPARK_MASTER} "
        f"--deploy-mode client "
        f"{SPARK_BASE_CONF} "
        f"{SPARK_NLP_CONF} "
        f"--executor-memory 4g --executor-cores 2 --num-executors 2 "
        f"--driver-memory 8g "
        f"--conf spark.driver.maxResultSize=2g "
        f"/opt/spark-jobs/05d_bert_toxic_threshold.py"
    ),
    dag=dag,
)


t9_sent_bert_mllib_mlp = BashOperator(
    task_id="spark_sent_bert_mllib_mlp",
    bash_command=(
        f"TF_CPP_MIN_LOG_LEVEL=2 "
        f"TF_FORCE_GPU_ALLOW_GROWTH=true "
        f"{SPARK_SUBMIT} "
        f"--master {SPARK_MASTER} "
        f"--deploy-mode client "
        f"{SPARK_BASE_CONF} "
        f"{SPARK_NLP_CONF} "
        f"--num-executors 1 --executor-cores 4 --executor-memory 4g "
        f"--driver-memory 8g "
        f"--conf spark.driver.maxResultSize=1g "
        f"/opt/spark-jobs/04c_spark_nlp_sent_bert_mllib_mlp.py"
    ),
    dag=dag,
)


t1_ingesta >> t2_etl >> t3_word2vec_mlp >> t4_use_mllib_mlp >> t6_distilbart_zeroshot >> t7_distilbert_toxicity >> t8_bert_toxic_threshold >> t9_sent_bert_mllib_mlp >> t5_minilm_mllib_mlp
