"""
DAG: explicit_lyrics_pipeline
Pipeline completo para deteccion de contenido explicito en canciones.

Tareas activas:
  1. ingesta_csv_pandas         -> Lee el CSV y escribe Parquet en volumen compartido
  2. spark_etl                  -> ETL distribuido: limpieza, HDFS raw + processed
  3. spark_word2vec_mlp         -> Baseline distribuida con Spark MLlib
  4. spark_use_mllib_mlp        -> USE + MLP MLlib, 100% distribuido
  5. spark_roberta_zeroshot     -> Inferencia distribuida zero-shot con RoBERTa (sin training)
  6. spark_tensorflow_baselines -> Spark como capa de datos + Keras en CPU

Nota sobre la tarea 5:
  El script original (05_spark_nlp_classifierdl.py) intentaba entrenar ClassifierDL
  de Spark NLP. Se descubrio que ClassifierDL llama Dataset.collect() internamente
  y entrena con TF en el driver, no en los workers. Ese script se conserva en
  spark-jobs/ como referencia historica del limite encontrado.

  La tarea 5 usa ahora 05b_roberta_zeroshot.py: inferencia zero-shot distribuida
  con RoBERTa-large-MNLI. El .transform() corre en workers sin collect() interno.
  Esto ilustra el patron de Desacoplamiento de Entrenamiento e Inferencia.

  El estudio comparativo de MLlib queda fuera del DAG principal.
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
    description="Pipeline Big Data: CSV -> HDFS -> MLlib -> USE+MLlib -> Spark NLP -> TensorFlow",
    schedule_interval=None,
    catchup=False,
    tags=["explicit-lyrics", "nlp", "spark", "hdfs", "mllib", "tensorflow"],
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


t5_roberta_zeroshot = BashOperator(
    task_id="spark_roberta_zeroshot",
    bash_command=(
        f"TF_CPP_MIN_LOG_LEVEL=2 "
        f"TF_FORCE_GPU_ALLOW_GROWTH=true "
        f"{SPARK_SUBMIT} "
        f"--master {SPARK_MASTER} "
        f"--deploy-mode client "
        f"{SPARK_BASE_CONF} "
        f"{SPARK_NLP_CONF} "
        f"--executor-memory 4g --executor-cores 2 --num-executors 2 "
        f"--driver-memory 6g "
        f"/opt/spark-jobs/05b_roberta_zeroshot.py"
    ),
    dag=dag,
)


t6_tensorflow = BashOperator(
    task_id="spark_tensorflow_baselines",
    bash_command=(
        f"{SPARK_SUBMIT} "
        f"--master {SPARK_MASTER} "
        f"--deploy-mode client "
        f"{SPARK_BASE_CONF} "
        f"{SPARK_RESOURCES} "
        f"--driver-memory 8g "
        f"/opt/spark-jobs/06_tensorflow_text_baselines.py"
    ),
    dag=dag,
)


t1_ingesta >> t2_etl >> t3_word2vec_mlp >> t4_use_mllib_mlp >> t5_roberta_zeroshot >> t6_tensorflow
