"""
05c_distilbert_toxicity.py - Inferencia con clasificador binario de toxicidad

Motivacion:
  Tras comprobar que DistilBART zero-shot funciona como prueba de viabilidad
  pero da resultados flojos para este problema, abrimos una variante con un
  Transformer ya entrenado especificamente para toxicidad.

Nota de diseno:
  Para este nivel usamos `roberta_classifier_toxicity`, que devuelve una
  clasificacion binaria mas limpia:

    toxic
    neutral

  En nuestro problema:

    toxic -> 1.0 (explicit)
    neutral -> 0.0 (clean)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import json
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, RoBertaForSequenceClassification


MODEL_NAME = "roberta_classifier_toxicity"
MODEL_LANG = "en"
POSITIVE_LABELS = {
    "toxic",
}
TEST_LIMIT_ROWS = 1000
HDFS_TMP_PRED = "hdfs://namenode:9000/tmp/roberta_toxicity_predictions/"
HDFS_TMP_PREVIEW = "hdfs://namenode:9000/tmp/roberta_toxicity_preview/"
HDFS_MODEL_DIR = "hdfs://namenode:9000/models/roberta_toxicity/"
HDFS_METRICS_DIR = "hdfs://namenode:9000/metrics/roberta_toxicity/"


def log_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def log_info(label, value):
    print(f"  {label:<28} {value}")


def create_spark_session():
    return (
        SparkSession.builder
        .appName("ExplicitLyrics_RoBERTa_Toxicity")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer", "512m")
        .config("spark.kryoserializer.buffer.max", "2000M")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def read_preview_json_line(sc, hdfs_path):
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    preview_path = sc._jvm.org.apache.hadoop.fs.Path(hdfs_path)

    if not fs.exists(preview_path):
        return None

    for status in fs.listStatus(preview_path):
        name = status.getPath().getName()
        if status.isFile() and name.startswith("part-"):
            stream = fs.open(status.getPath())
            reader = sc._jvm.java.io.BufferedReader(
                sc._jvm.java.io.InputStreamReader(stream, "UTF-8")
            )
            try:
                return reader.readLine()
            finally:
                reader.close()
    return None


def transform_predictions(df):
    raw_result = F.lower(F.col("class").getItem(0).getField("result"))
    raw_metadata = F.col("class").getItem(0).getField("metadata")

    toxic_score = F.coalesce(
        raw_metadata["toxic"].cast("double"),
        raw_metadata["toxicity"].cast("double"),
        F.when(raw_result.isin(*list(POSITIVE_LABELS)), F.lit(1.0)).otherwise(F.lit(0.0)),
    )

    return (
        df.withColumn("prediction_str", raw_result)
        .withColumn("score", toxic_score)
        .withColumn(
            "prediction",
            F.when(F.col("prediction_str").isin(*list(POSITIVE_LABELS)), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .select("label", "prediction", "prediction_str", "score")
    )


def compute_metrics(predictions):
    ev = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    f1_weighted = ev.setMetricName("f1").evaluate(predictions)
    acc = ev.setMetricName("accuracy").evaluate(predictions)
    prec = ev.setMetricName("weightedPrecision").evaluate(predictions)
    rec = ev.setMetricName("weightedRecall").evaluate(predictions)

    try:
        f1_explicit = ev.setMetricName("fMeasureByLabel").setMetricLabel(1.0).evaluate(predictions)
    except Exception:
        f1_explicit = None

    try:
        f1_not_explicit = ev.setMetricName("fMeasureByLabel").setMetricLabel(0.0).evaluate(predictions)
    except Exception:
        f1_not_explicit = None

    f1_macro = (
        (f1_explicit + f1_not_explicit) / 2.0
        if f1_explicit is not None and f1_not_explicit is not None
        else None
    )

    try:
        auc = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="score", metricName="areaUnderROC"
        ).evaluate(predictions)
        auc = round(auc, 4)
    except Exception:
        auc = None

    return {
        "f1_weighted": round(f1_weighted, 4),
        "f1_macro": None if f1_macro is None else round(f1_macro, 4),
        "f1_explicit": None if f1_explicit is None else round(f1_explicit, 4),
        "f1_not_explicit": None if f1_not_explicit is None else round(f1_not_explicit, 4),
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "auc_roc": auc,
    }


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    total_start = time.time()

    log_section("1) Carga de datos desde HDFS")
    spark.sparkContext.setJobDescription("RoBERTa Toxicity | Paso 1: Cargar datos")
    load_start = time.time()

    test = (
        spark.read.parquet("hdfs://namenode:9000/data/processed/test/")
        .select("text", "label")
        .limit(TEST_LIMIT_ROWS)
        .repartition(4)
    )
    seed_row = (
        spark.read.parquet("hdfs://namenode:9000/data/processed/train/")
        .select("text", "label")
        .limit(1)
    )

    log_info("Modo test", f"limitado a {TEST_LIMIT_ROWS} canciones")
    log_info("Particiones test", 4)
    log_info("Tiempo carga", f"{time.time() - load_start:.1f}s")

    log_section("2) Construccion del pipeline toxicidad")
    log_info("Modelo", MODEL_NAME)
    log_info("Anotador", "RoBertaForSequenceClassification")
    log_info("Estrategia", "Clasificador preentrenado de toxicidad")
    log_info("Mapeo", "toxic -> explicit(1.0), neutral -> clean(0.0)")

    document = DocumentAssembler().setInputCol("text").setOutputCol("document")
    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
    classifier = (
        RoBertaForSequenceClassification
        .pretrained(MODEL_NAME, MODEL_LANG)
        .setInputCols(["document", "token"])
        .setOutputCol("class")
        .setCaseSensitive(False)
        .setMaxSentenceLength(512)
    )
    pipeline = Pipeline(stages=[document, tokenizer, classifier])

    log_section("3) Carga del modelo pre-entrenado")
    fit_start = time.time()
    model = pipeline.fit(seed_row)
    log_info("Tiempo carga modelo", f"{time.time() - fit_start:.1f}s")

    log_section("4) Inferencia en test set")
    spark.sparkContext.setJobDescription("RoBERTa Toxicity | Paso 2: Inferencia")
    infer_start = time.time()

    sc = spark.sparkContext
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_PRED), True)
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_PREVIEW), True)

    (
        transform_predictions(model.transform(test))
        .write.mode("overwrite").parquet(HDFS_TMP_PRED)
    )

    infer_time = time.time() - infer_start
    log_info("Tiempo inferencia", f"{infer_time:.1f}s")
    log_info("Predicciones", f"OK {HDFS_TMP_PRED}")

    log_section("4b) Salida del modelo")
    (
        spark.read.parquet(HDFS_TMP_PRED)
        .coalesce(1)
        .write.mode("overwrite").json(HDFS_TMP_PREVIEW)
    )
    log_info("Preview", f"JSON guardado en {HDFS_TMP_PREVIEW}")
    preview_line = read_preview_json_line(sc, HDFS_TMP_PREVIEW)
    log_info("Prediccion modelo", preview_line if preview_line else "no disponible")

    log_section("5) Evaluacion")
    spark.sparkContext.setJobDescription("RoBERTa Toxicity | Paso 3: Metricas")
    eval_start = time.time()
    predictions = spark.read.parquet(HDFS_TMP_PRED)

    if TEST_LIMIT_ROWS <= 1:
        log_info("Modo evaluacion", "omitida en smoke test de 1 cancion")
        metrics = {
            "f1_weighted": None,
            "f1_macro": None,
            "f1_explicit": None,
            "f1_not_explicit": None,
            "accuracy": None,
            "precision_weighted": None,
            "recall_weighted": None,
            "auc_roc": None,
            "evaluation_skipped": True,
            "evaluation_reason": "single_row_smoke_test",
        }
    else:
        metrics = compute_metrics(predictions)

    metrics.update(
        {
            "model": "RoBERTa_Toxicity_SequenceClassifier",
            "pretrained_name": MODEL_NAME,
            "trained_on_data": False,
            "distributed": True,
            "collect_on_driver": False,
            "test_mode": f"limited_test_{TEST_LIMIT_ROWS}",
            "test_rows_target": TEST_LIMIT_ROWS,
            "infer_time_s": round(infer_time, 1),
            "eval_time_s": round(time.time() - eval_start, 1),
            "total_time_s": round(time.time() - total_start, 1),
        }
    )

    log_section("6) Resultados")
    for key, value in metrics.items():
        log_info(key, value)

    log_section("7) Persistencia en HDFS")
    model.write().overwrite().save(HDFS_MODEL_DIR)
    log_info("Modelo", f"OK {HDFS_MODEL_DIR}")

    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_METRICS_DIR), True)
    sc.parallelize([json.dumps(metrics, indent=2)]).saveAsTextFile(HDFS_METRICS_DIR)
    log_info("Metricas", f"OK {HDFS_METRICS_DIR}")

    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_PRED), True)
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_PREVIEW), True)
    spark.stop()


if __name__ == "__main__":
    main()
