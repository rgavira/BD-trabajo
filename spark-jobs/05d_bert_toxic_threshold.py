"""
05d_bert_toxic_threshold.py - Toxicidad con BERT y umbral para explicitud

Idea:
  `bert_classifier_toxic` no trae una clase negativa (`neutral`/`clean`), pero
  si devuelve scores por varias categorias toxicas. En este nivel convertimos
  esos scores a una decision binaria interpretable:

    explicit = 1.0 si max(score_toxico) >= 0.60
    clean    = 0.0 en caso contrario

  Ademas guardamos que etiqueta concreta disparo el positivo para trazabilidad.
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
from sparknlp.annotator import Tokenizer, BertForSequenceClassification


MODEL_NAME = "bert_classifier_toxic"
MODEL_LANG = "en"
TOXIC_LABELS = [
    "obscene",
    "insult",
    "severe_toxic",
    "identity_hate",
    "threat",
    "toxic",
]
EXPLICIT_THRESHOLD = 0.5
TEST_LIMIT_ROWS = 3000
HDFS_TMP_PRED = "hdfs://namenode:9000/tmp/bert_toxic_threshold_predictions/"
HDFS_TMP_PREVIEW = "hdfs://namenode:9000/tmp/bert_toxic_threshold_preview/"
HDFS_MODEL_DIR = "hdfs://namenode:9000/models/bert_toxic_threshold/"
HDFS_METRICS_DIR = "hdfs://namenode:9000/metrics/bert_toxic_threshold/"


def log_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def log_info(label, value):
    print(f"  {label:<28} {value}")


def create_spark_session():
    return (
        SparkSession.builder
        .appName("ExplicitLyrics_BERT_Toxic_Threshold")
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

    score_exprs = {label: F.coalesce(raw_metadata[label].cast("double"), F.lit(0.0)) for label in TOXIC_LABELS}
    max_toxic_score = F.greatest(*score_exprs.values())

    trigger_label = (
        F.when(score_exprs["obscene"] == max_toxic_score, F.lit("obscene"))
        .when(score_exprs["insult"] == max_toxic_score, F.lit("insult"))
        .when(score_exprs["severe_toxic"] == max_toxic_score, F.lit("severe_toxic"))
        .when(score_exprs["identity_hate"] == max_toxic_score, F.lit("identity_hate"))
        .when(score_exprs["threat"] == max_toxic_score, F.lit("threat"))
        .otherwise(F.lit("toxic"))
    )

    return (
        df.withColumn("prediction_str", raw_result)
        .withColumn("max_toxic_score", max_toxic_score)
        .withColumn(
            "trigger_label",
            F.when(F.col("max_toxic_score") >= F.lit(EXPLICIT_THRESHOLD), trigger_label).otherwise(F.lit("clean_below_threshold")),
        )
        .withColumn(
            "prediction",
            F.when(F.col("max_toxic_score") >= F.lit(EXPLICIT_THRESHOLD), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn("score", F.col("max_toxic_score"))
        .select("label", "prediction", "prediction_str", "trigger_label", "score")
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
    spark.sparkContext.setJobDescription("BERT Toxic Threshold | Paso 1: Cargar datos")
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
    log_info("Anotador", "BertForSequenceClassification")
    log_info("Regla", f"explicit si max(score_toxico) >= {EXPLICIT_THRESHOLD:.2f}")
    log_info("Etiquetas toxicas", ", ".join(TOXIC_LABELS))

    document = DocumentAssembler().setInputCol("text").setOutputCol("document")
    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
    classifier = (
        BertForSequenceClassification
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
    spark.sparkContext.setJobDescription("BERT Toxic Threshold | Paso 2: Inferencia")
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
    log_info("Trazabilidad", "trigger_label indica que etiqueta disparo el explicit")

    log_section("5) Evaluacion")
    spark.sparkContext.setJobDescription("BERT Toxic Threshold | Paso 3: Metricas")
    eval_start = time.time()
    predictions = spark.read.parquet(HDFS_TMP_PRED)
    metrics = compute_metrics(predictions)
    metrics.update(
        {
            "model": "BERT_Toxicity_Threshold_Classifier",
            "pretrained_name": MODEL_NAME,
            "threshold": EXPLICIT_THRESHOLD,
            "toxic_labels": TOXIC_LABELS,
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
