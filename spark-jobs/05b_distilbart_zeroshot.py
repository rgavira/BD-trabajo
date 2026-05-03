"""
05b_distilbart_zeroshot.py - Inferencia distribuida zero-shot con DistilBART-MNLI

Contexto:
  El script 05_spark_nlp_classifierdl.py intentaba entrenar ClassifierDL
  dentro de Spark NLP. Al investigar el fallo (EOFException en Kryo), se
  descubrio que ClassifierDL llama Dataset.collect() internamente y entrena
  con TensorFlow en el driver, no en los workers. La distribucion era una
  ilusion: el entrenamiento ocurria en el mismo sitio que el script 06.

  Ese choque llevo a una decision de diseno:
  Desacoplamiento de Entrenamiento e Inferencia.

  En lugar de entrenar un clasificador dentro de Spark NLP, usamos un modelo
  ya entrenado (DistilBART fine-tuned en MNLI) y hacemos solo inferencia.
  La inferencia si es distribuida: .transform() corre en los workers sin
  llamar a collect() ni acumular datos en el driver.

Nota sobre la eleccion del modelo:
  El intento inicial fue usar RoBertaForZeroShotClassification con el modelo
  roberta_classifier_large_mnli. Fallo con ClassCastException: ese modelo esta
  empaquetado en el hub de John Snow Labs como RoBertaForSequenceClassification,
  no como RoBertaForZeroShotClassification. Son tipos distintos en el registry
  de Spark NLP y no se pueden castear entre si.

  La solucion es BartForZeroShotClassification con distilbart_mnli_12_3, que
  si esta empaquetado como modelo zero-shot en el hub de JSL. El concepto es
  identico: modelo NLI preentrenado en MNLI, inferencia pura, sin entrenar
  sobre nuestros datos.
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
from sparknlp.annotator import Tokenizer, BartForZeroShotClassification


CANDIDATE_LABELS = ["explicit", "clean"]
EXPLICIT_LABEL = "explicit"
TEST_LIMIT_ROWS = 1000
TEST_SAMPLE_FRACTION = None
INFER_BATCHES = 1
HDFS_TMP_PRED = "hdfs://namenode:9000/tmp/distilbart_zeroshot_predictions/"
HDFS_TMP_PREVIEW = "hdfs://namenode:9000/tmp/distilbart_zeroshot_preview/"


def log_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def log_info(label, value):
    print(f"  {label:<28} {value}")


def create_spark_session():
    return (
        SparkSession.builder
        .appName("ExplicitLyrics_DistilBART_ZeroShot")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer", "512m")
        .config("spark.kryoserializer.buffer.max", "2000M")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def sample_test_split(df, fraction, seed):
    if fraction is None or fraction >= 1.0:
        return df, 1.0

    return df.sampleBy("label", fractions={0: fraction, 1: fraction}, seed=seed), fraction


def transform_predictions(df):
    return (
        df.withColumn(
            "prediction_str",
            F.col("prediction").getItem(0).getField("result"),
        )
        .withColumn(
            "score",
            F.coalesce(
                F.col("prediction").getItem(0).getField("metadata")[EXPLICIT_LABEL].cast("double"),
                F.lit(0.5),
            ),
        )
        .withColumn(
            "prediction",
            F.when(F.col("prediction_str") == EXPLICIT_LABEL, F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .select("label", "prediction", "prediction_str", "score")
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
    spark.sparkContext.setJobDescription("DistilBART ZeroShot | Paso 1: Cargar datos")
    load_start = time.time()

    test = (
        spark.read.parquet("hdfs://namenode:9000/data/processed/test/")
        .select("text", "label")
    )
    if TEST_LIMIT_ROWS is not None:
        test = test.limit(TEST_LIMIT_ROWS).repartition(4)
        test_fraction = None
    else:
        test, test_fraction = sample_test_split(test, TEST_SAMPLE_FRACTION, seed=42)
        test = test.repartition(4)

    seed_row = (
        spark.read.parquet("hdfs://namenode:9000/data/processed/train/")
        .select("text", "label")
        .limit(1)
    )

    log_info(
        "Modo test",
        f"limitado a {TEST_LIMIT_ROWS} canciones" if TEST_LIMIT_ROWS is not None else "muestra estratificada",
    )
    log_info("Target test rows", TEST_LIMIT_ROWS if TEST_LIMIT_ROWS is not None else "dataset completo")
    log_info("Fraccion test", "-" if test_fraction is None else f"{test_fraction:.4f}")
    log_info("Particiones test", 4)
    log_info("Tiempo carga", f"{time.time() - load_start:.1f}s")

    log_section("2) Construccion del pipeline zero-shot")
    log_info("Modelo", "distilbart_mnli_12_3")
    log_info("Anotador", "BartForZeroShotClassification")
    log_info("Estrategia", "Zero-shot NLI (sin entrenamiento sobre nuestros datos)")
    log_info("Etiquetas candidatas", str(CANDIDATE_LABELS))
    log_info("Entrenamiento", "ninguno - inferencia pura")
    log_info("Ejecucion del fit", "carga del modelo preentrenado")
    log_info(
        "Nota",
        "RoBertaForZeroShotClassification descartado: roberta_classifier_large_mnli se empaqueta como SequenceClassification en el hub JSL",
    )

    document = DocumentAssembler().setInputCol("text").setOutputCol("document")
    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
    zero_shot = (
        BartForZeroShotClassification
        .pretrained("distilbart_mnli_12_3", "en")
        .setInputCols(["document", "token"])
        .setOutputCol("prediction")
        .setCandidateLabels(CANDIDATE_LABELS)
    )
    pipeline = Pipeline(stages=[document, tokenizer, zero_shot])

    log_section("3) Carga del modelo pre-entrenado")
    log_info("Descarga", "automatica desde servers de John Snow Labs")
    log_info("Nota", ".fit() carga pesos, no ejecuta gradientes sobre nuestros datos")
    fit_start = time.time()
    model = pipeline.fit(seed_row)
    log_info("Tiempo carga modelo", f"{time.time() - fit_start:.1f}s")

    log_section("4) Inferencia distribuida en test set")
    spark.sparkContext.setJobDescription("DistilBART ZeroShot | Paso 2: Inferencia en workers")
    infer_start = time.time()
    batches = INFER_BATCHES if TEST_LIMIT_ROWS is None else 1
    log_info("Lotes inferencia", batches)
    log_info(
        "Filas por lote aprox",
        TEST_LIMIT_ROWS if TEST_LIMIT_ROWS is not None else f"~{max(1, 1_000 // INFER_BATCHES):,}",
    )

    sc = spark.sparkContext
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_PRED), True)
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_PREVIEW), True)

    test_batches = [test] if TEST_LIMIT_ROWS is not None else test.randomSplit([1.0] * INFER_BATCHES, seed=42)

    for idx, batch_df in enumerate(test_batches, start=1):
        batch_start = time.time()
        log_info(f"Lote {idx}/{batches}", "inicio")

        (
            transform_predictions(model.transform(batch_df))
            .write.mode("append").parquet(HDFS_TMP_PRED)
        )

        log_info(f"Lote {idx}/{batches}", f"fin | tiempo={time.time() - batch_start:.1f}s")

    infer_time = time.time() - infer_start
    log_info("Tiempo inferencia", f"{infer_time:.1f}s")
    log_info("Predicciones", f"OK {HDFS_TMP_PRED}")

    if TEST_LIMIT_ROWS is not None:
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
    spark.sparkContext.setJobDescription("DistilBART ZeroShot | Paso 3: Metricas")
    eval_start = time.time()
    predictions = spark.read.parquet(HDFS_TMP_PRED)
    metrics = compute_metrics(predictions)
    metrics.update(
        {
            "model": "DistilBART_ZeroShot_MNLI",
            "candidate_labels": CANDIDATE_LABELS,
            "trained_on_data": False,
            "distributed": True,
            "collect_on_driver": False,
            "test_mode": f"limited_test_{TEST_LIMIT_ROWS}" if TEST_LIMIT_ROWS is not None else "sampled_test",
            "test_rows_target": TEST_LIMIT_ROWS if TEST_LIMIT_ROWS is not None else None,
            "test_fraction": None if test_fraction is None else round(test_fraction, 4),
            "infer_batches": batches,
            "infer_time_s": round(infer_time, 1),
            "eval_time_s": round(time.time() - eval_start, 1),
            "total_time_s": round(time.time() - total_start, 1),
        }
    )

    log_section("6) Resultados")
    for key, value in metrics.items():
        log_info(key, value)

    log_section("7) Persistencia en HDFS")
    model.write().overwrite().save("hdfs://namenode:9000/models/distilbart_zeroshot/")
    log_info("Modelo", "OK hdfs:///models/distilbart_zeroshot/")

    metrics_path = "hdfs://namenode:9000/metrics/distilbart_zeroshot/"
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(metrics_path), True)
    sc.parallelize([json.dumps(metrics, indent=2)]).saveAsTextFile(metrics_path)
    log_info("Metricas", "OK hdfs:///metrics/distilbart_zeroshot/")

    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_PRED), True)
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_PREVIEW), True)
    spark.stop()


if __name__ == "__main__":
    main()
