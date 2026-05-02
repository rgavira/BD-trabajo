"""
05b_roberta_zeroshot.py — Inferencia distribuida zero-shot con DistilBART-MNLI

Contexto:
  El script 05_spark_nlp_classifierdl.py intentaba entrenar ClassifierDL
  dentro de Spark NLP. Al investigar el fallo (EOFException en Kryo), se
  descubrio que ClassifierDL llama Dataset.collect() internamente y entrena
  con TensorFlow en el driver — no en los workers. La distribucion era una
  ilusion: el entrenamiento ocurria en el mismo sitio que el script 06.

  Ese choque llevo a una decision de diseno:
  Desacoplamiento de Entrenamiento e Inferencia.

  En lugar de entrenar un clasificador dentro de Spark NLP, usamos un modelo
  ya entrenado (DistilBART fine-tuned en MNLI) y hacemos solo inferencia.
  La inferencia SI es distribuida: .transform() corre en los workers sin
  llamar a collect() ni acumular datos en el driver.

Nota sobre la eleccion del modelo:
  El intento inicial fue usar RoBertaForZeroShotClassification con el modelo
  roberta_classifier_large_mnli. Fallo con ClassCastException: ese modelo esta
  empaquetado en el hub de John Snow Labs como RoBertaForSequenceClassification,
  no como RoBertaForZeroShotClassification. Son tipos distintos en el registry
  de Spark NLP y no se pueden castear entre si.

  La solucion es BartForZeroShotClassification con distilbart_mnli_12_3, que
  SI esta empaquetado como modelo zero-shot en el hub de JSL. El concepto es
  identico: modelo NLI preentrenado en MNLI, inferencia pura, sin entrenar
  sobre nuestros datos.

Por que zero-shot:
  DistilBART entrenado en MNLI puede clasificar texto en categorias arbitrarias
  sin haber visto ejemplos de esa tarea. El modelo recibe el texto y etiquetas
  candidatas, y decide cual encaja mejor mediante razonamiento de implicacion
  logica (Natural Language Inference).

  Le decimos: etiquetas posibles = ["explicit", "clean"]
  El modelo decide cual implica mejor el texto de la cancion.

Que significa distribuido aqui:
  .transform() parte el dataset en particiones y cada worker clasifica la suya.
  No hay collect(). No hay Kryo entre datos de entrenamiento y driver.
  El modelo se broadcastea a los workers una sola vez al inicio.

Pipeline:
  text -> DocumentAssembler -> Tokenizer -> BartForZeroShotClassification

Limitacion conocida:
  Este modelo no fue entrenado sobre letras de canciones ni sobre la
  definicion de Explicit de Spotify. Es zero-shot: generaliza desde su
  preentrenamiento. Las metricas seran peores que los modelos entrenados
  sobre nuestro dataset, lo cual es parte del argumento del trabajo.

Resultados:
  hdfs:///models/roberta_zeroshot/
  hdfs:///metrics/roberta_zeroshot/
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
EXPLICIT_LABEL   = "explicit"
HDFS_TMP_PRED    = "hdfs://namenode:9000/tmp/roberta_zeroshot_predictions/"


def log_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def log_info(label, value):
    print(f"  {label:<28} {value}")


def create_spark_session():
    return (
        SparkSession.builder
        .appName("ExplicitLyrics_RoBERTa_ZeroShot")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer", "512m")
        .config("spark.kryoserializer.buffer.max", "2000M")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def compute_metrics(predictions):
    ev = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    f1_weighted     = ev.setMetricName("f1").evaluate(predictions)
    acc             = ev.setMetricName("accuracy").evaluate(predictions)
    prec            = ev.setMetricName("weightedPrecision").evaluate(predictions)
    rec             = ev.setMetricName("weightedRecall").evaluate(predictions)
    f1_explicit     = ev.setMetricName("fMeasureByLabel").setMetricLabel(1.0).evaluate(predictions)
    f1_not_explicit = ev.setMetricName("fMeasureByLabel").setMetricLabel(0.0).evaluate(predictions)
    f1_macro        = (f1_explicit + f1_not_explicit) / 2.0

    try:
        auc = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="score", metricName="areaUnderROC"
        ).evaluate(predictions)
        auc = round(auc, 4)
    except Exception:
        auc = None

    return {
        "f1_weighted":        round(f1_weighted, 4),
        "f1_macro":           round(f1_macro, 4),
        "f1_explicit":        round(f1_explicit, 4),
        "f1_not_explicit":    round(f1_not_explicit, 4),
        "accuracy":           round(acc, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted":    round(rec, 4),
        "auc_roc":            auc,
    }


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    total_start = time.time()

    # ──────────────────────────────────────────────────────────────────────
    log_section("1) Carga de datos desde HDFS")
    spark.sparkContext.setJobDescription("RoBERTa ZeroShot | Paso 1: Cargar datos")
    load_start = time.time()

    test = (
        spark.read.parquet("hdfs://namenode:9000/data/processed/test/")
        .select("text", "label")
    )
    # Una sola fila para satisfacer la API de Spark ML en el .fit()
    # No hay entrenamiento: el .fit() solo carga los pesos pre-entrenados
    seed_row = (
        spark.read.parquet("hdfs://namenode:9000/data/processed/train/")
        .select("text", "label")
        .limit(1)
    )
    log_info("Particiones test",       test.rdd.getNumPartitions())
    log_info("Tiempo carga",           f"{time.time() - load_start:.1f}s")

    # ──────────────────────────────────────────────────────────────────────
    log_section("2) Construccion del pipeline zero-shot")
    log_info("Modelo",                 "distilbart_mnli_12_3")
    log_info("Anotador",               "BartForZeroShotClassification")
    log_info("Estrategia",             "Zero-shot NLI (sin entrenamiento sobre nuestros datos)")
    log_info("Etiquetas candidatas",   str(CANDIDATE_LABELS))
    log_info("Entrenamiento",          "ninguno — inferencia pura")
    log_info("Ejecucion del fit",      "workers (transform distribuido, sin collect interno)")
    log_info("Nota",                   "RoBertaForZeroShotClassification descartado: roberta_classifier_large_mnli esta empaquetado como SequenceClassification en el hub JSL (ClassCastException)")

    document = (
        DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
    )

    tokenizer = (
        Tokenizer()
        .setInputCols(["document"])
        .setOutputCol("token")
    )

    zero_shot = (
        BartForZeroShotClassification
        .pretrained("distilbart_mnli_12_3", "en")
        .setInputCols(["document", "token"])
        .setOutputCol("prediction")
        .setCandidateLabels(CANDIDATE_LABELS)
    )

    pipeline = Pipeline(stages=[document, tokenizer, zero_shot])

    # ──────────────────────────────────────────────────────────────────────
    log_section("3) Carga del modelo pre-entrenado")
    log_info("Descarga",               "automatica desde servers de John Snow Labs")
    log_info("Nota",                   ".fit() carga pesos, no ejecuta gradientes sobre nuestros datos")
    fit_start = time.time()
    model = pipeline.fit(seed_row)
    log_info("Tiempo carga modelo",    f"{time.time() - fit_start:.1f}s")

    # ──────────────────────────────────────────────────────────────────────
    log_section("4) Inferencia distribuida en test set")
    spark.sparkContext.setJobDescription("RoBERTa ZeroShot | Paso 2: Inferencia en workers")
    infer_start = time.time()

    (
        model.transform(test)
        .withColumn("prediction_str",
            F.col("prediction").getItem(0).getField("result"))
        .withColumn("score",
            F.coalesce(
                F.col("prediction").getItem(0).getField("metadata")[EXPLICIT_LABEL].cast("double"),
                F.lit(0.5)
            ))
        .withColumn("prediction",
            F.when(F.col("prediction_str") == EXPLICIT_LABEL, F.lit(1.0))
            .otherwise(F.lit(0.0)))
        .select("label", "prediction", "prediction_str", "score")
        .write.mode("overwrite").parquet(HDFS_TMP_PRED)
    )
    infer_time = time.time() - infer_start
    log_info("Tiempo inferencia",      f"{infer_time:.1f}s")
    log_info("Predicciones",           f"✓ {HDFS_TMP_PRED}")

    # ──────────────────────────────────────────────────────────────────────
    log_section("5) Evaluacion")
    spark.sparkContext.setJobDescription("RoBERTa ZeroShot | Paso 3: Metricas")
    eval_start = time.time()
    predictions = spark.read.parquet(HDFS_TMP_PRED)
    metrics = compute_metrics(predictions)
    metrics.update({
        "model":            "DistilBART_ZeroShot_MNLI",
        "candidate_labels": CANDIDATE_LABELS,
        "trained_on_data":  False,
        "distributed":      True,
        "collect_on_driver": False,
        "infer_time_s":     round(infer_time, 1),
        "eval_time_s":      round(time.time() - eval_start, 1),
        "total_time_s":     round(time.time() - total_start, 1),
    })

    log_section("6) Resultados")
    for k, v in metrics.items():
        log_info(k, v)

    # ──────────────────────────────────────────────────────────────────────
    log_section("7) Persistencia en HDFS")
    model.write().overwrite().save("hdfs://namenode:9000/models/roberta_zeroshot/")
    log_info("Modelo",  "✓ hdfs:///models/roberta_zeroshot/")

    metrics_path = "hdfs://namenode:9000/metrics/roberta_zeroshot/"
    sc = spark.sparkContext
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(metrics_path), True)
    sc.parallelize([json.dumps(metrics, indent=2)]).saveAsTextFile(metrics_path)
    log_info("Metricas", "✓ hdfs:///metrics/roberta_zeroshot/")

    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_PRED), True)
    spark.stop()


if __name__ == "__main__":
    main()
