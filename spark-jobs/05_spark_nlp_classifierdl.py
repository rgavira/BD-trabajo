"""
05_spark_nlp_classifierdl.py — Clasificación distribuida con Spark NLP

Motivación:
  MLlib puro nos deja una línea base útil (Word2Vec + MLP), pero Spark NLP
  permite construir una pipeline más nativa de NLP moderno usando embeddings
  preentrenados de frase y un clasificador profundo integrado.

Pipeline:
  text → DocumentAssembler → UniversalSentenceEncoder → ClassifierDL

Qué aporta frente a 03_word2vec_mlp.py:
  - embeddings semánticos preentrenados en lugar de Word2Vec entrenado desde cero
  - clasificador profundo orientado a texto dentro del ecosistema Spark NLP
  - baseline más cercana a la lógica de AINE que un average embedding clásico

Resultados:
  hdfs:///models/spark_nlp_classifierdl/
  hdfs:///metrics/spark_nlp_classifierdl/
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"           # Reduce TF log noise
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"    # No pre-allocate GPU mem (safety)

import json
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import UniversalSentenceEncoder, ClassifierDLApproach


MODEL_NAME = "tfhub_use"
MODEL_LANG = "en"
SEED = 42
TRAIN_TARGET_ROWS = 10_000
TEST_TARGET_ROWS = None
TRAIN_SAMPLE_FRACTION = 0.0252
HDFS_TMP_TRAIN_ANNOT = "hdfs://namenode:9000/tmp/sparknlp_classifierdl_train_emb/"
HDFS_TMP_TEST_ANNOT = "hdfs://namenode:9000/tmp/sparknlp_classifierdl_test_emb/"
HDFS_TMP_PRED = "hdfs://namenode:9000/tmp/sparknlp_classifierdl_predictions/"


def log_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def log_info(label, value):
    print(f"  {label:<26} {value}")


def create_spark_session():
    return (
        SparkSession.builder
        .appName("ExplicitLyrics_SparkNLP_ClassifierDL")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "2000M")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def create_classifier_session():
    return (
        SparkSession.builder
        .appName("ExplicitLyrics_SparkNLP_ClassifierDL_Fit")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )


def stratified_sample(df, target_rows, seed, sample_fraction=None):
    if target_rows is None:
        return df, None, None

    fraction = sample_fraction if sample_fraction is not None else 1.0
    fraction = min(max(float(fraction), 0.0), 1.0)
    labels = [0, 1]
    fractions = {label: fraction for label in labels}
    sampled = df.sampleBy("label", fractions, seed=seed)
    return sampled, None, fraction


def compute_metrics(predictions):
    ev = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    f1_weighted = ev.setMetricName("f1").evaluate(predictions)
    acc = ev.setMetricName("accuracy").evaluate(predictions)
    prec = ev.setMetricName("weightedPrecision").evaluate(predictions)
    rec = ev.setMetricName("weightedRecall").evaluate(predictions)

    auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="score", metricName="areaUnderROC"
    ).evaluate(predictions)

    return {
        "f1_weighted": round(f1_weighted, 4),
        "f1_macro": None,
        "f1_explicit": None,
        "f1_not_explicit": None,
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "auc_roc": round(auc, 4),
    }


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    total_start = time.time()

    log_section("1) Carga de train/test desde HDFS")
    spark.sparkContext.setJobDescription("SparkNLP | Paso 1: Cargar train/test desde HDFS")
    load_start = time.time()
    train = (
        spark.read.parquet("hdfs://namenode:9000/data/processed/train/")
        .select("text", "label")
        .withColumn("label_str", F.col("label").cast("string"))
    )
    test = (
        spark.read.parquet("hdfs://namenode:9000/data/processed/test/")
        .select("text", "label")
        .withColumn("label_str", F.col("label").cast("string"))
    )
    train, train_rows, train_fraction = stratified_sample(
        train, TRAIN_TARGET_ROWS, SEED, sample_fraction=TRAIN_SAMPLE_FRACTION
    )
    test, test_rows, test_fraction = stratified_sample(test, TEST_TARGET_ROWS, SEED)
    log_info("Particiones train", train.rdd.getNumPartitions())
    log_info("Particiones test", test.rdd.getNumPartitions())
    log_info("Train rows usados", "~100k estratificado" if train_rows is None and TRAIN_TARGET_ROWS is not None else ("dataset completo" if train_rows is None else f"{train_rows:,}"))
    log_info("Test rows usados", "dataset completo" if test_rows is None else f"{test_rows:,}")
    log_info("Fraction train", "1.0 (sin muestreo)" if train_fraction is None else round(train_fraction, 4))
    log_info("Fraction test", "1.0 (sin muestreo)" if test_fraction is None else round(test_fraction, 4))
    log_info("Tiempo carga", f"{time.time() - load_start:.1f}s")

    log_section("2) Generacion de embeddings")
    log_info("Estrategia", "Parquet intermedio con sentence_embeddings (cortafuegos Kryo)")
    document = (
        DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
    )

    sentence_embeddings = (
        UniversalSentenceEncoder.pretrained(MODEL_NAME, MODEL_LANG)
        .setInputCols(["document"])
        .setOutputCol("sentence_embeddings")
    )
    embeddings_pipeline = Pipeline(stages=[document, sentence_embeddings])
    embeddings_model = embeddings_pipeline.fit(train.limit(1))

    spark.sparkContext.setJobDescription("SparkNLP ClassifierDL | Paso 2: Generar embeddings USE")
    embed_start = time.time()
    (
        embeddings_model.transform(train)
        .select("sentence_embeddings", "label", "label_str")
        .write.mode("overwrite").parquet(HDFS_TMP_TRAIN_ANNOT)
    )
    log_info("Train embeddings", f"✓ {HDFS_TMP_TRAIN_ANNOT}")
    (
        embeddings_model.transform(test)
        .select("sentence_embeddings", "label", "label_str")
        .write.mode("overwrite").parquet(HDFS_TMP_TEST_ANNOT)
    )
    log_info("Test embeddings", f"✓ {HDFS_TMP_TEST_ANNOT}")
    log_info("Tiempo embeddings", f"{time.time() - embed_start:.1f}s")

    log_section("2b) Reinicio controlado de Spark")
    log_info("Objetivo", "Separar la fase de embeddings de ClassifierDL.fit")
    spark.stop()
    time.sleep(5)
    spark = create_classifier_session()
    spark.sparkContext.setLogLevel("WARN")

    log_section("2c) Lectura de embeddings limpios desde HDFS")
    train = spark.read.parquet(HDFS_TMP_TRAIN_ANNOT)
    test = spark.read.parquet(HDFS_TMP_TEST_ANNOT)
    log_info("Particiones train emb", train.rdd.getNumPartitions())
    log_info("Particiones test emb", test.rdd.getNumPartitions())

    log_section("3) Construcción pipeline Spark NLP")

    classifier = (
        ClassifierDLApproach()
        .setInputCols(["sentence_embeddings"])
        .setOutputCol("category")
        .setLabelColumn("label_str")
        .setBatchSize(64)
        .setMaxEpochs(5)
        .setLr(5e-4)
        .setDropout(0.3)
    )

    log_info("Embeddings", f"UniversalSentenceEncoder.pretrained('{MODEL_NAME}')")
    log_info("Classifier", "ClassifierDLApproach")
    log_info("Batch size", 64)
    log_info("Max epochs", 5)
    log_info("Learning rate", "5e-4")
    log_info("Dropout", 0.3)
    log_info("Estrategia dataset", "Fit sobre embeddings materializados en HDFS")
    log_info("Target train rows", TRAIN_TARGET_ROWS)
    log_info("Train sample fraction", TRAIN_SAMPLE_FRACTION if TRAIN_TARGET_ROWS is not None else None)
    log_info("Target test rows", TEST_TARGET_ROWS)

    pipeline = Pipeline(stages=[classifier])

    log_section("4) Entrenamiento")
    spark.sparkContext.setJobDescription("SparkNLP | Paso 3: Entrenar ClassifierDL sobre embeddings materializados")
    train_start = time.time()
    model = pipeline.fit(train)
    train_time = time.time() - train_start
    log_info("Estado", "✓ Entrenamiento completado")
    log_info("Tiempo entrenamiento", f"{train_time:.1f}s")

    log_section("5) Evaluación")
    spark.sparkContext.setJobDescription("SparkNLP | Paso 4: Evaluacion en test set")
    eval_start = time.time()
    (
        model.transform(test)
        .withColumn("prediction_str", F.col("category").getItem(0).getField("result"))
        .withColumn("prediction", F.col("prediction_str").cast("double"))
        .withColumn("score", F.when(F.col("prediction") == 1.0, F.lit(1.0)).otherwise(F.lit(0.0)))
        .select("label", "prediction", "prediction_str", "score")
        .write.mode("overwrite").parquet(HDFS_TMP_PRED)
    )
    log_info("Predicciones eval", f"✓ {HDFS_TMP_PRED}")
    log_info("Modo metricas", "Evaluadores Spark JVM-only (sin collectToPython)")
    predictions = spark.read.parquet(HDFS_TMP_PRED)
    metrics = compute_metrics(predictions)
    metrics.update(
        {
            "model": "SparkNLP_USE_ClassifierDL",
            "embedding_model": MODEL_NAME,
            "train_time_s": round(train_time, 1),
            "eval_time_s": round(time.time() - eval_start, 1),
            "total_time_s": round(time.time() - total_start, 1),
            "train_rows": None if train_rows is None else int(train_rows),
            "test_rows": None if test_rows is None else int(test_rows),
            "train_fraction": None if train_fraction is None else round(train_fraction, 4),
            "test_fraction": None if test_fraction is None else round(test_fraction, 4),
            "dataset_strategy": "full_dataset_driver_side_classifierdl" if train_rows is None else "stratified_sample_for_driver_side_classifierdl",
        }
    )

    log_section("6) Resultados")
    for key, value in metrics.items():
        log_info(key, value)

    log_section("7) Persistencia en HDFS")
    model.write().overwrite().save("hdfs://namenode:9000/models/spark_nlp_classifierdl/")
    log_info("Modelo", "✓ Guardado en hdfs:///models/spark_nlp_classifierdl/")

    metrics_path = "hdfs://namenode:9000/metrics/spark_nlp_classifierdl/"
    sc = spark.sparkContext
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(metrics_path), True)
    metrics_rdd = sc.parallelize([json.dumps(metrics, indent=2)])
    metrics_rdd.saveAsTextFile(metrics_path)
    log_info("Métricas", "✓ Guardadas en hdfs:///metrics/spark_nlp_classifierdl/")

    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_TRAIN_ANNOT), True)
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_TEST_ANNOT), True)
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_PRED), True)

    spark.stop()


if __name__ == "__main__":
    main()
