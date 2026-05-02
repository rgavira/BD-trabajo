"""
03_word2vec_mlp.py — Word2Vec + GlobalAveragePooling + MLP con Spark MLlib

Replica el modelo Nivel 1 de AINE usando solo Spark MLlib nativo:
  Tokenizer → StopWordsRemover → Word2Vec → MultilayerPerceptronClassifier

Nota sobre Word2Vec en Spark:
  El Word2Vec de Spark MLlib promedia los vectores de todas las palabras
  del documento para producir un único vector de dimensión fija. Esto es
  exactamente lo que hace GlobalAveragePooling en Keras/PyTorch. No es
  necesario ningún paso adicional.

Pipeline:
  text → [Tokenizer] → words → [StopWordsRemover] → filtered →
  [Word2Vec(128d)] → features (GlobalAvgPool implícito) →
  [MLP: 128→64→32→2] → predicción

Resultados se guardan en:
  hdfs:///models/word2vec_mlp/     ← modelo serializado
  hdfs:///metrics/word2vec_mlp/    ← métricas JSON
"""

import json
import time
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


def log_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def log_info(label, value):
    print(f"  {label:<24} {value}")


def log_timer(label, started_at):
    log_info(label, f"{time.time() - started_at:.1f}s")


def create_spark_session():
    return (
        SparkSession.builder
        .appName("ExplicitLyrics_Word2Vec_MLP")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .getOrCreate()
    )


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    total_start = time.time()

    # ── Cargar datos desde HDFS ───────────────────────────────────────────────
    log_section("1) Carga de datos procesados desde HDFS")
    spark.sparkContext.setJobDescription("Word2Vec+MLP | Paso 1: Cargar train/test desde HDFS")
    load_start = time.time()
    train = (
        spark.read.parquet("hdfs://namenode:9000/data/processed/train/")
        .select("text", "label")
        .repartition(4)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    test = (
        spark.read.parquet("hdfs://namenode:9000/data/processed/test/")
        .select("text", "label")
        .repartition(4)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    log_info("Train rows", f"{train.count():,}")
    log_info("Test rows", f"{test.count():,}")
    log_info("Particiones train", train.rdd.getNumPartitions())
    log_info("Particiones test", test.rdd.getNumPartitions())
    log_timer("Tiempo carga+cache", load_start)

    # ── Pipeline MLlib ────────────────────────────────────────────────────────
    log_section("2) Construcción de la pipeline MLlib")

    tokenizer = Tokenizer(inputCol="text", outputCol="words")

    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    # Word2Vec: entrena embeddings desde cero sobre el corpus.
    # vectorSize=128 → misma dimensión que el Nivel 1 de AINE.
    # El resultado ya es el average de todos los vectores de palabras
    # → equivalente exacto a GlobalAveragePooling.
    word2vec = Word2Vec(
        vectorSize=128,
        minCount=10,         # ignora palabras que aparecen < 10 veces (vocab más pequeño → menos OOM)
        numPartitions=4,     # paralelismo para el training de Word2Vec
        maxIter=1,           # default de Spark, fijado explícitamente para trazabilidad
        windowSize=5,        # default de Spark, fijado explícitamente para trazabilidad
        inputCol="filtered_words",
        outputCol="features",
        seed=42,
    )
    log_info("Word2Vec.vectorSize", 128)
    log_info("Word2Vec.minCount", 10)
    log_info("Word2Vec.maxIter", 1)
    log_info("Word2Vec.windowSize", 5)
    log_info("Word2Vec.numPartitions", 4)

    # MLP: capas [128, 64, 32, 2]
    # La última capa tiene 2 neuronas (softmax binario: clase 0 y clase 1).
    # Equivale a Dense(64) → Dense(32) → Dense(1, sigmoid) de Keras.
    mlp = MultilayerPerceptronClassifier(
        featuresCol="features",
        labelCol="label",
        layers=[128, 64, 32, 2],
        maxIter=100,
        blockSize=128,
        stepSize=0.03,
        seed=42,
    )
    log_info("MLP.layers", "[128, 64, 32, 2]")
    log_info("MLP.maxIter", 100)
    log_info("MLP.blockSize", 128)
    log_info("MLP.stepSize", 0.03)

    pipeline = Pipeline(stages=[tokenizer, remover, word2vec, mlp])

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    log_section("3) Entrenamiento")
    print("  Word2Vec puede tardar varios minutos; el MLP va dentro de la misma fit().")
    spark.sparkContext.setJobDescription("Word2Vec+MLP | Paso 2: Entrenamiento Word2Vec + GlobalAvgPool + MLP")
    train_start = time.time()
    model = pipeline.fit(train)
    train_time = time.time() - train_start
    log_info("Estado", "✓ Entrenamiento completado")
    log_info("Tiempo entrenamiento", f"{train_time:.1f}s")

    # ── Evaluación ────────────────────────────────────────────────────────────
    log_section("4) Evaluación en test")
    spark.sparkContext.setJobDescription("Word2Vec+MLP | Paso 3: Evaluacion en test set")
    eval_start = time.time()
    predictions = (
        model.transform(test)
        .select("label", "prediction", "rawPrediction")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    predictions.count()

    auc   = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(predictions)

    f1_weighted = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    ).evaluate(predictions)

    # Confusion matrix en una sola pasada para evitar 4 scans completos del test set
    confusion = {
        (int(row["label"]), int(row["prediction"])): int(row["count"])
        for row in predictions.groupBy("label", "prediction").count().collect()
    }
    tp = confusion.get((1, 1), 0)
    fp = confusion.get((0, 1), 0)
    fn = confusion.get((1, 0), 0)
    tn = confusion.get((0, 0), 0)

    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_1  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_explicit = (2 * prec_1 * rec_1 / (prec_1 + rec_1)) if (prec_1 + rec_1) > 0 else 0.0

    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_not_explicit = (2 * prec_0 * rec_0 / (prec_0 + rec_0)) if (prec_0 + rec_0) > 0 else 0.0

    f1_macro = (f1_explicit + f1_not_explicit) / 2

    acc   = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(predictions)

    prec  = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    ).evaluate(predictions)

    rec   = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    ).evaluate(predictions)
    eval_time = time.time() - eval_start

    metrics = {
        "model":              "Word2Vec_GlobalAvgPool_MLP_SparkMLlib",
        "auc_roc":            round(auc,             4),
        "f1_weighted":        round(f1_weighted,      4),
        "f1_macro":           round(f1_macro,         4),
        "f1_explicit":        round(f1_explicit,      4),
        "f1_not_explicit":    round(f1_not_explicit,  4),
        "accuracy":           round(acc,              4),
        "precision_weighted": round(prec,             4),
        "recall_weighted":    round(rec,              4),
        "train_time_s":       round(train_time,       1),
        "eval_time_s":        round(eval_time,        1),
        "total_time_s":       round(time.time() - total_start, 1),
    }

    log_section("5) Resultados")
    for k, v in metrics.items():
        log_info(k, v)

    # ── Guardar modelo y métricas en HDFS ─────────────────────────────────────
    log_section("6) Persistencia en HDFS")
    model.write().overwrite().save("hdfs://namenode:9000/models/word2vec_mlp/")
    log_info("Modelo", "✓ Guardado en hdfs:///models/word2vec_mlp/")

    metrics_path = "hdfs://namenode:9000/metrics/word2vec_mlp/"
    sc = spark.sparkContext
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(metrics_path), True)
    metrics_rdd = sc.parallelize([json.dumps(metrics, indent=2)])
    metrics_rdd.saveAsTextFile(metrics_path)
    log_info("Métricas", "✓ Guardadas en hdfs:///metrics/word2vec_mlp/")
    log_info("Tiempo total", f"{metrics['total_time_s']:.1f}s")

    predictions.unpersist()
    train.unpersist()
    test.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
