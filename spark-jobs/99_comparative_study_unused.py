"""
99_comparative_study_unused.py — Estudio comparativo de algoritmos MLlib sobre TF-IDF

Motivación:
  La Task 3 replicó el modelo Word2Vec+MLP de AINE/MLE y tardó ~10 minutos en
  500k canciones. La pregunta natural es: ¿qué otros algoritmos ofrece Spark
  MLlib? ¿Son más rápidos? ¿Más precisos? ¿Merece la pena la complejidad?

  Todos los modelos usan las MISMAS features (TF-IDF) para comparación justa.
  Word2Vec+MLP (Task 3) actúa como línea base.

Algoritmos (uno por familia):
  1. Naive Bayes          — probabilístico, el más rápido en texto
  2. Logistic Regression  — lineal, interpretable, muy sólido
  3. Linear SVM           — lineal, clásico en clasificación de texto
  4. Random Forest        — ensemble de árboles, robusto al ruido
  5. GBT                  — gradient boosting, suele ser el más preciso

Pipeline de features (compartido por todos):
  Tokenizer → StopWordsRemover → HashingTF (20k) → IDF

Resultados: hdfs:///metrics/comparative_study/
"""

import json
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import (
    NaiveBayes,
    LogisticRegression,
    LinearSVC,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


NUM_FEATURES = 20_000   # dimensión del espacio TF-IDF
MIN_DOC_FREQ = 5        # IDF ignora términos que aparecen en < 5 documentos


def create_spark_session():
    return (
        SparkSession.builder
        .appName("ExplicitLyrics_ComparativeStudy")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .getOrCreate()
    )


def tfidf_stages():
    """Pipeline de features compartido por todos los clasificadores."""
    return [
        Tokenizer(inputCol="text", outputCol="words"),
        StopWordsRemover(inputCol="words", outputCol="filtered_words"),
        HashingTF(inputCol="filtered_words", outputCol="raw_features",
                  numFeatures=NUM_FEATURES),
        IDF(inputCol="raw_features", outputCol="features",
            minDocFreq=MIN_DOC_FREQ),
    ]


def compute_metrics(predictions, has_raw_pred=True):
    """Calcula F1 weighted, macro, por clase, accuracy y AUC."""
    f1_weighted = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    ).evaluate(predictions)

    acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(predictions)

    auc = None
    if has_raw_pred:
        try:
            auc = BinaryClassificationEvaluator(
                labelCol="label", rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            ).evaluate(predictions)
        except Exception:
            pass

    # Confusion matrix manual → F1 por clase y macro
    tp = predictions.filter((F.col("label") == 1) & (F.col("prediction") == 1)).count()
    fp = predictions.filter((F.col("label") == 0) & (F.col("prediction") == 1)).count()
    fn = predictions.filter((F.col("label") == 1) & (F.col("prediction") == 0)).count()
    tn = predictions.filter((F.col("label") == 0) & (F.col("prediction") == 0)).count()

    p1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_explicit = (2 * p1 * r1 / (p1 + r1)) if (p1 + r1) > 0 else 0.0

    p0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    r0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_not_explicit = (2 * p0 * r0 / (p0 + r0)) if (p0 + r0) > 0 else 0.0

    f1_macro = (f1_explicit + f1_not_explicit) / 2

    return {
        "f1_weighted":     round(f1_weighted,     4),
        "f1_macro":        round(f1_macro,         4),
        "f1_explicit":     round(f1_explicit,      4),
        "f1_not_explicit": round(f1_not_explicit,  4),
        "accuracy":        round(acc,              4),
        "auc_roc":         round(auc, 4) if auc is not None else None,
    }


def run_model(spark, name, classifier, train, test, has_raw_pred=True):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    pipeline = Pipeline(stages=tfidf_stages() + [classifier])

    spark.sparkContext.setJobDescription(f"ComparativeStudy | {name} | Entrenamiento")
    t0 = time.time()
    model = pipeline.fit(train)
    train_time = round(time.time() - t0, 1)
    print(f"  Tiempo entrenamiento : {train_time}s")

    spark.sparkContext.setJobDescription(f"ComparativeStudy | {name} | Evaluacion")
    predictions = model.transform(test)

    metrics = compute_metrics(predictions, has_raw_pred=has_raw_pred)
    metrics["model"]        = name
    metrics["train_time_s"] = train_time

    print(f"  F1 weighted          : {metrics['f1_weighted']}")
    print(f"  F1 macro             : {metrics['f1_macro']}")
    print(f"  F1 clase explícita   : {metrics['f1_explicit']}")
    print(f"  Accuracy             : {metrics['accuracy']}")
    if metrics["auc_roc"] is not None:
        print(f"  AUC-ROC              : {metrics['auc_roc']}")

    return metrics


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # ── Cargar datos desde HDFS ───────────────────────────────────────────────
    print("=" * 60)
    print("Cargando train/test desde HDFS...")
    spark.sparkContext.setJobDescription("ComparativeStudy | Cargar train/test desde HDFS")
    train = spark.read.parquet("hdfs://namenode:9000/data/processed/train/") \
                .withColumn("label", F.col("label").cast("double"))
    test  = spark.read.parquet("hdfs://namenode:9000/data/processed/test/") \
                .withColumn("label", F.col("label").cast("double"))
    print(f"  Train: {train.count():,}  |  Test: {test.count():,}")

    results = []

    # ── BLOQUE 1: Naive Bayes ─────────────────────────────────────────────────
    # Probabilístico, asume independencia entre términos.
    # El más rápido. Requiere features no negativas (TF-IDF lo cumple).
    nb = NaiveBayes(featuresCol="features", labelCol="label", smoothing=1.0)
    results.append(run_model(spark, "NaiveBayes_TFIDF", nb, train, test))

    # ── BLOQUE 2: Logistic Regression ────────────────────────────────────────
    # Modelo lineal, muy interpretable. Referencia estándar en clasificación de texto.
    lr = LogisticRegression(
        featuresCol="features", labelCol="label",
        maxIter=100, regParam=0.01, elasticNetParam=0.0,
    )
    results.append(run_model(spark, "LogisticRegression_TFIDF", lr, train, test))

    # ── BLOQUE 3: Linear SVM ─────────────────────────────────────────────────
    # Maximiza el margen entre clases. Muy bueno en espacios de alta dimensión.
    # No produce probabilidades → AUC no disponible.
    svm = LinearSVC(
        featuresCol="features", labelCol="label",
        maxIter=100, regParam=0.01,
    )
    results.append(run_model(spark, "LinearSVM_TFIDF", svm, train, test,
                             has_raw_pred=False))

    # ── BLOQUE 4: Random Forest ──────────────────────────────────────────────
    # Ensemble de árboles de decisión. Robusto al ruido y al desbalanceo de clases.
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="label",
        numTrees=100, maxDepth=10, seed=42,
    )
    results.append(run_model(spark, "RandomForest_TFIDF", rf, train, test))

    # ── BLOQUE 5: Gradient Boosted Trees ─────────────────────────────────────
    # Boosting secuencial: cada árbol corrige los errores del anterior.
    # Suele ser el más preciso de los métodos clásicos. Solo clasificación binaria.
    gbt = GBTClassifier(
        featuresCol="features", labelCol="label",
        maxIter=50, maxDepth=5, seed=42,
    )
    results.append(run_model(spark, "GBT_TFIDF", gbt, train, test,
                             has_raw_pred=False))

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("RESUMEN COMPARATIVO (referencia: Word2Vec+MLP Task 3 ~F1=0.85)")
    print("=" * 75)
    header = f"{'Modelo':<28} {'F1_w':>6} {'F1_mac':>7} {'F1_exp':>7} {'Acc':>6} {'AUC':>6} {'Tiempo':>8}"
    print(header)
    print("-" * 75)
    for m in results:
        auc_str = f"{m['auc_roc']:.4f}" if m["auc_roc"] is not None else "  N/A"
        print(
            f"{m['model']:<28} {m['f1_weighted']:>6.4f} {m['f1_macro']:>7.4f} "
            f"{m['f1_explicit']:>7.4f} {m['accuracy']:>6.4f} {auc_str:>6} "
            f"{m['train_time_s']:>7.1f}s"
        )

    # ── Guardar métricas en HDFS ──────────────────────────────────────────────
    spark.sparkContext.setJobDescription("ComparativeStudy | Guardar métricas en HDFS")
    metrics_path = "hdfs://namenode:9000/metrics/comparative_study/"
    sc = spark.sparkContext
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(metrics_path), True)
    metrics_rdd = sc.parallelize([json.dumps(results, indent=2)])
    metrics_rdd.saveAsTextFile(metrics_path)
    print("\n  ✓ Métricas guardadas en hdfs:///metrics/comparative_study/")

    spark.stop()


if __name__ == "__main__":
    main()

