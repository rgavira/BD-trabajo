"""
04c_spark_nlp_sent_bert_mllib_mlp.py - Pipeline distribuido: BertSentenceEmbeddings + MLP MLlib

Motivacion:
  El nivel 2b (04b) uso BertEmbeddings token-level + SentenceEmbeddings(AVERAGE).
  Ese promedio es naive: el modelo BERT no fue entrenado para que sus embeddings de token
  sean promediables y el vector resultante esta contaminado por stopwords y tokens frecuentes.

  BertSentenceEmbeddings usa modelos con prefijo sent_* del hub de Spark NLP, que fueron
  fine-tuneados especificamente para producir representaciones de frase utiles (estilo SBERT).
  Internamente usan el token [CLS] o pooling entrenado para tareas sentence-level, no
  promedio aritmetico de tokens.

  La hipotesis es que sent_small_bert_L4_512 produce mejores vectores de frase que
  small_bert_L4_512 + AVERAGE, manteniendo el mismo coste computacional aproximado
  y la misma dimension (512d) para comparabilidad directa con USE y el nivel 2b.

Pipeline:
  text -> DocumentAssembler -> BertSentenceEmbeddings(sent_small_bert_L4_512)
       -> Parquet intermedio (array<float> 512d)
       -> DenseVector -> MultilayerPerceptronClassifier [512,256,128,2]

  Nota: BertSentenceEmbeddings maneja su propia tokenizacion internamente.
  No se necesita un paso Tokenizer separado (a diferencia del nivel 2b).

Ejecucion: prueba limitada a 1000 filas de train y test.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import json
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.functions import array_to_vector, vector_to_array

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import BertSentenceEmbeddings


MODEL_NAME = "sent_small_bert_L4_512"
MODEL_LANG = "en"
EMBED_DIM = 512
#TRAIN_LIMIT_ROWS = 1000
#TEST_LIMIT_ROWS = 1000

HDFS_TMP_TRAIN = "hdfs://namenode:9000/tmp/sent_bert_embeddings_train/"
HDFS_TMP_TEST = "hdfs://namenode:9000/tmp/sent_bert_embeddings_test/"
HDFS_TMP_TRAIN_FEAT = "hdfs://namenode:9000/tmp/sent_bert_features_train/"
HDFS_TMP_TEST_FEAT = "hdfs://namenode:9000/tmp/sent_bert_features_test/"
HDFS_TMP_PRED = "hdfs://namenode:9000/tmp/sent_bert_predictions_eval/"


def log_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def log_info(label, value):
    print(f"  {label:<26} {value}")


def create_spark_nlp_session():
    return (
        SparkSession.builder
        .appName("ExplicitLyrics_SentBERT_Embeddings")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "2000M")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def create_spark_mllib_session():
    return (
        SparkSession.builder
        .appName("ExplicitLyrics_SentBERT_MLlib_MLP")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )


def compute_metrics(predictions):
    ev = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    f1_weighted = ev.setMetricName("f1").evaluate(predictions)
    acc = ev.setMetricName("accuracy").evaluate(predictions)
    prec = ev.setMetricName("weightedPrecision").evaluate(predictions)
    rec = ev.setMetricName("weightedRecall").evaluate(predictions)
    f1_explicit = ev.setMetricName("fMeasureByLabel").setMetricLabel(1.0).evaluate(predictions)
    f1_not_explicit = ev.setMetricName("fMeasureByLabel").setMetricLabel(0.0).evaluate(predictions)
    f1_macro = (f1_explicit + f1_not_explicit) / 2.0

    auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="score", metricName="areaUnderROC"
    ).evaluate(predictions)

    return {
        "f1_weighted": round(f1_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_explicit": round(f1_explicit, 4),
        "f1_not_explicit": round(f1_not_explicit, 4),
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "auc_roc": round(auc, 4),
    }


def main():
    total_start = time.time()
    spark = create_spark_nlp_session()
    spark.sparkContext.setLogLevel("WARN")

    log_section("1) Carga de train/test desde HDFS (limitado)")
    spark.sparkContext.setJobDescription("SentBERT+MLlib | Paso 1: Cargar datos")
    load_start = time.time()
    train = (
        spark.read.parquet("hdfs://namenode:9000/data/processed/train/")
        .select("text", "label")
        #.limit(TRAIN_LIMIT_ROWS)
        .repartition(4)
    )
    test = (
        spark.read.parquet("hdfs://namenode:9000/data/processed/test/")
        .select("text", "label")
        #.limit(TEST_LIMIT_ROWS)
        .repartition(4)
    )
    log_info("Tiempo carga", f"{time.time() - load_start:.1f}s")
    #log_info("Train limitado", TRAIN_LIMIT_ROWS)
    #log_info("Test limitado", TEST_LIMIT_ROWS)

    log_section("2) Pipeline Spark NLP - BertSentenceEmbeddings en workers")
    document = DocumentAssembler().setInputCol("text").setOutputCol("document")
    sent_bert = (
        BertSentenceEmbeddings.pretrained(MODEL_NAME, MODEL_LANG)
        .setInputCols(["document"])
        .setOutputCol("sentence_embeddings")
        .setCaseSensitive(False)
    )
    nlp_pipeline = Pipeline(stages=[document, sent_bert])
    log_info("Modelo embeddings", f"BertSentenceEmbeddings.pretrained('{MODEL_NAME}')")
    log_info("Tipo",              "Sentence-level (fine-tuned, sin pooling manual)")
    log_info("Dimension",         EMBED_DIM)
    log_info("vs nivel 2b",       "2b usa token AVERAGE naive; este usa repr. de frase entrenada")
    log_info("vs USE",            "ambos producen 512d; este es BERT contextual, USE es encoder de frase")

    log_section("3) Generar embeddings y materializar a HDFS")
    log_info("Estrategia", "Parquet intermedio (cortafuegos Kryo)")
    spark.sparkContext.setJobDescription("SentBERT+MLlib | Paso 2: Generar embeddings")
    embed_start = time.time()

    nlp_model = nlp_pipeline.fit(train)

    def extract_and_write(df, hdfs_path):
        (
            nlp_model.transform(df)
            .withColumn("embeddings", F.col("sentence_embeddings").getItem(0)["embeddings"])
            .select("embeddings", "label")
            .write.mode("overwrite").parquet(hdfs_path)
        )

    extract_and_write(train, HDFS_TMP_TRAIN)
    log_info("Train embeddings", f"OK {HDFS_TMP_TRAIN}")

    extract_and_write(test, HDFS_TMP_TEST)
    log_info("Test embeddings", f"OK {HDFS_TMP_TEST}")

    embed_time = time.time() - embed_start
    log_info("Tiempo embeddings", f"{embed_time:.1f}s")

    log_section("3b) Reinicio controlado de Spark")
    log_info("Objetivo", "Aislar Spark NLP/Kryo de la fase MLlib")
    spark.stop()
    time.sleep(5)
    spark = create_spark_mllib_session()
    spark.sparkContext.setLogLevel("WARN")

    log_section("4) Lectura de embeddings limpios + conversion a Vector")
    spark.sparkContext.setJobDescription("SentBERT+MLlib | Paso 3: Leer embeddings y convertir")

    train_raw = spark.read.parquet(HDFS_TMP_TRAIN)
    test_raw = spark.read.parquet(HDFS_TMP_TEST)

    train_raw = (
        train_raw
        .withColumn("emb_size", F.size("embeddings"))
        .where(F.col("emb_size") == EMBED_DIM)
        .drop("emb_size")
    )
    test_raw = (
        test_raw
        .withColumn("emb_size", F.size("embeddings"))
        .where(F.col("emb_size") == EMBED_DIM)
        .drop("emb_size")
    )

    (
        train_raw
        .withColumn("features", array_to_vector(F.col("embeddings")))
        .select("features", "label")
        .write.mode("overwrite").parquet(HDFS_TMP_TRAIN_FEAT)
    )
    (
        test_raw
        .withColumn("features", array_to_vector(F.col("embeddings")))
        .select("features", "label")
        .write.mode("overwrite").parquet(HDFS_TMP_TEST_FEAT)
    )
    log_info("Train features", f"OK {HDFS_TMP_TRAIN_FEAT}")
    log_info("Test features",  f"OK {HDFS_TMP_TEST_FEAT}")

    train_feat = spark.read.parquet(HDFS_TMP_TRAIN_FEAT).select("features", "label")
    test_feat = spark.read.parquet(HDFS_TMP_TEST_FEAT).select("features", "label")

    log_section("5) Clasificador MLlib MLP")
    mlp = MultilayerPerceptronClassifier(
        featuresCol="features",
        labelCol="label",
        layers=[EMBED_DIM, 256, 128, 2],
        blockSize=128,
        maxIter=100,
        seed=42,
    )
    log_info("Arquitectura",    f"[{EMBED_DIM}, 256, 128, 2]")
    log_info("Solver",          "L-BFGS distribuido (Spark MLlib)")
    log_info("Max iteraciones", 100)

    log_section("6) Entrenamiento MLP MLlib (L-BFGS distribuido)")
    spark.sparkContext.setJobDescription("SentBERT+MLlib | Paso 4: Entrenar MLP")
    train_start = time.time()
    mlp_model = mlp.fit(train_feat)
    train_time = time.time() - train_start
    log_info("Estado",              "OK Entrenamiento completado")
    log_info("Tiempo entrenamiento", f"{train_time:.1f}s")

    log_section("7) Evaluacion")
    spark.sparkContext.setJobDescription("SentBERT+MLlib | Paso 5: Evaluar en test set")
    eval_start = time.time()
    (
        mlp_model.transform(test_feat)
        .withColumn("score", vector_to_array(F.col("probability")).getItem(1))
        .select("label", "prediction", "score")
        .write.mode("overwrite").parquet(HDFS_TMP_PRED)
    )

    predictions = spark.read.parquet(HDFS_TMP_PRED)
    metrics = compute_metrics(predictions)
    metrics.update({
        "model":        "SparkNLP_SentBERT_MLlib_MLP",
        "embedding":    MODEL_NAME,
        "embedding_dim": EMBED_DIM,
        "mlp_layers":   [EMBED_DIM, 256, 128, 2],
        #"train_rows":   TRAIN_LIMIT_ROWS,
        #"test_rows":    TEST_LIMIT_ROWS,
        "embed_time_s": round(embed_time, 1),
        "train_time_s": round(train_time, 1),
        "eval_time_s":  round(time.time() - eval_start, 1),
        "total_time_s": round(time.time() - total_start, 1),
    })

    log_section("8) Resultados")
    for k, v in metrics.items():
        log_info(k, v)

    log_section("9) Persistencia en HDFS")
    mlp_model.write().overwrite().save("hdfs://namenode:9000/models/spark_nlp_sent_bert_mllib_mlp/")
    log_info("Modelo", "OK hdfs:///models/spark_nlp_sent_bert_mllib_mlp/")

    metrics_path = "hdfs://namenode:9000/metrics/spark_nlp_sent_bert_mllib_mlp/"
    sc = spark.sparkContext
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(metrics_path), True)
    sc.parallelize([json.dumps(metrics, indent=2)]).saveAsTextFile(metrics_path)
    log_info("Metricas", "OK hdfs:///metrics/spark_nlp_sent_bert_mllib_mlp/")

    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_TRAIN), True)
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_TEST), True)
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_TRAIN_FEAT), True)
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_TEST_FEAT), True)
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_TMP_PRED), True)
    spark.stop()


if __name__ == "__main__":
    main()
