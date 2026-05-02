"""
06_tensorflow_text_baselines.py - Baselines AINE con Spark + TensorFlow

Idea:
  - Spark se usa para leer train/test desde HDFS y preparar el dataset
    de forma distribuida.
  - El entrenamiento de TensorFlow se hace en CPU sobre el driver.

Esto nos permite responder a una pregunta distinta respecto a MLlib y Spark NLP:
  "Que pasa si mantenemos Spark como capa de datos, pero entrenamos con un
   framework deep learning clasico para replicar mejor los baselines de AINE?"

Modelos incluidos:
  1. Embedding propio + GlobalAveragePooling1D + MLP
  2. Embedding propio + CNN1D
  3. Word2Vec del corpus + CNN1D

Salidas:
  - Modelos Keras locales: /opt/models/tensorflow_text_baselines/
  - Metricas JSON en HDFS: hdfs:///metrics/tensorflow_text_baselines/
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score


SEED = 42
TRAIN_TARGET_ROWS = None
TEST_TARGET_ROWS = None
MAX_VOCAB_SIZE = 30_000
MAX_SEQUENCE_LENGTH = 256
EMBEDDING_DIM = 64
WORD2VEC_DIM = 128
BATCH_SIZE = 256
EPOCHS = 5
LOCAL_OUTPUT_DIR = Path("/opt/models/tensorflow_text_baselines")
HDFS_METRICS_DIR = "hdfs://namenode:9000/metrics/tensorflow_text_baselines/"


def log_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def log_info(label, value):
    print(f"  {label:<28} {value}")


def create_spark_session():
    return (
        SparkSession.builder
        .appName("ExplicitLyrics_TensorFlow_Baselines")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .getOrCreate()
    )


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_tensorflow():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    tf.get_logger().setLevel("ERROR")
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() or 1))


def sample_split(df, target_rows, seed):
    current_rows = df.count()
    if target_rows is None or current_rows <= target_rows:
        return df, current_rows, 1.0

    fraction = min(1.0, target_rows / current_rows)
    sampled = df.sample(withReplacement=False, fraction=fraction, seed=seed)
    return sampled, sampled.count(), fraction


def load_splits_for_tensorflow(spark):
    train = spark.read.parquet("hdfs://namenode:9000/data/processed/train/").select("text", "label")
    test = spark.read.parquet("hdfs://namenode:9000/data/processed/test/").select("text", "label")

    sampled_train, train_rows, train_fraction = sample_split(train, TRAIN_TARGET_ROWS, SEED)
    sampled_test, test_rows, test_fraction = sample_split(test, TEST_TARGET_ROWS, SEED)

    train_pd = sampled_train.withColumn("text", F.coalesce(F.col("text"), F.lit(""))).toPandas()
    test_pd = sampled_test.withColumn("text", F.coalesce(F.col("text"), F.lit(""))).toPandas()

    return train_pd, test_pd, train_rows, test_rows, train_fraction, test_fraction


def prepare_tokenizer(train_texts):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=MAX_VOCAB_SIZE,
        oov_token="[UNK]",
    )
    tokenizer.fit_on_texts(train_texts)
    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
    return tokenizer, vocab_size


def texts_to_padded_sequences(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )


def build_avgpool_mlp(vocab_size):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,)),
            tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="embedding_avgpool_mlp",
    )


def build_embedding_cnn(vocab_size):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,)),
            tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM),
            tf.keras.layers.Conv1D(128, 5, activation="relu"),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="embedding_cnn1d",
    )


def build_word2vec_cnn(vocab_size, embedding_matrix):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,)),
            tf.keras.layers.Embedding(
                vocab_size,
                WORD2VEC_DIM,
                weights=[embedding_matrix],
                trainable=False,
            ),
            tf.keras.layers.Conv1D(128, 5, activation="relu"),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="word2vec_cnn1d",
    )


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=1,
            restore_best_weights=True,
        )
    ]


def compute_class_weight(y_train):
    values, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    return {
        int(label): float(total / (len(values) * count))
        for label, count in zip(values, counts)
    }


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype("int32")
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_weighted": round(float(weighted[0]), 4),
        "recall_weighted": round(float(weighted[1]), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_not_explicit": round(float(per_class[2][0]), 4),
        "f1_explicit": round(float(per_class[2][1]), 4),
        "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def train_word2vec_embedding_matrix(train_texts, tokenizer, vocab_size):
    tokenized_train = [text.split() for text in train_texts]

    w2v = Word2Vec(
        sentences=tokenized_train,
        vector_size=WORD2VEC_DIM,
        window=5,
        min_count=5,
        workers=max(1, (os.cpu_count() or 1) - 1),
        sg=1,
        epochs=5,
        seed=SEED,
    )

    embedding_matrix = np.zeros((vocab_size, WORD2VEC_DIM), dtype="float32")
    for word, idx in tokenizer.word_index.items():
        if idx >= vocab_size:
            continue
        if word in w2v.wv:
            embedding_matrix[idx] = w2v.wv[word]

    return w2v, embedding_matrix


def fit_and_evaluate(model_name, model, x_train, y_train, x_test, y_test, class_weight):
    started_at = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=build_callbacks(),
        class_weight=class_weight,
        verbose=2,
    )
    train_time = time.time() - started_at

    y_prob = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    metrics = compute_metrics(y_test, y_prob)
    metrics.update(
        {
            "model": model_name,
            "epochs_ran": len(history.history.get("loss", [])),
            "train_time_s": round(train_time, 1),
        }
    )
    return metrics, history


def save_local_artifacts(tokenizer, histories, metrics_summary, word2vec_model=None):
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (LOCAL_OUTPUT_DIR / "tokenizer_config.json").write_text(tokenizer.to_json(), encoding="utf-8")
    (LOCAL_OUTPUT_DIR / "results_summary.json").write_text(
        json.dumps(metrics_summary, indent=2),
        encoding="utf-8",
    )

    for model_name, history in histories.items():
        (LOCAL_OUTPUT_DIR / f"{model_name}_history.json").write_text(
            json.dumps(history.history, indent=2),
            encoding="utf-8",
        )

    if word2vec_model is not None:
        word2vec_model.save(str(LOCAL_OUTPUT_DIR / "word2vec_corpus.model"))


def save_hdfs_metrics(spark, payload):
    sc = spark.sparkContext
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(HDFS_METRICS_DIR), True)
    sc.parallelize([json.dumps(payload, indent=2)]).saveAsTextFile(HDFS_METRICS_DIR)


def main():
    configure_tensorflow()
    set_seeds(SEED)

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    total_start = time.time()

    log_section("1) Carga completa desde HDFS para TensorFlow")
    load_start = time.time()
    train_pd, test_pd, train_rows, test_rows, train_fraction, test_fraction = load_splits_for_tensorflow(spark)
    log_info("Train rows usados", f"{train_rows:,}")
    log_info("Test rows usados", f"{test_rows:,}")
    log_info("Fraction train", round(train_fraction, 4))
    log_info("Fraction test", round(test_fraction, 4))
    log_info("Tiempo carga", f"{time.time() - load_start:.1f}s")

    train_texts = train_pd["text"].fillna("").astype(str).tolist()
    test_texts = test_pd["text"].fillna("").astype(str).tolist()
    y_train = train_pd["label"].astype("int32").to_numpy()
    y_test = test_pd["label"].astype("int32").to_numpy()

    log_section("2) Preparacion local para Keras")
    tokenizer, vocab_size = prepare_tokenizer(train_texts)
    x_train = texts_to_padded_sequences(tokenizer, train_texts)
    x_test = texts_to_padded_sequences(tokenizer, test_texts)
    class_weight = compute_class_weight(y_train)
    log_info("Vocabulario efectivo", vocab_size)
    log_info("Longitud maxima", MAX_SEQUENCE_LENGTH)
    log_info("Batch size", BATCH_SIZE)
    log_info("Epochs max", EPOCHS)
    log_info("Class weight", class_weight)

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    histories = {}

    log_section("3) Baseline AINE - Embedding propio + GlobalAvgPool + MLP")
    avgpool_model = compile_model(build_avgpool_mlp(vocab_size))
    avgpool_metrics, avgpool_history = fit_and_evaluate(
        "AINE_Baseline_Embedding_GlobalAvgPool_MLP",
        avgpool_model,
        x_train,
        y_train,
        x_test,
        y_test,
        class_weight,
    )
    avgpool_model.save(str(LOCAL_OUTPUT_DIR / "baseline_embedding_avgpool_mlp.keras"))
    histories["baseline_embedding_avgpool_mlp"] = avgpool_history
    results.append(avgpool_metrics)
    for key, value in avgpool_metrics.items():
        log_info(key, value)

    log_section("4) Nivel 1 AINE - Embedding propio + CNN1D")
    embedding_cnn_model = compile_model(build_embedding_cnn(vocab_size))
    embedding_cnn_metrics, embedding_cnn_history = fit_and_evaluate(
        "AINE_Level1_Embedding_CNN1D",
        embedding_cnn_model,
        x_train,
        y_train,
        x_test,
        y_test,
        class_weight,
    )
    embedding_cnn_model.save(str(LOCAL_OUTPUT_DIR / "level1_embedding_cnn1d.keras"))
    histories["level1_embedding_cnn1d"] = embedding_cnn_history
    results.append(embedding_cnn_metrics)
    for key, value in embedding_cnn_metrics.items():
        log_info(key, value)

    log_section("5) Nivel 2 AINE - Word2Vec del corpus + CNN1D")
    w2v_start = time.time()
    word2vec_model, embedding_matrix = train_word2vec_embedding_matrix(train_texts, tokenizer, vocab_size)
    log_info("Tiempo Word2Vec", f"{time.time() - w2v_start:.1f}s")
    log_info("Word2Vec dim", WORD2VEC_DIM)
    log_info("Word2Vec vocab", len(word2vec_model.wv))

    word2vec_cnn_model = compile_model(build_word2vec_cnn(vocab_size, embedding_matrix))
    word2vec_cnn_metrics, word2vec_cnn_history = fit_and_evaluate(
        "AINE_Level2_Word2Vec_CNN1D",
        word2vec_cnn_model,
        x_train,
        y_train,
        x_test,
        y_test,
        class_weight,
    )
    word2vec_cnn_metrics["word2vec_train_time_s"] = round(time.time() - w2v_start, 1)
    word2vec_cnn_model.save(str(LOCAL_OUTPUT_DIR / "level2_word2vec_cnn1d.keras"))
    histories["level2_word2vec_cnn1d"] = word2vec_cnn_history
    results.append(word2vec_cnn_metrics)
    for key, value in word2vec_cnn_metrics.items():
        log_info(key, value)

    best_model = max(results, key=lambda item: item["f1_weighted"])
    summary = {
        "dataset_source": "hdfs_processed_splits_full_for_tensorflow_driver_training",
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "train_fraction": round(train_fraction, 4),
        "test_fraction": round(test_fraction, 4),
        "max_vocab_size": MAX_VOCAB_SIZE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "total_time_s": round(time.time() - total_start, 1),
        "best_model_by_f1_weighted": best_model["model"],
        "results": results,
    }

    log_section("6) Resumen y persistencia")
    log_info("Mejor modelo", best_model["model"])
    log_info("F1 weighted", best_model["f1_weighted"])
    log_info("Tiempo total", f"{summary['total_time_s']:.1f}s")

    save_local_artifacts(tokenizer, histories, summary, word2vec_model=word2vec_model)
    save_hdfs_metrics(spark, summary)
    log_info("Modelos locales", str(LOCAL_OUTPUT_DIR))
    log_info("Metricas HDFS", "hdfs:///metrics/tensorflow_text_baselines/")

    spark.stop()


if __name__ == "__main__":
    main()
