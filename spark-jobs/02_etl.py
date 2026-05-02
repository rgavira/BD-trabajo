"""
02_etl.py — ETL distribuido con Spark

Pasos:
  1. Lee el Parquet crudo desde el volumen compartido (./data/)
  2. Escribe TODAS las columnas a HDFS como data lake (songs_raw/)
  3. Lee de HDFS seleccionando SOLO text + Explicit  <- beneficio columnar Parquet
  4. Limpieza completa: normalización, regex, filtros, deduplicación
  5. Muestra estratificada ~100k  (comenta el bloque para usar ~500k completas)
  6. Train/test split 80/20
  7. Escribe train/ y test/ a HDFS
"""

import re
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType


def create_spark_session():
    return (
        SparkSession.builder
        .appName("ExplicitLyrics_ETL")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .getOrCreate()
    )


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "explicit", "e"}:
        return 1
    if s in {"0", "false", "no", "non-explicit", "non explicit", "clean", "ne"}:
        return 0
    return None


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    clean_text_udf      = F.udf(clean_text, StringType())
    normalize_label_udf = F.udf(normalize_label, IntegerType())

    # PASO 1: Leer Parquet crudo
    print("=" * 60)
    print("PASO 1: Leyendo Parquet crudo del volumen compartido...")
    spark.sparkContext.setJobDescription("ETL | Paso 1: Leer Parquet crudo del volumen compartido")
    df_raw = spark.read.parquet("file:///opt/data/songs_raw.parquet")
    print(f"  Filas    : {df_raw.count():,}")
    print(f"  Columnas : {df_raw.columns}")

    # PASO 2: Subir dataset completo a HDFS (data lake)
    print("\nPASO 2: Escribiendo dataset completo a HDFS (data lake)...")
    spark.sparkContext.setJobDescription("ETL | Paso 2: Escribir dataset completo a HDFS (data lake)")
    df_raw.write.mode("overwrite").parquet("hdfs://namenode:9000/data/raw/songs_raw/")
    print("  Dataset completo guardado en hdfs:///data/raw/songs_raw/")

    # PASO 3: Leer de HDFS seleccionando SOLO columnas necesarias
    # Aqui esta el beneficio de Parquet columnar: Spark solo lee text + Explicit
    # del disco, sin tocar el resto de columnas aunque esten en HDFS.
    print("\nPASO 3: Leyendo de HDFS con seleccion de columnas (beneficio Parquet)...")
    spark.sparkContext.setJobDescription("ETL | Paso 3: Leer de HDFS con seleccion columnar (text + Explicit)")
    available = df_raw.columns
    text_col  = next((c for c in available if c.lower() == "text"), None)
    label_col = next((c for c in available if c.lower() == "explicit"), None)

    if not text_col or not label_col:
        raise ValueError(
            f"No se encuentran las columnas 'text' y/o 'Explicit'.\n"
            f"Columnas disponibles: {available}"
        )

    df = (
        spark.read.parquet("hdfs://namenode:9000/data/raw/songs_raw/")
        .select(text_col, label_col)
    )
    print(f"  Columnas seleccionadas: {text_col}, {label_col}")

    # PASO 4: Limpieza completa
    print("\nPASO 4: Limpieza completa (ETL distribuido)...")
    spark.sparkContext.setJobDescription("ETL | Paso 4: Limpieza - dropna, normalize labels, clean text, dedup")

    df = df.dropna(subset=[text_col, label_col])
    df = df.withColumn("label", normalize_label_udf(F.col(label_col)))
    df = df.dropna(subset=["label"])
    df = df.withColumn("label", F.col("label").cast(IntegerType()))
    df = df.withColumn("text", clean_text_udf(F.col(text_col)))
    df = df.filter(F.size(F.split(F.col("text"), " ")) >= 5)
    df = df.dropDuplicates(["text"])
    df = df.select("text", "label")

    print("  Distribucion de labels tras ETL:")
    df.groupBy("label").count().orderBy("label").show()

    # PASO 5: Muestra estratificada
    #
    # Para usar el dataset COMPLETO (~500k canciones), el bloque esta comentado.
    # Descomenta para reducir a ~100k (mas rapido, comparable con MLE/AINE).
    #
    # --- BEGIN SAMPLE ---
    # TARGET = 100_000
    # df = df.cache()
    # total = df.count()
    # print(f"\nPASO 5: Muestra estratificada {TARGET:,} de {total:,} canciones...")
    # fraction = min(1.0, TARGET / total)
    # class_keys = {r["label"] for r in df.groupBy("label").count().collect()}
    # df = df.sampleBy("label", fractions={k: fraction for k in class_keys}, seed=42)
    # print(f"  Muestra final: {df.count():,} canciones")
    # --- END SAMPLE ---

    print(f"\nPASO 5: Usando dataset completo tras ETL")

    # PASO 6: Train/test split 80/20
    print("\nPASO 6: Train/test split 80/20...")
    spark.sparkContext.setJobDescription("ETL | Paso 6: Train/test split 80/20")
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    print(f"  Train: {train.count():,}  |  Test: {test.count():,}")

    # PASO 7: Escribir a HDFS
    print("\nPASO 7: Guardando train y test en HDFS...")
    spark.sparkContext.setJobDescription("ETL | Paso 7: Escribir train y test procesados a HDFS")
    train.write.mode("overwrite").parquet("hdfs://namenode:9000/data/processed/train/")
    test.write.mode("overwrite").parquet("hdfs://namenode:9000/data/processed/test/")
    print("  hdfs:///data/processed/train/")
    print("  hdfs:///data/processed/test/")
    print("\nETL completado.")

    spark.stop()


if __name__ == "__main__":
    main()
