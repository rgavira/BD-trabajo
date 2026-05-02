#!/bin/bash
set -e

echo ">>> Inicializando base de datos de Airflow..."
airflow db migrate

echo ">>> Creando usuario admin..."
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    2>/dev/null || echo "Usuario ya existe, continuando..."

echo ">>> Arrancando Airflow scheduler en background..."
airflow scheduler &

echo ">>> Arrancando Airflow webserver..."
exec airflow webserver
