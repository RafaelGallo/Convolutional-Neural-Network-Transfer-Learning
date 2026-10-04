"""Airflow DAG that registers the already-trained Fashion-MNIST models
(CNN baseline + transfer learning backbones) into MLflow.

It is manual-trigger by default (schedule=None): run it from the Airflow UI
whenever the models/ metrics or artifacts change and you want them reflected
in MLflow.
"""

from __future__ import annotations

import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

SCRIPTS_DIR = "/opt/airflow/scripts"


def _run_log_models_to_mlflow():
    sys.path.insert(0, SCRIPTS_DIR)
    from log_models_to_mlflow import main

    main()


with DAG(
    dag_id="log_fashion_mnist_models_to_mlflow",
    description="Registra metricas, historicos e artifacts dos modelos Fashion-MNIST no MLflow",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlflow", "fashion-mnist"],
) as dag:
    log_models_to_mlflow = PythonOperator(
        task_id="log_models_to_mlflow",
        python_callable=_run_log_models_to_mlflow,
    )
