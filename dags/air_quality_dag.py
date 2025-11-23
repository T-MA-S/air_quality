"""Airflow DAG for air quality ETL pipeline."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Default arguments
default_args = {
    "owner": "air_quality_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    "air_quality_etl",
    default_args=default_args,
    description="ETL pipeline for air quality and weather data",
    schedule_interval=timedelta(hours=1),  # Run every hour
    start_date=days_ago(1),
    catchup=False,
    tags=["air_quality", "weather", "etl"],
)


# Tasks
extract_task = BashOperator(
    task_id="extract_and_load",
    bash_command="docker exec air_quality-app-1 python -m src.main --days 7",
    dag=dag,
)

validate_task = BashOperator(
    task_id="validate_data_quality",
    bash_command="docker exec air_quality-app-1 python -c \"from src.database.connection import get_engine; from src.data.quality import DataQualityChecker; import pandas as pd; from sqlalchemy import text; engine = get_engine(); checker = DataQualityChecker(); df = pd.read_sql(text('SELECT * FROM v_air_quality WHERE timestamp >= NOW() - INTERVAL \\'7 days\\''), engine); report = checker.validate_air_quality_data(df); assert report['valid'], f'Validation failed: {report[\\'warnings\\']}'\"",
    dag=dag,
)

create_views_task = BashOperator(
    task_id="create_views",
    bash_command="docker exec -i air_quality-postgres-1 psql -U air_quality_user -d air_quality < /opt/airflow/project/sql/views.sql || true",
    dag=dag,
)

# Task dependencies
extract_task >> validate_task >> create_views_task

