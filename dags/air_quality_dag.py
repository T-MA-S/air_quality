"""Airflow DAG for air quality ETL pipeline."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import sys
sys.path.insert(0, '/opt/airflow/src')

from src.etl.pipeline import ETLPipeline
from src.data.models import CITIES
from src.database.connection import get_engine
from src.data.quality import DataQualityChecker
from src.utils.logger import setup_logger
import pandas as pd
from sqlalchemy import text

logger = setup_logger(__name__)

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


def run_etl(**context):
    """Run ETL pipeline for last 7 days."""
    execution_date = context.get("execution_date", datetime.now())
    date_to = execution_date
    date_from = date_to - timedelta(days=7)

    logger.info(f"Running ETL from {date_from} to {date_to}")

    pipeline = ETLPipeline(cities=CITIES)
    results = pipeline.run(date_from=date_from, date_to=date_to)

    logger.info(f"ETL completed: {results}")

    # Check for quality issues
    if results["quality_issues"]:
        logger.warning(f"Quality issues detected: {results['quality_issues']}")

    return results


def validate_data_quality(**context):
    """Validate data quality and generate report."""
    engine = get_engine()
    checker = DataQualityChecker()

    # Load recent data
    query = """
    SELECT *
    FROM v_air_quality
    WHERE timestamp >= NOW() - INTERVAL '7 days'
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    if df.empty:
        logger.warning("No data to validate")
        return

    # Run quality checks
    report = checker.validate_air_quality_data(df)

    if not report["valid"]:
        logger.error(f"Data quality issues: {report['warnings']}")
        raise ValueError("Data quality validation failed")

    logger.info("Data quality validation passed")


def create_views(**context):
    """Create views in database."""
    engine = get_engine()
    
    # Read SQL file
    with open("/opt/airflow/project/sql/views.sql", "r") as f:
        sql_content = f.read()
    
    # Выполняем SQL
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            # Разделяем на отдельные команды
            for statement in sql_content.split(";"):
                statement = statement.strip()
                if statement and not statement.startswith("--"):
                    try:
                        conn.execute(text(statement))
                    except Exception as e:
                        # Игнорируем ошибки, если представления уже существуют
                        logger.debug(f"View might already exist: {e}")
            trans.commit()
        except Exception as e:
            trans.rollback()
            logger.warning(f"Error creating views: {e}")
    
    logger.info("Views created/updated")


# Tasks
extract_task = PythonOperator(
    task_id="extract_and_load",
    python_callable=run_etl,
    dag=dag,
)

validate_task = PythonOperator(
    task_id="validate_data_quality",
    python_callable=validate_data_quality,
    dag=dag,
)

create_views_task = PythonOperator(
    task_id="create_views",
    python_callable=create_views,
    dag=dag,
)

# Task dependencies
extract_task >> validate_task >> create_views_task

