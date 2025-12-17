#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Скрипт для создания тестовых пропусков в данных (для демонстрации функциональности)."""

import sys
sys.path.insert(0, '/opt/airflow/src')

from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import text, delete, and_
from src.database.connection import get_engine
from src.database.schema import fact_air_quality, dim_city, dim_metric
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_missing_data_pattern(city_name: str, metric_code: str, missing_percent: float = 30.0):
    """
    Создает паттерн пропусков, удаляя случайные записи из данных.
    
    Args:
        city_name: Название города
        metric_code: Код метрики (pm25, pm10, etc.)
        missing_percent: Процент данных для удаления (0-100)
    """
    engine = get_engine()
    
    # Получаем city_id и metric_id
    with engine.connect() as conn:
        city_query = text("SELECT city_id FROM dim_city WHERE city_name = :city_name")
        city_result = conn.execute(city_query, {"city_name": city_name}).first()
        if not city_result:
            logger.error(f"City {city_name} not found")
            return
        
        city_id = city_result[0]
        
        metric_query = text("SELECT metric_id FROM dim_metric WHERE metric_code = :metric_code")
        metric_result = conn.execute(metric_query, {"metric_code": metric_code}).first()
        if not metric_result:
            logger.error(f"Metric {metric_code} not found")
            return
        
        metric_id = metric_result[0]
        
        # Получаем последние 7 дней данных
        date_from = datetime.now() - timedelta(days=7)
        
        query = text("""
            SELECT id, timestamp
            FROM fact_air_quality
            WHERE city_id = :city_id
                AND metric_id = :metric_id
                AND timestamp >= :date_from
            ORDER BY timestamp
        """)
        
        df = pd.read_sql(query, conn, params={
            "city_id": city_id,
            "metric_id": metric_id,
            "date_from": date_from
        })
    
    if df.empty:
        logger.warning(f"No data found for {city_name} / {metric_code}")
        return
    
    # Вычисляем количество записей для удаления
    total_records = len(df)
    records_to_delete = int(total_records * (missing_percent / 100.0))
    
    if records_to_delete == 0:
        logger.info(f"Not enough records to create {missing_percent}% missing data")
        return
    
    # Случайно выбираем записи для удаления
    import random
    random.seed(42)  # Для воспроизводимости
    indices_to_delete = random.sample(range(total_records), records_to_delete)
    ids_to_delete = df.iloc[indices_to_delete]["id"].tolist()
    
    # Удаляем записи
    with engine.begin() as conn:
        delete_stmt = delete(fact_air_quality).where(
            fact_air_quality.c.id.in_(ids_to_delete)
        )
        result = conn.execute(delete_stmt)
        deleted_count = result.rowcount
    
    logger.info(
        f"Created missing data pattern for {city_name} / {metric_code}: "
        f"Deleted {deleted_count} out of {total_records} records ({missing_percent:.1f}% missing)"
    )


def create_missing_data_by_hours(city_name: str, metric_code: str, hours_to_remove: list):
    """
    Удаляет данные за конкретные часы.
    
    Args:
        city_name: Название города
        metric_code: Код метрики
        hours_to_remove: Список часов для удаления (0-23)
    """
    engine = get_engine()
    
    with engine.connect() as conn:
        city_query = text("SELECT city_id FROM dim_city WHERE city_name = :city_name")
        city_result = conn.execute(city_query, {"city_name": city_name}).first()
        if not city_result:
            logger.error(f"City {city_name} not found")
            return
        
        city_id = city_result[0]
        
        metric_query = text("SELECT metric_id FROM dim_metric WHERE metric_code = :metric_code")
        metric_result = conn.execute(metric_query, {"metric_code": metric_code}).first()
        if not metric_result:
            logger.error(f"Metric {metric_code} not found")
            return
        
        metric_id = metric_result[0]
        
        # Получаем последние 3 дня данных
        date_from = datetime.now() - timedelta(days=3)
        
        # Создаем список условий для часов
        hours_conditions = " OR ".join([f"EXTRACT(HOUR FROM timestamp) = {h}" for h in hours_to_remove])
        
        query = text(f"""
            SELECT id
            FROM fact_air_quality
            WHERE city_id = :city_id
                AND metric_id = :metric_id
                AND timestamp >= :date_from
                AND ({hours_conditions})
        """)
        
        result = conn.execute(query, {
            "city_id": city_id,
            "metric_id": metric_id,
            "date_from": date_from
        })
        
        ids_to_delete = [row[0] for row in result]
    
    if not ids_to_delete:
        logger.warning(f"No records found for specified hours")
        return
    
    # Удаляем записи
    with engine.begin() as conn:
        delete_stmt = delete(fact_air_quality).where(
            fact_air_quality.c.id.in_(ids_to_delete)
        )
        result = conn.execute(delete_stmt)
        deleted_count = result.rowcount
    
    logger.info(
        f"Removed data for hours {hours_to_remove} in {city_name} / {metric_code}: "
        f"Deleted {deleted_count} records"
    )


def main():
    """Создает реалистичные тестовые пропуски в данных (1-3% для каждой метрики)."""
    logger.info("Creating realistic test missing data patterns (1-3% per metric)...")
    
    # Предопределенные реалистичные проценты пропусков для разных комбинаций
    # Значения в диапазоне 1.0-3.0% с двумя знаками после запятой
    missing_data_config = [
        ("London", "pm25", 1.77),
        ("London", "pm10", 2.13),
        ("London", "no2", 1.45),
        ("London", "o3", 2.89),
        ("Paris", "pm25", 1.92),
        ("Paris", "pm10", 2.34),
        ("Paris", "no2", 1.68),
        ("Paris", "o3", 2.56),
        ("Berlin", "pm25", 1.23),
        ("Berlin", "pm10", 2.67),
        ("Berlin", "no2", 1.89),
        ("Berlin", "o3", 2.01),
        ("New York", "pm25", 2.45),
        ("New York", "pm10", 1.56),
        ("New York", "no2", 2.78),
        ("New York", "o3", 1.34),
        ("Tokyo", "pm25", 1.67),
        ("Tokyo", "pm10", 2.23),
        ("Tokyo", "no2", 1.91),
        ("Tokyo", "o3", 2.12),
    ]
    
    success_count = 0
    for city_name, metric_code, missing_percent in missing_data_config:
        logger.info(f"Creating missing data for {city_name} / {metric_code}: {missing_percent}%")
        try:
            create_missing_data_pattern(city_name, metric_code, missing_percent=missing_percent)
            success_count += 1
        except Exception as e:
            logger.warning(f"Failed to create missing data for {city_name} / {metric_code}: {e}")
            continue
    
    logger.info(f"Realistic test missing data patterns created successfully!")
    logger.info(f"Created missing data for {success_count} city/metric combinations")
    logger.info("Refresh the dashboard to see the missing data visualization.")
    logger.info("Expected missing percentages: 1-3% per city/metric combination")


if __name__ == "__main__":
    main()

