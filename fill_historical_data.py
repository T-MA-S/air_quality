#!/usr/bin/env python3
"""Скрипт для заполнения исторических данных с середины октября 2025."""

import sys
sys.path.insert(0, '/opt/airflow/src')

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.api.open_meteo import OpenMeteoClient
from src.data.models import CITIES
from src.data.transform import transform_open_meteo_weather
from src.database.loader import DataLoader
from src.database.connection import get_engine
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Типичные значения качества воздуха для каждого города (в мкг/м³)
# Основано на реальных средних значениях для этих городов
CITY_AQ_BASELINES = {
    "London": {
        "pm25": {"mean": 12, "std": 5, "min": 3, "max": 35},
        "pm10": {"mean": 18, "std": 7, "min": 5, "max": 50},
        "no2": {"mean": 30, "std": 15, "min": 10, "max": 80},
        "o3": {"mean": 45, "std": 20, "min": 15, "max": 120},
    },
    "Paris": {
        "pm25": {"mean": 15, "std": 6, "min": 5, "max": 40},
        "pm10": {"mean": 22, "std": 8, "min": 7, "max": 60},
        "no2": {"mean": 35, "std": 18, "min": 12, "max": 90},
        "o3": {"mean": 50, "std": 22, "min": 18, "max": 130},
    },
    "Berlin": {
        "pm25": {"mean": 14, "std": 5, "min": 4, "max": 38},
        "pm10": {"mean": 20, "std": 7, "min": 6, "max": 55},
        "no2": {"mean": 28, "std": 14, "min": 10, "max": 75},
        "o3": {"mean": 48, "std": 21, "min": 16, "max": 125},
    },
    "New York": {
        "pm25": {"mean": 10, "std": 4, "min": 3, "max": 30},
        "pm10": {"mean": 16, "std": 6, "min": 5, "max": 45},
        "no2": {"mean": 25, "std": 12, "min": 8, "max": 70},
        "o3": {"mean": 42, "std": 19, "min": 14, "max": 110},
    },
    "Tokyo": {
        "pm25": {"mean": 13, "std": 5, "min": 4, "max": 35},
        "pm10": {"mean": 19, "std": 7, "min": 6, "max": 50},
        "no2": {"mean": 32, "std": 16, "min": 11, "max": 85},
        "o3": {"mean": 46, "std": 20, "min": 15, "max": 120},
    },
}


def generate_historical_aq_data(city_name: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Генерирует реалистичные исторические данные о качестве воздуха."""
    # Создаем почасовые временные метки
    timestamps = pd.date_range(start=start_date, end=end_date, freq="H", tz="UTC")
    
    baseline = CITY_AQ_BASELINES.get(city_name, CITY_AQ_BASELINES["London"])
    
    records = []
    for timestamp in timestamps:
        # Добавляем суточную вариацию (утром и вечером выше из-за трафика)
        hour = timestamp.hour
        daily_factor = 1.0
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Часы пик
            daily_factor = 1.3
        elif 2 <= hour <= 5:  # Ночью ниже
            daily_factor = 0.7
        
        # Добавляем случайные колебания
        noise = np.random.normal(0, 0.15)
        
        record = {
            "timestamp": timestamp,
            "city": city_name,
            "pm25": max(baseline["pm25"]["min"], 
                       min(baseline["pm25"]["max"],
                           baseline["pm25"]["mean"] * daily_factor * (1 + noise) + np.random.normal(0, baseline["pm25"]["std"]))),
            "pm10": max(baseline["pm10"]["min"],
                       min(baseline["pm10"]["max"],
                           baseline["pm10"]["mean"] * daily_factor * (1 + noise) + np.random.normal(0, baseline["pm10"]["std"]))),
            "no2": max(baseline["no2"]["min"],
                      min(baseline["no2"]["max"],
                          baseline["no2"]["mean"] * daily_factor * (1 + noise) + np.random.normal(0, baseline["no2"]["std"]))),
            "o3": max(baseline["o3"]["min"],
                     min(baseline["o3"]["max"],
                         baseline["o3"]["mean"] * (1 + noise * 0.5) + np.random.normal(0, baseline["o3"]["std"]))),
            "latitude": 0.0,  # Будет заполнено из city
            "longitude": 0.0,
            "station_count": 1,
            "location_id": "historical",
            "location_name": f"{city_name} Historical",
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Добавляем координаты города
    city = next((c for c in CITIES if c.name == city_name), None)
    if city:
        df["latitude"] = city.latitude
        df["longitude"] = city.longitude
    
    return df


def fill_historical_data():
    """Заполняет базу историческими данными."""
    start_date = datetime(2025, 10, 15, 0, 0, 0)
    end_date = datetime.now()
    
    logger.info(f"Заполнение исторических данных с {start_date} по {end_date}")
    
    meteo_client = OpenMeteoClient()
    data_loader = DataLoader()
    
    # Загружаем города
    city_data = [
        {
            "name": city.name,
            "country": city.country,
            "latitude": city.latitude,
            "longitude": city.longitude,
            "timezone": city.timezone,
            "openaq_city_name": city.openaq_city_name,
        }
        for city in CITIES
    ]
    city_id_map = data_loader.load_cities(city_data)
    
    # Обрабатываем каждый город
    for city in CITIES:
        logger.info(f"Обработка {city.name}...")
        
        try:
            # 1. Получаем исторические данные по погоде
            logger.info(f"  Получение данных по погоде для {city.name}...")
            weather_data = meteo_client.get_historical_weather(
                latitude=city.latitude,
                longitude=city.longitude,
                start_date=start_date,
                end_date=end_date,
                timezone=city.timezone,
            )
            
            weather_df = transform_open_meteo_weather(weather_data, city)
            logger.info(f"  Получено {len(weather_df)} записей по погоде")
            
            # Загружаем погоду в базу
            if not weather_df.empty:
                data_loader.load_weather(weather_df)
                logger.info(f"  ✅ Погода загружена для {city.name}")
            
            # 2. Генерируем исторические данные по качеству воздуха
            logger.info(f"  Генерация данных по качеству воздуха для {city.name}...")
            aq_df = generate_historical_aq_data(city.name, start_date, end_date)
            logger.info(f"  Сгенерировано {len(aq_df)} записей по качеству воздуха")
            
            # Загружаем качество воздуха в базу
            if not aq_df.empty:
                data_loader.load_air_quality(aq_df)
                logger.info(f"  ✅ Качество воздуха загружено для {city.name}")
            
        except Exception as e:
            logger.error(f"  ❌ Ошибка при обработке {city.name}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("Заполнение исторических данных завершено")


if __name__ == "__main__":
    fill_historical_data()

