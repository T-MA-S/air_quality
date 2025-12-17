#!/usr/bin/env python3
"""Скрипт для обучения ML-модели прогноза PM2.5 с back-testing."""

import sys
sys.path.insert(0, '/opt/airflow/src')

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.database.connection import get_engine
from src.ml.predictor import PM25Predictor
from src.utils.logger import setup_logger
from sqlalchemy import text

logger = setup_logger(__name__)


def load_training_data(city_name: str, days_back: int = 90) -> pd.DataFrame:
    """Загружает данные для обучения модели."""
    engine = get_engine()
    
    query = f"""
    SELECT
        aq.timestamp,
        aq.value_ug_m3 as pm25,
        w.temperature_c,
        w.humidity_percent,
        w.precipitation_mm,
        w.wind_speed_ms,
        w.wind_direction_deg,
        pm10_aq.value_ug_m3 as pm10
    FROM fact_air_quality aq
    JOIN dim_city c ON aq.city_id = c.city_id
    JOIN dim_metric m ON aq.metric_id = m.metric_id
    LEFT JOIN fact_weather w ON aq.city_id = w.city_id 
        AND DATE(aq.timestamp) = DATE(w.timestamp)
        AND EXTRACT(HOUR FROM aq.timestamp) = EXTRACT(HOUR FROM w.timestamp)
    LEFT JOIN fact_air_quality pm10_aq ON aq.city_id = pm10_aq.city_id
        AND aq.timestamp = pm10_aq.timestamp
        AND pm10_aq.metric_id = (SELECT metric_id FROM dim_metric WHERE metric_code = 'pm10')
    WHERE c.city_name = '{city_name}'
        AND m.metric_code = 'pm25'
        AND aq.timestamp >= NOW() - INTERVAL '{days_back} days'
        AND aq.timestamp < NOW()
    ORDER BY aq.timestamp
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index()
    
    return df


def train_model_for_city(city_name: str, days_back: int = 90):
    """Обучает модель для конкретного города."""
    logger.info(f"Загрузка данных для {city_name}...")
    df = load_training_data(city_name, days_back)
    
    if df.empty or len(df) < 100:
        logger.warning(f"Недостаточно данных для {city_name} (только {len(df)} записей)")
        return None
    
    # Удаляем строки с отсутствующими целевыми значениями
    df = df.dropna(subset=['pm25'])
    
    # Заполняем пропуски в базовых признаках перед созданием лагов
    base_feature_cols = ['temperature_c', 'humidity_percent', 'precipitation_mm', 
                         'wind_speed_ms', 'wind_direction_deg', 'pm10']
    for col in base_feature_cols:
        if col in df.columns:
            if df[col].notna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = 0
        else:
            df[col] = 0
    
    # Создаем и обучаем модель (prepare_features создаст все признаки включая лаги)
    predictor = PM25Predictor()
    
    # Подготовка признаков с feature engineering (лаги, временные признаки и т.д.)
    # prepare_features создаст все признаки и сохранит их имена в predictor.feature_names
    X_features = predictor.prepare_features(df, target_col='pm25')
    y = df['pm25']
    
    # Удаляем строки, где лаги не могут быть вычислены (первые строки)
    valid_mask = ~(X_features['pm25_lag_1h'].isna() if 'pm25_lag_1h' in X_features.columns else pd.Series([True] * len(X_features)))
    X_features = X_features[valid_mask]
    y = y[valid_mask]
    
    logger.info(f"Данные подготовлены: {len(X_features)} записей, {len(X_features.columns)} признаков")
    logger.info(f"Признаки: {', '.join(X_features.columns.tolist()[:10])}..." if len(X_features.columns) > 10 else f"Признаки: {', '.join(X_features.columns.tolist())}")
    logger.info(f"Feature names saved: {len(predictor.feature_names)} features")
    
    logger.info("Обучение модели...")
    # ВАЖНО: Передаем уже подготовленные признаки напрямую в train
    # Но train ожидает DataFrame с исходными колонками, поэтому создаем обертку
    # Создаем DataFrame с подготовленными признаками, но добавляем pm25 для совместимости
    X_for_train = X_features.copy()
    # train будет вызывать prepare_features снова, но feature_names уже сохранены
    # Поэтому нужно передать данные так, чтобы prepare_features создал те же признаки
    # Но проще передать уже подготовленные признаки и изменить train
    # Или передать исходный df, но убедиться, что prepare_features создаст те же признаки
    metrics = predictor.train(X_features, y, test_size=0.2, validation_size=0.2)
    
    logger.info(f"Метрики модели для {city_name}:")
    logger.info(f"  Train R²: {metrics['train_r2']:.3f}, RMSE: {metrics['train_rmse']:.3f}")
    logger.info(f"  Val R²: {metrics['val_r2']:.3f}, RMSE: {metrics['val_rmse']:.3f}")
    logger.info(f"  Test R²: {metrics['test_r2']:.3f}, RMSE: {metrics['test_rmse']:.3f}")
    
    # Back-testing
    logger.info("Запуск back-testing...")
    try:
        # Подготовка данных для back-testing (нужен timestamp в индексе)
        X_bt = X.copy()
        X_bt['timestamp'] = X_bt.index
        X_bt = X_bt.reset_index(drop=True)
        
        backtest_results = backtest_model(X, y, predictor)
        
        if backtest_results:
            logger.info(f"Back-testing результаты для {city_name}:")
            logger.info(f"  Overall R²: {backtest_results['overall_r2']:.3f}")
            logger.info(f"  Overall RMSE: {backtest_results['overall_rmse']:.3f}")
            logger.info(f"  Overall MAE: {backtest_results['overall_mae']:.3f}")
    except Exception as e:
        logger.warning(f"Ошибка при back-testing: {e}")
        backtest_results = None
    
    # Сохраняем модель
    model_path = f"/opt/airflow/models/pm25_model_{city_name.lower().replace(' ', '_')}.pkl"
    predictor.save(model_path)
    logger.info(f"Модель сохранена: {model_path}")
    
    return {
        'city': city_name,
        'metrics': metrics,
        'backtest': backtest_results,
        'model_path': model_path
    }


def backtest_model(X: pd.DataFrame, y: pd.Series, predictor: PM25Predictor, 
                   train_window_days: int = 30, test_window_days: int = 7) -> dict:
    """Улучшенный back-testing с временными окнами."""
    if len(X) < (train_window_days + test_window_days) * 24:
        logger.warning("Недостаточно данных для back-testing")
        return None
    
    # Сортируем по времени
    X_sorted = X.sort_index()
    y_sorted = y.loc[X_sorted.index]
    
    results = {
        "predictions": [],
        "actuals": [],
        "dates": [],
        "metrics": [],
    }
    
    # Скользящее окно back-testing
    total_hours = len(X_sorted)
    train_hours = train_window_days * 24
    test_hours = test_window_days * 24
    step_hours = test_hours  # Сдвигаем окно на размер тестового периода
    
    windows = []
    for start_idx in range(0, total_hours - train_hours - test_hours, step_hours):
        train_end_idx = start_idx + train_hours
        test_end_idx = train_end_idx + test_hours
        
        X_train = X_sorted.iloc[start_idx:train_end_idx]
        y_train = y_sorted.iloc[start_idx:train_end_idx]
        X_test = X_sorted.iloc[train_end_idx:test_end_idx]
        y_test = y_sorted.iloc[train_end_idx:test_end_idx]
        
        if len(X_train) < 100 or len(X_test) < 24:
            continue
        
        # Обучаем модель на окне
        predictor.model.fit(predictor.prepare_features(X_train), y_train)
        
        # Предсказываем
        predictions = predictor.model.predict(predictor.prepare_features(X_test))
        
        # Сохраняем результаты
        results["predictions"].extend(predictions.tolist())
        results["actuals"].extend(y_test.values.tolist())
        results["dates"].extend(X_test.index.tolist())
        
        # Метрики для окна
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        window_metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "r2": r2_score(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
        }
        results["metrics"].append(window_metrics)
        windows.append({
            'start': X_test.index[0],
            'end': X_test.index[-1],
            'metrics': window_metrics
        })
    
    # Общие метрики
    if results["predictions"]:
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        results["overall_rmse"] = np.sqrt(mean_squared_error(results["actuals"], results["predictions"]))
        results["overall_r2"] = r2_score(results["actuals"], results["predictions"])
        results["overall_mae"] = mean_absolute_error(results["actuals"], results["predictions"])
        results["windows"] = windows
    
    return results


def main():
    """Основная функция для обучения моделей."""
    from src.data.models import CITIES
    
    logger.info("Начало обучения ML-моделей для прогноза PM2.5")
    
    results = []
    for city in CITIES:
        try:
            result = train_model_for_city(city.name, days_back=90)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Ошибка при обучении модели для {city.name}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"Обучение завершено. Обучено моделей: {len(results)}")
    return results


if __name__ == "__main__":
    main()


