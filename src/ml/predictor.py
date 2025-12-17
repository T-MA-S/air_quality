"""ML model for PM2.5 prediction."""

import joblib
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PM25Predictor:
    """Gradient Boosting model for PM2.5 prediction."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize PM2.5 predictor.

        Args:
            model_path: Path to saved model file
        """
        # Улучшенные гиперпараметры для лучшей точности
        self.model = GradientBoostingRegressor(
            n_estimators=200,  # Увеличено с 100
            max_depth=7,  # Увеличено с 5
            learning_rate=0.05,  # Уменьшено с 0.1 для более плавного обучения
            min_samples_split=10,  # Добавлено для предотвращения переобучения
            min_samples_leaf=5,  # Добавлено
            subsample=0.8,  # Добавлено для уменьшения переобучения
            random_state=42,
        )
        self.model_path = model_path or "./models/pm25_model.pkl"
        # Базовые признаки (будут расширены в prepare_features)
        self.base_feature_names = [
            "temperature_c",
            "humidity_percent",
            "precipitation_mm",
            "wind_speed_ms",
            "wind_direction_deg",
            "pm10",  # If available
        ]
        self.feature_names = []  # Будет заполнено при подготовке признаков
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame, target_col: str = 'pm25') -> pd.DataFrame:
        """
        Prepare features for model with feature engineering.

        Args:
            df: DataFrame with weather and air quality data
            target_col: Name of target column (for creating lags)

        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Базовые признаки (погода и PM10)
        for col in self.base_feature_names:
            if col in df.columns:
                features[col] = df[col]
            else:
                features[col] = 0
        
        # 2. Лаги PM2.5 (важно для временных рядов!)
        if target_col in df.columns:
            # Сортируем по индексу (timestamp)
            df_sorted = df.sort_index()
            target_sorted = df_sorted[target_col]
            
            # Создаем лаги
            features['pm25_lag_1h'] = target_sorted.shift(1)
            features['pm25_lag_2h'] = target_sorted.shift(2)
            features['pm25_lag_3h'] = target_sorted.shift(3)
            features['pm25_lag_6h'] = target_sorted.shift(6)
            features['pm25_lag_24h'] = target_sorted.shift(24)  # Вчера в это же время
        
        # 3. Скользящие средние PM2.5
        if target_col in df.columns:
            df_sorted = df.sort_index()
            target_sorted = df_sorted[target_col]
            
            features['pm25_ma_3h'] = target_sorted.rolling(window=3, min_periods=1).mean()
            features['pm25_ma_6h'] = target_sorted.rolling(window=6, min_periods=1).mean()
            features['pm25_ma_24h'] = target_sorted.rolling(window=24, min_periods=1).mean()
        
        # 4. Временные признаки (если индекс - timestamp)
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['month'] = df.index.month
            # Циклические признаки для времени
            features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            features['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            features['hour'] = df['timestamp'].dt.hour
            features['day_of_week'] = df['timestamp'].dt.dayofweek
            features['month'] = df['timestamp'].dt.month
            features['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
            features['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        
        # 5. Взаимодействия признаков
        if 'temperature_c' in features.columns and 'humidity_percent' in features.columns:
            features['temp_humidity'] = features['temperature_c'] * features['humidity_percent']
        if 'wind_speed_ms' in features.columns and 'wind_direction_deg' in features.columns:
            # Векторные компоненты ветра
            features['wind_x'] = features['wind_speed_ms'] * np.cos(np.radians(features['wind_direction_deg']))
            features['wind_y'] = features['wind_speed_ms'] * np.sin(np.radians(features['wind_direction_deg']))
        
        # 6. Handle missing values
        for col in features.columns:
            if features[col].isna().any():
                if features[col].notna().sum() > 0:
                    # Используем медиану для заполнения
                    median_val = features[col].median()
                    if pd.notna(median_val):
                        features[col] = features[col].fillna(median_val)
                    else:
                        features[col] = features[col].fillna(0)
                else:
                    # Если все значения NaN, заполняем 0
                    features[col] = features[col].fillna(0)
        
        # 7. Заменяем бесконечные значения на 0
        features = features.replace([np.inf, -np.inf], 0)
        
        # 8. Финальная проверка на NaN
        if features.isna().any().any():
            logger.warning(f"Still have NaN values after filling. Columns: {features.columns[features.isna().any()].tolist()}")
            features = features.fillna(0)
        
        # Сохраняем имена признаков для использования при предсказании
        # Только если они еще не сохранены (при обучении)
        if not self.feature_names or len(self.feature_names) == 0:
            self.feature_names = features.columns.tolist()
        # Если feature_names уже сохранены (при загрузке модели), используем их для упорядочивания
        elif self.feature_names and len(self.feature_names) > 0:
            # Убеждаемся, что все сохраненные признаки присутствуют
            missing_cols = set(self.feature_names) - set(features.columns)
            if missing_cols:
                logger.warning(f"Missing features during prediction: {missing_cols}. Filling with 0.")
                for col in missing_cols:
                    features[col] = 0
            # Упорядочиваем колонки как при обучении
            features = features[self.feature_names]

        return features

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        validation_size: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X: Feature DataFrame
            y: Target values (PM2.5)
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation

        Returns:
            Dictionary with metrics
        """
        # Проверяем, переданы ли уже подготовленные признаки
        # Признаки подготовлены, если есть engineered признаки (лаги, скользящие средние, временные)
        has_engineered_features = any(
            'lag' in str(col) or 'ma_' in str(col) or 
            col in ['hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'temp_humidity', 'wind_x', 'wind_y']
            for col in X.columns
        )
        has_pm25 = 'pm25' in X.columns
        
        if has_engineered_features and not has_pm25:
            # Признаки уже подготовлены, используем их напрямую
            X_features = X.copy()
            logger.info(f"Using pre-prepared features for training ({len(X_features.columns)} features)")
            # Убеждаемся, что feature_names сохранены
            if not self.feature_names or len(self.feature_names) == 0:
                self.feature_names = X_features.columns.tolist()
                logger.info(f"Feature names saved during training: {len(self.feature_names)} features")
        else:
            # Нужно подготовить признаки
            target_col = 'pm25' if has_pm25 else None
            X_features = self.prepare_features(X, target_col=target_col)
            
            # Сохраняем feature_names после первой подготовки (важно для предсказания!)
            # prepare_features уже сохраняет feature_names, но убедимся
            if not self.feature_names or len(self.feature_names) == 0:
                self.feature_names = X_features.columns.tolist()
                logger.info(f"Feature names saved during training: {len(self.feature_names)} features")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42
        )

        # Further split training for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=42
        )

        # Train model
        logger.info(f"Training model on {len(X_train)} samples")
        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        test_pred = self.model.predict(X_test)

        metrics = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
            "train_r2": r2_score(y_train, train_pred),
            "train_mae": mean_absolute_error(y_train, train_pred),
            "val_rmse": np.sqrt(mean_squared_error(y_val, val_pred)),
            "val_r2": r2_score(y_val, val_pred),
            "val_mae": mean_absolute_error(y_val, val_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
            "test_r2": r2_score(y_test, test_pred),
            "test_mae": mean_absolute_error(y_test, test_pred),
        }

        self.is_trained = True
        logger.info(f"Model trained. Test R²: {metrics['test_r2']:.3f}, RMSE: {metrics['test_rmse']:.3f}")

        return metrics

    def predict(self, X: pd.DataFrame, target_col: str = 'pm25') -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature DataFrame
            target_col: Name of target column (for creating lags if needed)

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")

        # Подготавливаем признаки (создаст лаги и временные признаки)
        # prepare_features уже упорядочит признаки согласно self.feature_names
        X_features = self.prepare_features(X, target_col=target_col)
        
        # Критическая проверка: убеждаемся, что все признаки на месте и в правильном порядке
        if not self.feature_names or len(self.feature_names) == 0:
            raise ValueError("Model feature_names not set! Cannot make predictions.")
        
        # Проверяем наличие всех признаков
        missing_cols = set(self.feature_names) - set(X_features.columns)
        if missing_cols:
            logger.error(f"Critical: Missing features after preparation: {missing_cols}")
            logger.error(f"Expected features: {len(self.feature_names)}, Got: {len(X_features.columns)}")
            logger.error(f"Expected: {self.feature_names[:10]}...")
            logger.error(f"Got: {list(X_features.columns)[:10]}...")
            for col in missing_cols:
                X_features[col] = 0
                logger.warning(f"Added missing feature '{col}' with zeros")
        
        # Упорядочиваем признаки точно в том порядке, в котором они были при обучении
        if list(X_features.columns) != self.feature_names:
            logger.warning(f"Feature order mismatch. Reordering from {len(X_features.columns)} to {len(self.feature_names)} features...")
            logger.debug(f"Expected order: {self.feature_names[:5]}...")
            logger.debug(f"Current order: {list(X_features.columns)[:5]}...")
            X_features = X_features[self.feature_names]
        
        # Финальная проверка
        if list(X_features.columns) != self.feature_names:
            raise ValueError(f"Feature order still incorrect after reordering! Expected {len(self.feature_names)} features, got {len(X_features.columns)}")
        
        logger.debug(f"Making prediction with {len(X_features.columns)} features in correct order")
        predictions = self.model.predict(X_features)
        return predictions

    def save(self, path: Optional[str] = None) -> None:
        """Save model to file."""
        save_path = path or self.model_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """Load model from file."""
        load_path = path or self.model_path

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        model_data = joblib.load(load_path)
        self.model = model_data["model"]
        # Восстанавливаем feature_names из сохраненной модели
        saved_feature_names = model_data.get("feature_names", [])
        if saved_feature_names and len(saved_feature_names) > 0:
            self.feature_names = saved_feature_names
            logger.info(f"Model loaded with {len(self.feature_names)} features")
            logger.debug(f"Feature names: {', '.join(self.feature_names[:10])}...")
        else:
            logger.warning(f"Model loaded but feature_names not found. Using default feature names.")
            # Если feature_names не найдены, пытаемся получить их из модели sklearn
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
                logger.info(f"Using feature_names from sklearn model: {len(self.feature_names)} features")
            else:
                logger.error("Cannot determine feature names from model!")
        self.is_trained = model_data.get("is_trained", False)

        logger.info(f"Model loaded from {load_path}")

    def backtest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_window_days: int = 30,
        test_window_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Backtest model on historical data.

        Args:
            X: Feature DataFrame with timestamp index
            y: Target values
            train_window_days: Days to use for training
            test_window_days: Days to use for testing

        Returns:
            Backtest results dictionary
        """
        if "timestamp" not in X.index.names and "timestamp" not in X.columns:
            raise ValueError("DataFrame must have timestamp index or column")

        # Sort by timestamp
        if "timestamp" in X.columns:
            X = X.sort_values("timestamp")
            y = y.loc[X.index]

        results = {
            "predictions": [],
            "actuals": [],
            "dates": [],
            "metrics": [],
        }

        # Sliding window backtest
        total_days = (X.index.max() - X.index.min()).days if hasattr(X.index, "max") else len(X) // 24

        for start_day in range(0, total_days - train_window_days - test_window_days, test_window_days):
            train_end = start_day + train_window_days
            test_start = train_end
            test_end = test_start + test_window_days

            # Split data (simplified - assumes daily frequency)
            X_train = X.iloc[start_day * 24 : train_end * 24]
            y_train = y.iloc[start_day * 24 : train_end * 24]
            X_test = X.iloc[test_start * 24 : test_end * 24]
            y_test = y.iloc[test_start * 24 : test_end * 24]

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            # Train on window
            self.model.fit(self.prepare_features(X_train), y_train)

            # Predict
            predictions = self.predict(X_test)

            # Store results
            results["predictions"].extend(predictions)
            results["actuals"].extend(y_test.values)
            results["dates"].extend(X_test.index if hasattr(X_test.index, "tolist") else range(len(X_test)))

            # Calculate metrics
            window_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "r2": r2_score(y_test, predictions),
                "mae": mean_absolute_error(y_test, predictions),
            }
            results["metrics"].append(window_metrics)

        # Overall metrics
        if results["predictions"]:
            overall_metrics = {
                "overall_rmse": np.sqrt(mean_squared_error(results["actuals"], results["predictions"])),
                "overall_r2": r2_score(results["actuals"], results["predictions"]),
                "overall_mae": mean_absolute_error(results["actuals"], results["predictions"]),
            }
            results["overall_metrics"] = overall_metrics

        return results

