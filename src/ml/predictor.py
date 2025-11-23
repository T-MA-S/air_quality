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
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        self.model_path = model_path or "./models/pm25_model.pkl"
        self.feature_names = [
            "temperature_c",
            "humidity_percent",
            "precipitation_mm",
            "wind_speed_ms",
            "wind_direction_deg",
            "pm10",  # If available
        ]
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model.

        Args:
            df: DataFrame with weather and air quality data

        Returns:
            DataFrame with features
        """
        features = df[self.feature_names].copy()

        # Handle missing values
        features = features.fillna(features.mean())

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
        # Prepare features
        X_features = self.prepare_features(X)

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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")

        X_features = self.prepare_features(X)
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
        self.feature_names = model_data.get("feature_names", self.feature_names)
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

