"""Tests for ML models."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.ml.predictor import PM25Predictor


class TestPM25Predictor:
    """Tests for PM2.5 predictor."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=720, freq="H")
        return pd.DataFrame(
            {
                "timestamp": dates,
                "temperature_c": np.random.normal(20, 5, 720),
                "humidity_percent": np.random.normal(60, 10, 720),
                "precipitation_mm": np.random.exponential(0.5, 720),
                "wind_speed_ms": np.random.normal(5, 2, 720),
                "wind_direction_deg": np.random.uniform(0, 360, 720),
                "pm10": np.random.normal(25, 10, 720),
                "pm25": np.random.normal(15, 5, 720),  # Target
            }
        )

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = PM25Predictor()
        assert predictor.model is not None
        assert not predictor.is_trained

    def test_prepare_features(self, sample_data):
        """Test feature preparation."""
        predictor = PM25Predictor()
        features = predictor.prepare_features(sample_data)

        assert len(features) == len(sample_data)
        assert all(col in features.columns for col in predictor.feature_names)

    def test_train(self, sample_data):
        """Test model training."""
        predictor = PM25Predictor()
        X = sample_data
        y = sample_data["pm25"]

        metrics = predictor.train(X, y)

        assert predictor.is_trained
        assert "test_r2" in metrics
        assert "test_rmse" in metrics
        assert metrics["test_r2"] > -1  # R² can be negative but should be reasonable

    def test_predict(self, sample_data):
        """Test prediction."""
        predictor = PM25Predictor()
        X = sample_data
        y = sample_data["pm25"]

        predictor.train(X, y)
        predictions = predictor.predict(X.head(10))

        assert len(predictions) == 10
        assert all(np.isfinite(predictions))

    def test_save_load(self, sample_data, tmp_path):
        """Test model saving and loading."""
        predictor = PM25Predictor(model_path=str(tmp_path / "model.pkl"))
        X = sample_data
        y = sample_data["pm25"]

        predictor.train(X, y)
        predictor.save()

        # Create new predictor and load
        predictor2 = PM25Predictor(model_path=str(tmp_path / "model.pkl"))
        predictor2.load()

        assert predictor2.is_trained
        predictions1 = predictor.predict(X.head(5))
        predictions2 = predictor2.predict(X.head(5))

        np.testing.assert_array_almost_equal(predictions1, predictions2)

