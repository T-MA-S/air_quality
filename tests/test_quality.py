"""Tests for data quality checks."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.quality import DataQualityChecker


class TestDataQualityChecker:
    """Tests for data quality checking."""

    def test_check_missing_values(self, quality_checker):
        """Test missing value detection."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, None, 4, 5],
                "col2": [1, 2, 3, 4, 5],
            }
        )

        missing_ratios = quality_checker.check_missing_values(df)

        assert missing_ratios["col1"] == 0.2
        assert missing_ratios["col2"] == 0.0

    def test_check_value_ranges(self, quality_checker):
        """Test value range validation."""
        df = pd.DataFrame({"pm25": [10, 20, 30, 100, 200]})

        violations = quality_checker.check_value_ranges(df, "pm25", 0, 500)
        assert violations == 0

        violations = quality_checker.check_value_ranges(df, "pm25", 0, 50)
        assert violations == 2

    def test_check_temporal_monotonicity(self, quality_checker):
        """Test temporal monotonicity check."""
        dates = pd.date_range(start=datetime.now(), periods=5, freq="H")
        df = pd.DataFrame({"timestamp": dates})

        assert quality_checker.check_temporal_monotonicity(df) is True

        # Non-monotonic
        df_reversed = df.sort_values("timestamp", ascending=False)
        assert quality_checker.check_temporal_monotonicity(df_reversed) is False

    def test_check_sudden_jumps(self, quality_checker):
        """Test sudden jump detection."""
        # Normal data
        df_normal = pd.DataFrame({"pm25": [10, 11, 12, 13, 14]})
        jumps = quality_checker.check_sudden_jumps(df_normal, "pm25")
        assert jumps == 0

        # Data with outlier
        df_outlier = pd.DataFrame({"pm25": [10, 11, 12, 100, 14]})
        jumps = quality_checker.check_sudden_jumps(df_outlier, "pm25")
        assert jumps > 0

    def test_validate_air_quality_data(self, quality_checker):
        """Test comprehensive air quality validation."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=100, freq="H")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "city": ["Test City"] * 100,
                "pm25": np.random.normal(15, 5, 100),
                "pm10": np.random.normal(25, 10, 100),
            }
        )

        report = quality_checker.validate_air_quality_data(df)

        assert "valid" in report
        assert "checks" in report
        assert "warnings" in report

    def test_check_data_completeness(self, quality_checker):
        """Test data completeness check."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=2), periods=48, freq="H")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "city": ["City1"] * 30 + ["City2"] * 18,  # Different completeness
            }
        )

        completeness = quality_checker.check_data_completeness(df)

        assert "City1" in completeness
        assert "City2" in completeness
        assert completeness["City1"]["completeness_ratio"] > completeness["City2"]["completeness_ratio"]

