"""Data quality checks and validation."""

from typing import Dict, List, Optional

import pandas as pd
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataQualityChecker:
    """Check data quality and generate reports."""

    def __init__(
        self,
        max_missing_ratio: float = None,
        min_data_points_per_day: int = None,
    ):
        """
        Initialize data quality checker.

        Args:
            max_missing_ratio: Maximum allowed ratio of missing values
            min_data_points_per_day: Minimum data points per day
        """
        self.max_missing_ratio = max_missing_ratio or settings.max_missing_ratio
        self.min_data_points_per_day = (
            min_data_points_per_day or settings.min_data_points_per_day
        )

    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Check missing value ratios.

        Returns:
            Dictionary with column names and missing ratios
        """
        missing_ratios = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_ratio = missing_count / len(df) if len(df) > 0 else 1.0
            missing_ratios[col] = missing_ratio

        return missing_ratios

    def check_value_ranges(
        self, df: pd.DataFrame, column: str, min_val: float, max_val: float
    ) -> int:
        """
        Check if values are within expected range.

        Returns:
            Number of violations
        """
        if column not in df.columns:
            return 0

        violations = ((df[column] < min_val) | (df[column] > max_val)).sum()
        return int(violations)

    def check_temporal_monotonicity(self, df: pd.DataFrame, time_col: str = "timestamp") -> bool:
        """
        Check if timestamps are monotonically increasing.

        Returns:
            True if monotonic, False otherwise
        """
        if time_col not in df.columns:
            return False

        df_sorted = df.sort_values(time_col)
        return df_sorted[time_col].is_monotonic_increasing

    def check_data_freshness(
        self, df: pd.DataFrame, time_col: str = "timestamp", max_age_hours: int = 24
    ) -> bool:
        """
        Check if data is fresh enough.

        Returns:
            True if data is fresh, False otherwise
        """
        if time_col not in df.columns or df.empty:
            return False

        latest_timestamp = pd.to_datetime(df[time_col]).max()
        now = pd.Timestamp.now(tz=latest_timestamp.tz)

        age_hours = (now - latest_timestamp).total_seconds() / 3600
        return age_hours <= max_age_hours

    def check_sudden_jumps(
        self, df: pd.DataFrame, column: str, threshold_std: float = 3.0
    ) -> int:
        """
        Detect sudden jumps in values (outliers).

        Returns:
            Number of detected jumps
        """
        if column not in df.columns:
            return 0

        values = df[column].dropna()
        if len(values) < 2:
            return 0

        mean = values.mean()
        std = values.std()

        if std == 0:
            return 0

        z_scores = abs((values - mean) / std)
        jumps = (z_scores > threshold_std).sum()

        return int(jumps)

    def check_data_completeness(
        self, df: pd.DataFrame, time_col: str = "timestamp", group_col: str = "city"
    ) -> Dict[str, Dict[str, float]]:
        """
        Check data completeness by group (e.g., city).

        Returns:
            Dictionary with completeness metrics per group
        """
        if df.empty or time_col not in df.columns:
            return {}

        df[time_col] = pd.to_datetime(df[time_col])
        df["date"] = df[time_col].dt.date

        completeness = {}
        for group_value in df[group_col].unique():
            group_df = df[df[group_col] == group_value]

            # Expected data points per day
            date_range = pd.date_range(
                group_df[time_col].min(), group_df[time_col].max(), freq="H"
            )
            expected_points = len(date_range)

            # Actual data points
            actual_points = len(group_df)

            # Completeness ratio
            completeness_ratio = actual_points / expected_points if expected_points > 0 else 0

            completeness[group_value] = {
                "expected_points": expected_points,
                "actual_points": actual_points,
                "completeness_ratio": completeness_ratio,
                "missing_ratio": 1 - completeness_ratio,
            }

        return completeness

    def validate_air_quality_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive validation of air quality data.

        Returns:
            Validation report dictionary
        """
        report = {
            "valid": True,
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        if df.empty:
            report["valid"] = False
            report["errors"].append("DataFrame is empty")
            return report

        # Check missing values
        missing_ratios = self.check_missing_values(df)
        report["checks"]["missing_ratios"] = missing_ratios

        for col, ratio in missing_ratios.items():
            if ratio > self.max_missing_ratio:
                report["warnings"].append(
                    f"Column {col} has {ratio:.2%} missing values (threshold: {self.max_missing_ratio:.2%})"
                )

        # Check value ranges for common parameters
        range_checks = {
            "pm25": (0, 500),  # µg/m³
            "pm10": (0, 1000),  # µg/m³
            "no2": (0, 500),  # µg/m³
            "o3": (0, 500),  # µg/m³
        }

        for param, (min_val, max_val) in range_checks.items():
            if param in df.columns:
                violations = self.check_value_ranges(df, param, min_val, max_val)
                if violations > 0:
                    report["warnings"].append(
                        f"Parameter {param} has {violations} values outside range [{min_val}, {max_val}]"
                    )

        # Check temporal monotonicity
        if "timestamp" in df.columns:
            is_monotonic = self.check_temporal_monotonicity(df)
            if not is_monotonic:
                report["warnings"].append("Timestamps are not monotonically increasing")

        # Check data freshness
        if "timestamp" in df.columns:
            is_fresh = self.check_data_freshness(df)
            if not is_fresh:
                report["warnings"].append("Data is not fresh (older than 24 hours)")

        # Check sudden jumps
        for param in ["pm25", "pm10", "no2", "o3"]:
            if param in df.columns:
                jumps = self.check_sudden_jumps(df, param)
                if jumps > 0:
                    report["warnings"].append(
                        f"Parameter {param} has {jumps} sudden jumps detected"
                    )

        # Check completeness
        if "city" in df.columns:
            completeness = self.check_data_completeness(df)
            report["checks"]["completeness"] = completeness

            for city, metrics in completeness.items():
                if metrics["missing_ratio"] > self.max_missing_ratio:
                    report["warnings"].append(
                        f"City {city} has {metrics['missing_ratio']:.2%} missing data"
                    )

        if report["warnings"] or report["errors"]:
            report["valid"] = False

        return report

