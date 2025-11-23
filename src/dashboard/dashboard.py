"""Streamlit dashboard for air quality and weather data."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text

from src.database.connection import get_engine
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_data(query: str) -> pd.DataFrame:
    """Load data from database using SQL query."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="Air Quality & Weather Dashboard",
        page_icon="🌍",
        layout="wide",
    )

    st.title("🌍 Air Quality & Weather Dashboard")
    st.markdown("Multi-city air quality and weather data aggregator")

    # Sidebar filters
    st.sidebar.header("Filters")

    # Load cities
    cities_df = load_data("SELECT DISTINCT city_name FROM dim_city ORDER BY city_name")
    cities = cities_df["city_name"].tolist() if not cities_df.empty else []

    selected_cities = st.sidebar.multiselect("Select Cities", cities, default=cities[:3] if cities else [])

    # Load metrics
    metrics_df = load_data("SELECT metric_code, metric_name FROM dim_metric ORDER BY metric_code")
    metrics = metrics_df["metric_code"].tolist() if not metrics_df.empty else ["pm25", "pm10"]

    # Filter default values to only include those that exist in metrics list
    default_metrics = ["pm25", "pm10"]
    valid_defaults = [m for m in default_metrics if m in metrics]
    # If no valid defaults, use first available metrics (or empty list)
    if not valid_defaults and metrics:
        valid_defaults = metrics[:2] if len(metrics) >= 2 else metrics

    selected_metrics = st.sidebar.multiselect(
        "Select Metrics", metrics, default=valid_defaults
    )

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(pd.Timestamp.now() - pd.Timedelta(days=7), pd.Timestamp.now()),
    )

    if not selected_cities or not selected_metrics:
        st.warning("Please select at least one city and one metric")
        return

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Correlations", "Risk Map", "Statistics"])

    with tab1:
        st.header("Air Quality Trends by City")

        city_filter = "', '".join(selected_cities)
        metric_filter = "', '".join(selected_metrics)

        query = f"""
        SELECT
            timestamp,
            city_name,
            metric_code,
            value_ug_m3
        FROM v_air_quality
        WHERE city_name IN ('{city_filter}')
            AND metric_code IN ('{metric_filter}')
            AND timestamp >= '{date_range[0]}'
            AND timestamp <= '{date_range[1]}'
        ORDER BY timestamp
        """

        df = load_data(query)

        if not df.empty:
            for metric in selected_metrics:
                metric_df = df[df["metric_code"] == metric]
                if not metric_df.empty:
                    fig = px.line(
                        metric_df,
                        x="timestamp",
                        y="value_ug_m3",
                        color="city_name",
                        title=f"{metric.upper()} Trends",
                        labels={"value_ug_m3": "Value (µg/m³)", "timestamp": "Date"},
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for selected filters")

    with tab2:
        st.header("Correlation Matrix: Air Quality vs Weather")

        city_filter = "', '".join(selected_cities)
        metric_filter = "', '".join(selected_metrics)

        query = f"""
        SELECT
            aq.value_ug_m3,
            aq.metric_code,
            w.temperature_c,
            w.humidity_percent,
            w.precipitation_mm,
            w.wind_speed_ms,
            w.wind_direction_deg
        FROM v_air_quality_weather aq
        JOIN v_weather w
            ON aq.timestamp = w.timestamp
            AND aq.city_name = w.city_name
        WHERE aq.city_name IN ('{city_filter}')
            AND aq.metric_code IN ('{metric_filter}')
            AND aq.timestamp >= '{date_range[0]}'
            AND aq.timestamp <= '{date_range[1]}'
        """

        df = load_data(query)

        if not df.empty:
            for metric in selected_metrics:
                metric_df = df[df["metric_code"] == metric]
                if not metric_df.empty:
                    # Select numeric columns for correlation
                    numeric_cols = [
                        "value_ug_m3",
                        "temperature_c",
                        "humidity_percent",
                        "precipitation_mm",
                        "wind_speed_ms",
                    ]
                    corr_df = metric_df[numeric_cols].corr()

                    fig = px.imshow(
                        corr_df,
                        title=f"{metric.upper()} Correlation Matrix",
                        color_continuous_scale="RdBu",
                        aspect="auto",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display correlation values
                    st.subheader(f"{metric.upper()} Correlations")
                    st.dataframe(corr_df["value_ug_m3"].sort_values(ascending=False))
        else:
            st.info("No data available for correlation analysis")

    with tab3:
        st.header("Risk Map: Air Quality Exceedances")

        city_filter = "', '".join(selected_cities)
        metric_filter = "', '".join(selected_metrics)

        query = f"""
        SELECT
            date,
            hour,
            city_name,
            metric_code,
            risk_level,
            COUNT(*) as count
        FROM v_risk_map
        WHERE city_name IN ('{city_filter}')
            AND metric_code IN ('{metric_filter}')
            AND date >= '{date_range[0]}'
            AND date <= '{date_range[1]}'
        GROUP BY date, hour, city_name, metric_code, risk_level
        ORDER BY date, hour
        """

        df = load_data(query)

        if not df.empty:
            for metric in selected_metrics:
                metric_df = df[df["metric_code"] == metric]
                if not metric_df.empty:
                    pivot_df = metric_df.pivot_table(
                        index=["date", "hour"],
                        columns="risk_level",
                        values="count",
                        fill_value=0,
                    ).reset_index()

                    fig = px.scatter(
                        pivot_df,
                        x="date",
                        y="hour",
                        size="HIGH",
                        color="MODERATE",
                        title=f"{metric.upper()} Risk Map",
                        labels={"date": "Date", "hour": "Hour of Day"},
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risk data available")

    with tab4:
        st.header("Statistics and Data Quality")

        city_filter = "', '".join(selected_cities)

        query = f"""
        SELECT
            city_name,
            metric_code,
            AVG(avg_value) as avg_value,
            MIN(min_value) as min_value,
            MAX(max_value) as max_value,
            AVG(data_points) as avg_data_points,
            AVG(avg_temperature) as avg_temperature,
            AVG(avg_wind_speed) as avg_wind_speed
        FROM v_daily_aggregates
        WHERE city_name IN ('{city_filter}')
            AND date >= '{date_range[0]}'
            AND date <= '{date_range[1]}'
        GROUP BY city_name, metric_code
        ORDER BY city_name, metric_code
        """

        df = load_data(query)

        if not df.empty:
            st.dataframe(df, use_container_width=True)

            # Missing data ratio
            st.subheader("Data Completeness")
            completeness_query = f"""
            SELECT
                city_name,
                metric_code,
                COUNT(*) as total_records,
                COUNT(value_ug_m3) as non_null_records,
                1.0 - COUNT(value_ug_m3)::float / NULLIF(COUNT(*), 0) as missing_ratio
            FROM v_air_quality
            WHERE city_name IN ('{city_filter}')
                AND timestamp >= '{date_range[0]}'
                AND timestamp <= '{date_range[1]}'
            GROUP BY city_name, metric_code
            """
            completeness_df = load_data(completeness_query)
            if not completeness_df.empty:
                st.dataframe(completeness_df, use_container_width=True)
        else:
            st.info("No statistics available")


if __name__ == "__main__":
    main()

