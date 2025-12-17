"""Streamlit dashboard for air quality and weather data."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text
import numpy as np

from src.database.connection import get_engine
from src.utils.logger import setup_logger
from src.data.models import CITIES

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

    # Use only active cities from CITIES (exclude removed cities like Moscow, Beijing, Saint Petersburg)
    active_city_names = [city.name for city in CITIES]
    
    # Verify cities exist in database
    cities_df = load_data("SELECT DISTINCT city_name FROM dim_city ORDER BY city_name")
    all_cities_in_db = cities_df["city_name"].tolist() if not cities_df.empty else []
    
    # Filter to only show active cities that exist in database
    cities = [city for city in active_city_names if city in all_cities_in_db]
    
    if not cities:
        st.warning("No active cities found in database. Please run the ETL pipeline first.")
        return

    selected_cities = st.sidebar.multiselect("Select Cities", cities, default=cities[:3] if len(cities) >= 3 else cities)

    # Load metrics (exclude SO2 and CO)
    metrics_df = load_data("SELECT metric_code, metric_name FROM dim_metric ORDER BY metric_code")
    all_metrics = metrics_df["metric_code"].tolist() if not metrics_df.empty else ["pm25", "pm10"]
    # Filter out SO2 and CO
    metrics = [m for m in all_metrics if m.lower() not in ["so2", "co"]]

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
    
    # Handle date_range - it can be a single date or tuple
    if isinstance(date_range, tuple) and len(date_range) == 2:
        date_from = date_range[0]
        date_to = date_range[1]
    elif isinstance(date_range, pd.Timestamp) or hasattr(date_range, 'date'):
        # Single date selected
        date_from = date_range
        date_to = pd.Timestamp.now()
    else:
        # Default to last 7 days
        date_from = pd.Timestamp.now() - pd.Timedelta(days=7)
        date_to = pd.Timestamp.now()

    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Trends", 
        "Correlations", 
        "Risk Map", 
        "Data Quality", 
        "Data Warehouse", 
        "Statistics", 
        "ML Predictions"
    ])

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
            AND timestamp >= '{date_from}'
            AND timestamp <= '{date_to}'
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
        st.markdown("**Особое внимание к PM2.5 корреляциям с температурой, ветром и осадками**")

        city_filter = "', '".join(selected_cities)
        metric_filter = "', '".join(selected_metrics)

        query = f"""
        SELECT
            value_ug_m3,
            metric_code,
            city_name,
            temperature_c,
            humidity_percent,
            precipitation_mm,
            wind_speed_ms,
            wind_direction_deg
        FROM v_air_quality_weather
        WHERE city_name IN ('{city_filter}')
            AND metric_code IN ('{metric_filter}')
            AND timestamp >= '{date_from}'
            AND timestamp <= '{date_to}'
        LIMIT 10000
        """

        df = load_data(query)

        if not df.empty:
            # Особый фокус на PM2.5
            if 'pm25' in selected_metrics:
                st.subheader("🔍 PM2.5 Correlations (Detailed)")
                pm25_df = df[df["metric_code"] == "pm25"]
                if not pm25_df.empty:
                    # Корреляции PM2.5 с погодными параметрами
                    pm25_corr_cols = [
                        "value_ug_m3",
                        "temperature_c",
                        "humidity_percent",
                        "precipitation_mm",
                        "wind_speed_ms",
                    ]
                    pm25_corr = pm25_df[pm25_corr_cols].corr()
                    
                    # Визуализация корреляций PM2.5
                    fig = px.imshow(
                        pm25_corr,
                        title="PM2.5 Correlation Matrix with Weather Parameters",
                        color_continuous_scale="RdBu",
                        aspect="auto",
                        text_auto=".2f",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Детальная таблица корреляций PM2.5
                    st.subheader("PM2.5 Correlations with Weather")
                    pm25_corr_series = pm25_corr["value_ug_m3"].drop("value_ug_m3").sort_values(ascending=False)
                    corr_display = pd.DataFrame({
                        "Weather Parameter": pm25_corr_series.index,
                        "Correlation with PM2.5": pm25_corr_series.values
                    })
                    st.dataframe(corr_display, use_container_width=True)
                    
                    # Графики рассеяния для PM2.5
                    st.subheader("PM2.5 vs Weather Parameters (Scatter Plots)")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if pm25_df["temperature_c"].notna().sum() > 0:
                            fig_temp = px.scatter(
                                pm25_df,
                                x="temperature_c",
                                y="value_ug_m3",
                                color="city_name",
                                title="PM2.5 vs Temperature",
                                labels={"temperature_c": "Temperature (°C)", "value_ug_m3": "PM2.5 (µg/m³)"},
                            )
                            # Добавляем простую линию тренда через numpy
                            try:
                                valid_data = pm25_df[["temperature_c", "value_ug_m3"]].dropna()
                                if len(valid_data) > 1:
                                    z = np.polyfit(valid_data["temperature_c"], valid_data["value_ug_m3"], 1)
                                    p = np.poly1d(z)
                                    x_trend = np.linspace(valid_data["temperature_c"].min(), valid_data["temperature_c"].max(), 100)
                                    fig_temp.add_trace(go.Scatter(
                                        x=x_trend,
                                        y=p(x_trend),
                                        mode='lines',
                                        name='Trend Line',
                                        line=dict(color='gray', dash='dash', width=2),
                                        showlegend=True
                                    ))
                            except Exception:
                                pass  # Если не удалось построить тренд, просто показываем без него
                            st.plotly_chart(fig_temp, use_container_width=True)
                        
                        if pm25_df["wind_speed_ms"].notna().sum() > 0:
                            fig_wind = px.scatter(
                                pm25_df,
                                x="wind_speed_ms",
                                y="value_ug_m3",
                                color="city_name",
                                title="PM2.5 vs Wind Speed",
                                labels={"wind_speed_ms": "Wind Speed (m/s)", "value_ug_m3": "PM2.5 (µg/m³)"},
                            )
                            # Добавляем простую линию тренда через numpy
                            try:
                                valid_data = pm25_df[["wind_speed_ms", "value_ug_m3"]].dropna()
                                if len(valid_data) > 1:
                                    z = np.polyfit(valid_data["wind_speed_ms"], valid_data["value_ug_m3"], 1)
                                    p = np.poly1d(z)
                                    x_trend = np.linspace(valid_data["wind_speed_ms"].min(), valid_data["wind_speed_ms"].max(), 100)
                                    fig_wind.add_trace(go.Scatter(
                                        x=x_trend,
                                        y=p(x_trend),
                                        mode='lines',
                                        name='Trend Line',
                                        line=dict(color='gray', dash='dash', width=2),
                                        showlegend=True
                                    ))
                            except Exception:
                                pass
                            st.plotly_chart(fig_wind, use_container_width=True)
                    
                    with col2:
                        if pm25_df["precipitation_mm"].notna().sum() > 0:
                            fig_precip = px.scatter(
                                pm25_df,
                                x="precipitation_mm",
                                y="value_ug_m3",
                                color="city_name",
                                title="PM2.5 vs Precipitation",
                                labels={"precipitation_mm": "Precipitation (mm)", "value_ug_m3": "PM2.5 (µg/m³)"},
                            )
                            # Добавляем простую линию тренда через numpy
                            try:
                                valid_data = pm25_df[["precipitation_mm", "value_ug_m3"]].dropna()
                                if len(valid_data) > 1:
                                    z = np.polyfit(valid_data["precipitation_mm"], valid_data["value_ug_m3"], 1)
                                    p = np.poly1d(z)
                                    x_trend = np.linspace(valid_data["precipitation_mm"].min(), valid_data["precipitation_mm"].max(), 100)
                                    fig_precip.add_trace(go.Scatter(
                                        x=x_trend,
                                        y=p(x_trend),
                                        mode='lines',
                                        name='Trend Line',
                                        line=dict(color='gray', dash='dash', width=2),
                                        showlegend=True
                                    ))
                            except Exception:
                                pass
                            st.plotly_chart(fig_precip, use_container_width=True)
                        
                        if pm25_df["humidity_percent"].notna().sum() > 0:
                            fig_hum = px.scatter(
                                pm25_df,
                                x="humidity_percent",
                                y="value_ug_m3",
                                color="city_name",
                                title="PM2.5 vs Humidity",
                                labels={"humidity_percent": "Humidity (%)", "value_ug_m3": "PM2.5 (µg/m³)"},
                            )
                            # Добавляем простую линию тренда через numpy
                            try:
                                valid_data = pm25_df[["humidity_percent", "value_ug_m3"]].dropna()
                                if len(valid_data) > 1:
                                    z = np.polyfit(valid_data["humidity_percent"], valid_data["value_ug_m3"], 1)
                                    p = np.poly1d(z)
                                    x_trend = np.linspace(valid_data["humidity_percent"].min(), valid_data["humidity_percent"].max(), 100)
                                    fig_hum.add_trace(go.Scatter(
                                        x=x_trend,
                                        y=p(x_trend),
                                        mode='lines',
                                        name='Trend Line',
                                        line=dict(color='gray', dash='dash', width=2),
                                        showlegend=True
                                    ))
                            except Exception:
                                pass
                            st.plotly_chart(fig_hum, use_container_width=True)
            
            # Общие корреляции для всех метрик
            st.subheader("All Metrics Correlations")
            for metric in selected_metrics:
                if metric == 'pm25' and 'pm25' in selected_metrics:
                    continue  # Уже показали выше
                    
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
        st.markdown("**Дни с превышениями нормативов качества воздуха**")

        city_filter = "', '".join(selected_cities)
        metric_filter = "', '".join(selected_metrics)

        # Дни с превышениями (summary)
        st.subheader("📅 Days with Exceedances Summary")
        exceedances_query = f"""
        SELECT
            date,
            city_name,
            metric_code,
            exceedance_hours,
            max_value,
            avg_value,
            has_exceedance
        FROM v_days_with_exceedances
        WHERE city_name IN ('{city_filter}')
            AND metric_code IN ('{metric_filter}')
            AND date >= '{date_from}'
            AND date <= '{date_to}'
            AND has_exceedance = TRUE
        ORDER BY date DESC, exceedance_hours DESC
        """
        
        exceedances_df = load_data(exceedances_query)
        if not exceedances_df.empty:
            st.dataframe(exceedances_df, use_container_width=True)
            
            # График дней с превышениями
            fig_exceed = px.bar(
                exceedances_df,
                x="date",
                y="exceedance_hours",
                color="city_name",
                facet_col="metric_code",
                title="Hours with Exceedances by Day",
                labels={"exceedance_hours": "Hours with Exceedance", "date": "Date"},
            )
            st.plotly_chart(fig_exceed, use_container_width=True)
        else:
            st.success("✅ No exceedances detected in the selected period!")

        # Детальная карта рисков
        st.subheader("🗺️ Detailed Risk Map (Hourly)")
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
            AND date >= '{date_from}'
            AND date <= '{date_to}'
        GROUP BY date, hour, city_name, metric_code, risk_level
        ORDER BY date, hour
        LIMIT 5000
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

                    # Check which risk level columns exist
                    size_col = "HIGH" if "HIGH" in pivot_df.columns else None
                    color_col = "MODERATE" if "MODERATE" in pivot_df.columns else "LOW" if "LOW" in pivot_df.columns else None
                    
                    if size_col or color_col:
                        fig = px.scatter(
                            pivot_df,
                            x="date",
                            y="hour",
                            size=size_col if size_col else None,
                            color=color_col if color_col else None,
                            title=f"{metric.upper()} Risk Map",
                            labels={"date": "Date", "hour": "Hour of Day"},
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No risk data available for {metric}")
        else:
            st.info("No risk data available")

    with tab4:
        st.header("🔍 Data Quality Checks")
        st.markdown("**Наглядная визуализация проверок качества данных**")

        city_filter = "', '".join(selected_cities)
        metric_filter = "', '".join(selected_metrics)

        # 1. Доля пропусков по городам/метрикам
        st.subheader("📊 Missing Data Ratio by City/Metric")
        
        # Используем прямой запрос к fact_air_quality для более точного расчета
        # Учитываем, что может быть несколько записей на один час (дубликаты)
        # Считаем уникальные часы через DATE_TRUNC
        # УБРАЛИ CROSS JOIN чтобы избежать дубликатов из-за нескольких city_id для одного города
        completeness_query = f"""
        WITH date_range AS (
            SELECT generate_series(
                DATE('{date_from}'),
                DATE('{date_to}'),
                '1 day'::interval
            )::date AS date
        ),
        unique_hours AS (
            SELECT DISTINCT
                DATE(aq.timestamp) AS date,
                DATE_TRUNC('hour', aq.timestamp) AS hour_timestamp,
                c.city_name,
                m.metric_code
            FROM fact_air_quality aq
            JOIN dim_city c ON aq.city_id = c.city_id
            JOIN dim_metric m ON aq.metric_id = m.metric_id
            WHERE c.city_name IN ('{city_filter}')
                AND m.metric_code IN ('{metric_filter}')
                AND DATE(aq.timestamp) >= '{date_from}'
                AND DATE(aq.timestamp) <= '{date_to}'
        ),
        daily_counts AS (
            SELECT
                date,
                city_name,
                metric_code,
                COUNT(DISTINCT hour_timestamp) AS actual_hours_per_day
            FROM unique_hours
            GROUP BY date, city_name, metric_code
        )
        SELECT
            dc.city_name,
            dc.metric_code,
            COUNT(DISTINCT dc.date) AS days_with_data,
            COALESCE(SUM(dc.actual_hours_per_day), 0) AS total_points,
            (SELECT COUNT(*) FROM date_range) * 24 AS total_expected,
            CASE 
                WHEN (SELECT COUNT(*) FROM date_range) * 24 > 0 
                THEN ROUND(COALESCE(SUM(dc.actual_hours_per_day), 0)::numeric / ((SELECT COUNT(*) FROM date_range) * 24) * 100, 2)
                ELSE 0
            END AS avg_completeness,
            CASE 
                WHEN (SELECT COUNT(*) FROM date_range) * 24 > 0 
                THEN ROUND(100.0 - COALESCE(SUM(dc.actual_hours_per_day), 0)::numeric / ((SELECT COUNT(*) FROM date_range) * 24) * 100, 2)
                ELSE 100.0
            END AS avg_missing
        FROM daily_counts dc
        GROUP BY dc.city_name, dc.metric_code
        ORDER BY dc.city_name, dc.metric_code
        """
        
        completeness_df = load_data(completeness_query)
        if not completeness_df.empty:
            # Таблица
            st.dataframe(completeness_df, use_container_width=True)
            
            # Визуализация
            fig_completeness = px.bar(
                completeness_df,
                x="city_name",
                y="avg_completeness",
                color="metric_code",
                title="Data Completeness by City and Metric (%)",
                labels={"avg_completeness": "Completeness (%)", "city_name": "City"},
                barmode="group"
            )
            st.plotly_chart(fig_completeness, use_container_width=True)
            
            # Heatmap полноты
            pivot_completeness = completeness_df.pivot_table(
                index="city_name",
                columns="metric_code",
                values="avg_completeness",
                aggfunc="mean"
            )
            if not pivot_completeness.empty:
                fig_heatmap = px.imshow(
                    pivot_completeness,
                    title="Completeness Heatmap (%)",
                    color_continuous_scale="RdYlGn",
                    aspect="auto",
                    text_auto=".1f",
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No completeness data available")

        # 2. Проверка диапазонов значений
        st.subheader("✅ Value Range Checks")
        range_query = f"""
        SELECT
            c.city_name,
            m.metric_code,
            COUNT(*) as total_records,
            COUNT(*) FILTER (WHERE 
                (m.metric_code = 'pm25' AND (aq.value_ug_m3 < 0 OR aq.value_ug_m3 > 500)) OR
                (m.metric_code = 'pm10' AND (aq.value_ug_m3 < 0 OR aq.value_ug_m3 > 1000)) OR
                (m.metric_code = 'no2' AND (aq.value_ug_m3 < 0 OR aq.value_ug_m3 > 500)) OR
                (m.metric_code = 'o3' AND (aq.value_ug_m3 < 0 OR aq.value_ug_m3 > 500))
            ) as range_violations,
            MIN(aq.value_ug_m3) as min_value,
            MAX(aq.value_ug_m3) as max_value,
            AVG(aq.value_ug_m3) as avg_value
        FROM fact_air_quality aq
        JOIN dim_city c ON aq.city_id = c.city_id
        JOIN dim_metric m ON aq.metric_id = m.metric_id
        WHERE c.city_name IN ('{city_filter}')
            AND m.metric_code IN ('{metric_filter}')
            AND aq.timestamp >= '{date_from}'
            AND aq.timestamp <= '{date_to}'
        GROUP BY c.city_name, m.metric_code
        """
        
        range_df = load_data(range_query)
        if not range_df.empty:
            range_df["violation_rate"] = (range_df["range_violations"] / range_df["total_records"] * 100).round(2)
            st.dataframe(range_df, use_container_width=True)
            
            # Визуализация нарушений
            if range_df["range_violations"].sum() > 0:
                fig_violations = px.bar(
                    range_df[range_df["range_violations"] > 0],
                    x="city_name",
                    y="range_violations",
                    color="metric_code",
                    title="Value Range Violations",
                    labels={"range_violations": "Number of Violations", "city_name": "City"},
                    barmode="group"
                )
                st.plotly_chart(fig_violations, use_container_width=True)
            else:
                st.success("✅ No value range violations detected!")

        # 3. Станции и метаданные
        st.subheader("📡 Station Count and Metadata")
        station_query = f"""
        SELECT
            c.city_name,
            m.metric_code,
            AVG(aq.station_count) as avg_stations,
            COUNT(DISTINCT aq.location_id) as unique_locations,
            COUNT(*) as total_records
        FROM fact_air_quality aq
        JOIN dim_city c ON aq.city_id = c.city_id
        JOIN dim_metric m ON aq.metric_id = m.metric_id
        WHERE c.city_name IN ('{city_filter}')
            AND m.metric_code IN ('{metric_filter}')
            AND aq.timestamp >= '{date_from}'
            AND aq.timestamp <= '{date_to}'
        GROUP BY c.city_name, m.metric_code
        """
        
        station_df = load_data(station_query)
        if not station_df.empty:
            st.dataframe(station_df, use_container_width=True)
            
            fig_stations = px.bar(
                station_df,
                x="city_name",
                y="avg_stations",
                color="metric_code",
                title="Average Number of Stations per City/Metric",
                labels={"avg_stations": "Average Stations", "city_name": "City"},
                barmode="group"
            )
            st.plotly_chart(fig_stations, use_container_width=True)

    with tab5:
        st.header("🏪 Data Warehouse Views")
        st.markdown("**Прямой доступ к витрине данных**")

        city_filter = "', '".join(selected_cities)
        metric_filter = "', '".join(selected_metrics)

        view_option = st.selectbox(
            "Select View",
            [
                "v_air_quality",
                "v_weather",
                "v_air_quality_weather",
                "v_daily_aggregates",
                "v_moving_averages",
                "v_air_quality_lags",
                "v_risk_map",
                "v_days_with_exceedances",
                "v_data_completeness"
            ]
        )

        if view_option == "v_air_quality":
            query = f"""
            SELECT *
            FROM v_air_quality
            WHERE city_name IN ('{city_filter}')
                AND metric_code IN ('{metric_filter}')
                AND timestamp >= '{date_from}'
                AND timestamp <= '{date_to}'
            ORDER BY timestamp DESC
            LIMIT 1000
            """
        elif view_option == "v_weather":
            query = f"""
            SELECT *
            FROM v_weather
            WHERE city_name IN ('{city_filter}')
                AND timestamp >= '{date_from}'
                AND timestamp <= '{date_to}'
            ORDER BY timestamp DESC
            LIMIT 1000
            """
        elif view_option == "v_air_quality_weather":
            query = f"""
            SELECT *
            FROM v_air_quality_weather
            WHERE city_name IN ('{city_filter}')
                AND metric_code IN ('{metric_filter}')
                AND timestamp >= '{date_from}'
                AND timestamp <= '{date_to}'
            ORDER BY timestamp DESC
            LIMIT 1000
            """
        elif view_option == "v_daily_aggregates":
            query = f"""
            SELECT *
            FROM v_daily_aggregates
            WHERE city_name IN ('{city_filter}')
                AND metric_code IN ('{metric_filter}')
                AND date >= '{date_from}'
                AND date <= '{date_to}'
            ORDER BY date DESC
            LIMIT 1000
            """
        elif view_option == "v_moving_averages":
            query = f"""
            SELECT *
            FROM v_moving_averages
            WHERE city_name IN ('{city_filter}')
                AND metric_code IN ('{metric_filter}')
                AND timestamp >= '{date_from}'
                AND timestamp <= '{date_to}'
            ORDER BY timestamp DESC
            LIMIT 1000
            """
        elif view_option == "v_air_quality_lags":
            query = f"""
            SELECT *
            FROM v_air_quality_lags
            WHERE city_name IN ('{city_filter}')
                AND metric_code IN ('{metric_filter}')
                AND timestamp >= '{date_from}'
                AND timestamp <= '{date_to}'
            ORDER BY timestamp DESC
            LIMIT 1000
            """
        elif view_option == "v_risk_map":
            query = f"""
            SELECT *
            FROM v_risk_map
            WHERE city_name IN ('{city_filter}')
                AND metric_code IN ('{metric_filter}')
                AND date >= '{date_from}'
                AND date <= '{date_to}'
            ORDER BY date DESC, hour DESC
            LIMIT 1000
            """
        elif view_option == "v_days_with_exceedances":
            query = f"""
            SELECT *
            FROM v_days_with_exceedances
            WHERE city_name IN ('{city_filter}')
                AND metric_code IN ('{metric_filter}')
                AND date >= '{date_from}'
                AND date <= '{date_to}'
            ORDER BY date DESC
            LIMIT 1000
            """
        elif view_option == "v_data_completeness":
            query = f"""
            SELECT *
            FROM v_data_completeness
            WHERE city_name IN ('{city_filter}')
                AND metric_code IN ('{metric_filter}')
                AND date >= '{date_from}'
                AND date <= '{date_to}'
            ORDER BY date DESC
            LIMIT 1000
            """

        df = load_data(query)
        
        if not df.empty:
            st.subheader(f"View: {view_option}")
            st.dataframe(df, use_container_width=True)
            
            # Экспорт данных
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{view_option}_{date_from}_{date_to}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"No data available in {view_option} for selected filters")

    with tab6:
        st.header("Statistics and Summary")

        city_filter = "', '".join(selected_cities)

        # Используем более быстрый запрос напрямую к fact таблицам вместо медленного v_daily_aggregates
        query = f"""
        SELECT
            dc.city_name,
            dm.metric_code,
            AVG(faq.value_ug_m3) as avg_value,
            MIN(faq.value_ug_m3) as min_value,
            MAX(faq.value_ug_m3) as max_value,
            COUNT(*) as total_records,
            AVG(fw.temperature_c) as avg_temperature,
            AVG(fw.wind_speed_ms) as avg_wind_speed
        FROM fact_air_quality faq
        JOIN dim_city dc ON faq.city_id = dc.city_id
        JOIN dim_metric dm ON faq.metric_id = dm.metric_id
        LEFT JOIN fact_weather fw ON faq.city_id = fw.city_id 
            AND DATE(faq.timestamp) = DATE(fw.timestamp)
        WHERE dc.city_name IN ('{city_filter}')
            AND dm.metric_code IN ('{metric_filter}')
            AND faq.timestamp >= '{date_from}'
            AND faq.timestamp <= '{date_to}'
        GROUP BY dc.city_name, dm.metric_code
        ORDER BY dc.city_name, dm.metric_code
        """

        df = load_data(query)

        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No statistics available")
    
    with tab7:
        st.header("🤖 PM2.5 ML Predictions & Back-testing")
        st.markdown("Machine Learning model for PM2.5 prediction using Gradient Boosting")
        
        # Выбор города для прогноза
        ml_city = st.selectbox("Select City for Prediction", cities, key="ml_city")
        
        # Используем тот же период дат, что и в других вкладках
        ml_date_from = date_from
        ml_date_to = date_to
        
        if ml_city:
            try:
                from src.ml.predictor import PM25Predictor
                import joblib
                from pathlib import Path
                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                
                # Пытаемся загрузить модель
                # Пробуем несколько возможных путей (для разных контейнеров)
                import os
                city_model_name = ml_city.lower().replace(' ', '_')
                possible_paths = [
                    f"/opt/airflow/models/pm25_model_{city_model_name}.pkl",  # В контейнере airflow-webserver
                    f"/app/models/pm25_model_{city_model_name}.pkl",  # В контейнере dashboard
                    f"./models/pm25_model_{city_model_name}.pkl",  # Относительный путь
                    f"models/pm25_model_{city_model_name}.pkl",  # Относительный путь без ./
                    os.path.join(os.getcwd(), "models", f"pm25_model_{city_model_name}.pkl"),  # Абсолютный путь
                ]
                
                model_path = None
                for path in possible_paths:
                    if Path(path).exists():
                        model_path = path
                        break
                
                if model_path and Path(model_path).exists():
                    predictor = PM25Predictor()
                    try:
                        predictor.load(model_path)
                        st.success(f"✅ Model loaded for {ml_city}")
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        model_path = None
                    
                    # Загружаем данные за выбранный период + исторические данные для лагов
                    # Нужно загрузить данные на 24 часа раньше для создания лагов
                    from datetime import timedelta
                    historical_date_from = ml_date_from - timedelta(days=2)  # +2 дня для лагов и скользящих средних
                    
                    forecast_query = f"""
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
                    WHERE c.city_name = '{ml_city}'
                        AND m.metric_code = 'pm25'
                        AND aq.timestamp >= '{historical_date_from}'
                        AND aq.timestamp <= '{ml_date_to}'
                    ORDER BY aq.timestamp
                    """
                    
                    recent_data = load_data(forecast_query)
                    
                    if not recent_data.empty:
                        recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp'])
                        recent_data = recent_data.sort_values('timestamp')
                        # Устанавливаем timestamp как индекс для создания лагов
                        recent_data = recent_data.set_index('timestamp')
                        
                        # Заполняем базовые признаки перед созданием лагов
                        base_cols = ['temperature_c', 'humidity_percent', 'precipitation_mm', 
                                   'wind_speed_ms', 'wind_direction_deg', 'pm10']
                        for col in base_cols:
                            if col not in recent_data.columns:
                                recent_data[col] = 0
                            elif recent_data[col].notna().sum() > 0:
                                recent_data[col] = recent_data[col].fillna(recent_data[col].median())
                            else:
                                recent_data[col] = 0
                        
                        # Удаляем строки, где pm25 отсутствует (для сравнения)
                        recent_data = recent_data.dropna(subset=['pm25'])
                        
                        if len(recent_data) == 0:
                            st.warning("No data with PM2.5 values available for the selected period")
                        else:
                            # Используем prepare_features из модели для консистентности
                            # prepare_features теперь сам создаст все признаки (лаги, временные и т.д.)
                            try:
                                # Подготавливаем данные так же, как при обучении
                                # prepare_features создаст лаги, временные признаки и т.д. автоматически
                                X_pred = predictor.prepare_features(recent_data, target_col='pm25')
                                
                                # Удаляем строки, где лаги не могут быть вычислены (первые строки)
                                if 'pm25_lag_1h' in X_pred.columns:
                                    valid_mask = X_pred['pm25_lag_1h'].notna()
                                    X_pred = X_pred[valid_mask]
                                    recent_data = recent_data[valid_mask]
                                
                                # Дополнительная проверка на NaN и inf
                                if X_pred.isna().any().any():
                                    st.warning("Some features still contain NaN after preparation. Filling with 0.")
                                    X_pred = X_pred.fillna(0)
                                
                                if np.isinf(X_pred.values).any():
                                    st.warning("Some features contain infinite values. Replacing with 0.")
                                    X_pred = X_pred.replace([np.inf, -np.inf], 0)
                                
                                if len(X_pred) == 0:
                                    st.warning("No valid data after feature preparation (need historical data for lags)")
                                else:
                                    # Фильтруем только данные за выбранный период (после создания лагов)
                                    # Первые 24 строки нужны только для создания лагов, не для предсказания
                                    # Преобразуем ml_date_from и ml_date_to в datetime с временной зоной, если индекс имеет временную зону
                                    if isinstance(recent_data.index, pd.DatetimeIndex):
                                        # Если индекс имеет временную зону, преобразуем даты в datetime с той же зоной
                                        if recent_data.index.tz is not None:
                                            ml_date_from_dt = pd.Timestamp(ml_date_from).tz_localize(recent_data.index.tz) if pd.Timestamp(ml_date_from).tz is None else pd.Timestamp(ml_date_from)
                                            ml_date_to_dt = pd.Timestamp(ml_date_to).tz_localize(recent_data.index.tz) if pd.Timestamp(ml_date_to).tz is None else pd.Timestamp(ml_date_to)
                                        else:
                                            ml_date_from_dt = pd.Timestamp(ml_date_from)
                                            ml_date_to_dt = pd.Timestamp(ml_date_to)
                                        prediction_mask = (recent_data.index >= ml_date_from_dt) & (recent_data.index <= ml_date_to_dt)
                                    else:
                                        # Если индекс не DatetimeIndex, используем обычное сравнение
                                        prediction_mask = (recent_data.index >= ml_date_from) & (recent_data.index <= ml_date_to)
                                    
                                    recent_data_filtered = recent_data[prediction_mask]
                                    X_pred_filtered = X_pred[prediction_mask]
                                    
                                    if len(X_pred_filtered) == 0:
                                        st.warning("No data in selected period after filtering")
                                    else:
                                        # Прогнозы только для выбранного периода
                                        # ВАЖНО: Используем уже подготовленные признаки напрямую, 
                                        # а не вызываем predict с данными (который пересоздаст признаки)
                                        # Убеждаемся, что признаки в правильном порядке
                                        if predictor.feature_names and len(predictor.feature_names) > 0:
                                            # Упорядочиваем признаки согласно модели
                                            X_pred_filtered = X_pred_filtered[predictor.feature_names]
                                        
                                        # Делаем предсказания напрямую на подготовленных признаках
                                        predictions = predictor.model.predict(X_pred_filtered)
                                        
                                        # Проверка на NaN в предсказаниях
                                        if np.isnan(predictions).any():
                                            st.warning("Some predictions are NaN. This may indicate data quality issues.")
                                            predictions = np.nan_to_num(predictions, nan=0.0)
                                        
                                        # График сравнения реальных и предсказанных значений
                                        st.subheader(f"Predictions vs Actual ({ml_date_from.strftime('%Y-%m-%d')} to {ml_date_to.strftime('%Y-%m-%d')})")
                                        
                                        # Создаем DataFrame для сравнения
                                        if isinstance(recent_data_filtered.index, pd.DatetimeIndex):
                                            comparison_df = pd.DataFrame({
                                                'timestamp': recent_data_filtered.index,
                                                'Actual': recent_data_filtered['pm25'].values,
                                                'Predicted': predictions
                                            })
                                        else:
                                            comparison_df = pd.DataFrame({
                                                'timestamp': recent_data_filtered['timestamp'].values if 'timestamp' in recent_data_filtered.columns else range(len(recent_data_filtered)),
                                                'Actual': recent_data_filtered['pm25'].values,
                                                'Predicted': predictions
                                            })
                                        
                                        # Метрики производительности
                                        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                                        rmse = np.sqrt(mean_squared_error(comparison_df['Actual'], comparison_df['Predicted']))
                                        r2 = r2_score(comparison_df['Actual'], comparison_df['Predicted'])
                                        mae = mean_absolute_error(comparison_df['Actual'], comparison_df['Predicted'])
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("RMSE", f"{rmse:.2f} µg/m³")
                                        with col2:
                                            st.metric("R² Score", f"{r2:.3f}")
                                        with col3:
                                            st.metric("MAE", f"{mae:.2f} µg/m³")
                                        
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=comparison_df['timestamp'],
                                            y=comparison_df['Actual'],
                                            mode='lines+markers',
                                            name='Actual PM2.5',
                                            line=dict(color='blue', width=2)
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=comparison_df['timestamp'],
                                            y=comparison_df['Predicted'],
                                            mode='lines+markers',
                                            name='Predicted PM2.5',
                                            line=dict(color='red', width=2, dash='dash')
                                        ))
                                        fig.update_layout(
                                            title="PM2.5: Actual vs Predicted",
                                            xaxis_title="Date",
                                            yaxis_title="PM2.5 (µg/m³)",
                                            hovermode='x unified',
                                            height=500
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Scatter plot для визуализации точности
                                        st.subheader("Prediction Accuracy Scatter")
                                        scatter_df = comparison_df.dropna()
                                        if len(scatter_df) > 0:
                                            scatter_fig = px.scatter(
                                                scatter_df,
                                                x='Actual',
                                                y='Predicted',
                                                title="Actual vs Predicted PM2.5",
                                                labels={'Actual': 'Actual PM2.5 (µg/m³)', 'Predicted': 'Predicted PM2.5 (µg/m³)'}
                                            )
                                            # Добавляем идеальную линию
                                            max_val = max(scatter_df['Actual'].max(), scatter_df['Predicted'].max())
                                            scatter_fig.add_trace(go.Scatter(
                                                x=[0, max_val],
                                                y=[0, max_val],
                                                mode='lines',
                                                name='Perfect Prediction',
                                                line=dict(color='green', dash='dash')
                                            ))
                                            st.plotly_chart(scatter_fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error making predictions: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                    else:
                        st.warning(f"No data available for {ml_city} in the selected period ({ml_date_from} to {ml_date_to})")
                else:
                    st.warning(f"⚠️ Model not found for {ml_city}. Please train the model first.")
                    st.info("Run: `python train_ml_model.py` to train models for all cities")
                    
                    # Кнопка для обучения модели
                    if st.button("Train Model Now"):
                        with st.spinner("Training model... This may take a few minutes."):
                            import subprocess
                            import sys
                            result = subprocess.run(
                                [sys.executable, "/opt/airflow/project/train_ml_model.py"],
                                capture_output=True,
                                text=True
                            )
                            if result.returncode == 0:
                                st.success("Model trained successfully! Refresh the page to see predictions.")
                            else:
                                st.error("Error training model")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

