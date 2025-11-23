-- Views for data warehouse

-- View: Air quality with city and metric names
CREATE OR REPLACE VIEW v_air_quality AS
SELECT
    aq.timestamp,
    c.city_name,
    c.country,
    m.metric_code,
    m.metric_name,
    m.unit,
    aq.value,
    aq.value_ug_m3,
    aq.station_count,
    aq.latitude,
    aq.longitude
FROM fact_air_quality aq
JOIN dim_city c ON aq.city_id = c.city_id
JOIN dim_metric m ON aq.metric_id = m.metric_id;

-- View: Weather with city names
CREATE OR REPLACE VIEW v_weather AS
SELECT
    w.timestamp,
    c.city_name,
    c.country,
    w.temperature_c,
    w.humidity_percent,
    w.precipitation_mm,
    w.wind_speed_ms,
    w.wind_direction_deg,
    w.latitude,
    w.longitude
FROM fact_weather w
JOIN dim_city c ON w.city_id = c.city_id;

-- View: Combined air quality and weather
CREATE OR REPLACE VIEW v_air_quality_weather AS
SELECT
    COALESCE(aq.timestamp, w.timestamp) AS timestamp,
    COALESCE(aq.city_name, w.city_name) AS city_name,
    COALESCE(aq.country, w.country) AS country,
    aq.metric_code,
    aq.metric_name,
    aq.value_ug_m3,
    w.temperature_c,
    w.humidity_percent,
    w.precipitation_mm,
    w.wind_speed_ms,
    w.wind_direction_deg
FROM v_air_quality aq
FULL OUTER JOIN v_weather w
    ON aq.timestamp = w.timestamp
    AND aq.city_name = w.city_name;

-- View: Daily aggregates
CREATE OR REPLACE VIEW v_daily_aggregates AS
SELECT
    DATE(aq.timestamp) AS date,
    c.city_name,
    c.country,
    m.metric_code,
    AVG(aq.value_ug_m3) AS avg_value,
    MIN(aq.value_ug_m3) AS min_value,
    MAX(aq.value_ug_m3) AS max_value,
    COUNT(*) AS data_points,
    AVG(w.temperature_c) AS avg_temperature,
    AVG(w.humidity_percent) AS avg_humidity,
    SUM(w.precipitation_mm) AS total_precipitation,
    AVG(w.wind_speed_ms) AS avg_wind_speed
FROM fact_air_quality aq
JOIN dim_city c ON aq.city_id = c.city_id
JOIN dim_metric m ON aq.metric_id = m.metric_id
LEFT JOIN fact_weather w
    ON aq.city_id = w.city_id
    AND DATE(aq.timestamp) = DATE(w.timestamp)
GROUP BY DATE(aq.timestamp), c.city_name, c.country, m.metric_code;

-- View: Moving averages (7-day)
CREATE OR REPLACE VIEW v_moving_averages AS
SELECT
    aq.timestamp,
    c.city_name,
    m.metric_code,
    aq.value_ug_m3,
    AVG(aq.value_ug_m3) OVER (
        PARTITION BY c.city_name, m.metric_code
        ORDER BY aq.timestamp
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
FROM fact_air_quality aq
JOIN dim_city c ON aq.city_id = c.city_id
JOIN dim_metric m ON aq.metric_id = m.metric_id;

-- View: Risk map (exceedances)
CREATE OR REPLACE VIEW v_risk_map AS
SELECT
    DATE(aq.timestamp) AS date,
    EXTRACT(HOUR FROM aq.timestamp) AS hour,
    c.city_name,
    m.metric_code,
    aq.value_ug_m3,
    CASE
        WHEN m.metric_code = 'pm25' AND aq.value_ug_m3 > 25 THEN 'HIGH'
        WHEN m.metric_code = 'pm25' AND aq.value_ug_m3 > 15 THEN 'MODERATE'
        WHEN m.metric_code = 'pm10' AND aq.value_ug_m3 > 50 THEN 'HIGH'
        WHEN m.metric_code = 'pm10' AND aq.value_ug_m3 > 30 THEN 'MODERATE'
        WHEN m.metric_code = 'no2' AND aq.value_ug_m3 > 200 THEN 'HIGH'
        WHEN m.metric_code = 'no2' AND aq.value_ug_m3 > 100 THEN 'MODERATE'
        WHEN m.metric_code = 'o3' AND aq.value_ug_m3 > 180 THEN 'HIGH'
        WHEN m.metric_code = 'o3' AND aq.value_ug_m3 > 120 THEN 'MODERATE'
        ELSE 'LOW'
    END AS risk_level,
    w.temperature_c,
    w.wind_speed_ms,
    w.precipitation_mm
FROM fact_air_quality aq
JOIN dim_city c ON aq.city_id = c.city_id
JOIN dim_metric m ON aq.metric_id = m.metric_id
LEFT JOIN fact_weather w
    ON aq.city_id = w.city_id
    AND aq.timestamp = w.timestamp;

