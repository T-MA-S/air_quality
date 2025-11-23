# Словарь данных и источники

## Источники данных

### 1. OpenAQ API

**URL**: https://api.openaq.org/v2

**Описание**: Открытый API для данных о качестве воздуха со станций мониторинга по всему миру.

**Параметры качества воздуха**:
- **PM2.5** (pm25): Частицы диаметром ≤ 2.5 мкм, единица: µg/m³
- **PM10** (pm10): Частицы диаметром ≤ 10 мкм, единица: µg/m³
- **NO₂** (no2): Диоксид азота, единица: µg/m³
- **O₃** (o3): Озон, единица: µg/m³
- **CO** (co): Оксид углерода, единица: µg/m³
- **SO₂** (so2): Диоксид серы, единица: µg/m³

**Rate Limit**: 100 запросов/минуту (по умолчанию)

**Лицензия**: Open Data Commons Open Database License (ODbL)

**Документация**: https://docs.openaq.org/

### 2. Open-Meteo API

**URL**: https://api.open-meteo.com/v1

**Описание**: Бесплатный API для исторических и прогнозных метеорологических данных.

**Параметры погоды**:
- **temperature_2m**: Температура на высоте 2м, единица: °C
- **relative_humidity_2m**: Относительная влажность на высоте 2м, единица: %
- **precipitation**: Осадки, единица: mm
- **wind_speed_10m**: Скорость ветра на высоте 10м, единица: m/s
- **wind_direction_10m**: Направление ветра на высоте 10м, единица: градусы (0-360)

**Rate Limit**: Нет официального лимита, рекомендуется ≤ 100 запросов/минуту

**Лицензия**: Open Data, бесплатное использование

**Документация**: https://open-meteo.com/en/docs

## Схема данных

### Справочники (Dimensions)

#### dim_city

| Поле | Тип | Описание | Ограничения |
|------|-----|----------|-------------|
| city_id | INTEGER | Уникальный идентификатор города | PRIMARY KEY, AUTO_INCREMENT |
| city_name | VARCHAR(100) | Название города | NOT NULL |
| country | VARCHAR(2) | Код страны (ISO 3166-1 alpha-2) | NOT NULL |
| latitude | FLOAT | Широта | NOT NULL, -90 до 90 |
| longitude | FLOAT | Долгота | NOT NULL, -180 до 180 |
| timezone | VARCHAR(50) | Часовой пояс (IANA) | NOT NULL |
| openaq_city_name | VARCHAR(100) | Название города в OpenAQ (может отличаться) | NULL |
| created_at | TIMESTAMP | Дата создания записи | DEFAULT now() |
| updated_at | TIMESTAMP | Дата обновления записи | DEFAULT now() |

**Индексы**: city_name

#### dim_metric

| Поле | Тип | Описание | Ограничения |
|------|-----|----------|-------------|
| metric_id | INTEGER | Уникальный идентификатор метрики | PRIMARY KEY, AUTO_INCREMENT |
| metric_code | VARCHAR(20) | Код метрики (pm25, pm10, no2, o3, co, so2) | NOT NULL, UNIQUE |
| metric_name | VARCHAR(100) | Название метрики | NOT NULL |
| unit | VARCHAR(20) | Единица измерения | NOT NULL |
| description | TEXT | Описание метрики | NULL |
| created_at | TIMESTAMP | Дата создания записи | DEFAULT now() |

**Индексы**: metric_code

### Факт-таблицы (Facts)

#### fact_air_quality

| Поле | Тип | Описание | Ограничения |
|------|-----|----------|-------------|
| id | INTEGER | Уникальный идентификатор записи | PRIMARY KEY, AUTO_INCREMENT |
| timestamp | TIMESTAMP | Временная метка измерения | NOT NULL |
| city_id | INTEGER | Идентификатор города | NOT NULL, FK → dim_city |
| metric_id | INTEGER | Идентификатор метрики | NOT NULL, FK → dim_metric |
| value | FLOAT | Значение измерения | NOT NULL |
| value_ug_m3 | FLOAT | Значение в µg/m³ (нормализованное) | NULL |
| station_count | INTEGER | Количество станций, использованных для агрегации | NULL |
| latitude | FLOAT | Широта станции | NULL |
| longitude | FLOAT | Долгота станции | NULL |
| location_id | VARCHAR(50) | Идентификатор станции в OpenAQ | NULL |
| location_name | VARCHAR(200) | Название станции | NULL |
| created_at | TIMESTAMP | Дата создания записи | DEFAULT now() |

**Индексы**: 
- timestamp
- (city_id, metric_id)
- (city_id, timestamp)

**Ожидаемые диапазоны значений**:
- PM2.5: 0-500 µg/m³
- PM10: 0-1000 µg/m³
- NO₂: 0-500 µg/m³
- O₃: 0-500 µg/m³

#### fact_weather

| Поле | Тип | Описание | Ограничения |
|------|-----|----------|-------------|
| id | INTEGER | Уникальный идентификатор записи | PRIMARY KEY, AUTO_INCREMENT |
| timestamp | TIMESTAMP | Временная метка измерения | NOT NULL |
| city_id | INTEGER | Идентификатор города | NOT NULL, FK → dim_city |
| temperature_c | FLOAT | Температура в °C | NULL, -50 до 50 |
| humidity_percent | FLOAT | Относительная влажность в % | NULL, 0 до 100 |
| precipitation_mm | FLOAT | Осадки в мм | NULL, ≥ 0 |
| wind_speed_ms | FLOAT | Скорость ветра в м/с | NULL, ≥ 0 |
| wind_direction_deg | FLOAT | Направление ветра в градусах | NULL, 0 до 360 |
| latitude | FLOAT | Широта | NULL |
| longitude | FLOAT | Долгота | NULL |
| created_at | TIMESTAMP | Дата создания записи | DEFAULT now() |

**Индексы**:
- timestamp
- (city_id, timestamp)

## Представления (Views)

### v_air_quality
Объединяет fact_air_quality с dim_city и dim_metric для удобных запросов.

### v_weather
Объединяет fact_weather с dim_city.

### v_air_quality_weather
Объединяет данные о воздухе и погоде по timestamp и городу.

### v_daily_aggregates
Дневные агрегаты: средние, минимум, максимум значений, количество точек данных.

### v_moving_averages
Скользящие средние за 7 дней для метрик качества воздуха.

### v_risk_map
Карта рисков с уровнями (LOW, MODERATE, HIGH) на основе превышений нормативов.

## Ограничения данных

### Временные ограничения
- Данные доступны с 2013 года (OpenAQ)
- Частота обновления: от часовой до ежедневной (зависит от станции)
- Задержка данных: до 24 часов

### Географические ограничения
- Данные доступны только для городов со станциями мониторинга
- Не все города имеют данные по всем параметрам
- Покрытие варьируется по странам

### Качественные ограничения
- Пропуски данных: до 30% допустимо
- Внезапные скачки: проверяются через Z-score (порог: 3σ)
- Валидация диапазонов: значения вне ожидаемых диапазонов помечаются как предупреждения

## Нормативы качества воздуха

### ВОЗ (2021)
- PM2.5: 15 µg/m³ (годовая), 25 µg/m³ (24-часовая)
- PM10: 45 µg/m³ (годовая), 50 µg/m³ (24-часовая)
- NO₂: 25 µg/m³ (годовая), 200 µg/m³ (1-часовая)
- O₃: 100 µg/m³ (8-часовая)

### Используемые пороги в системе
- PM2.5: 15 µg/m³ (MODERATE), 25 µg/m³ (HIGH)
- PM10: 30 µg/m³ (MODERATE), 50 µg/m³ (HIGH)
- NO₂: 100 µg/m³ (MODERATE), 200 µg/m³ (HIGH)
- O₃: 120 µg/m³ (MODERATE), 180 µg/m³ (HIGH)

## Лицензии и использование

### OpenAQ
- **Лицензия**: Open Data Commons Open Database License (ODbL)
- **Использование**: Бесплатное, с указанием источника
- **Ограничения**: Нет коммерческих ограничений

### Open-Meteo
- **Лицензия**: Open Data
- **Использование**: Бесплатное, без ограничений
- **Ограничения**: Нет

## Происхождение данных (Data Lineage)

```
OpenAQ API → OpenAQ Client → Transform → Quality Check → fact_air_quality
Open-Meteo API → Open-Meteo Client → Transform → Quality Check → fact_weather
                                                                      ↓
                                                              v_air_quality_weather
                                                                      ↓
                                                                    Dashboard
```

## Версионирование датасетов

Текущая версия схемы: **v1.0**

Изменения схемы документируются в ADR и миграциях Alembic (если будут добавлены).

