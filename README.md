# Air Quality & Weather Aggregator

Проект для агрегации данных о качестве воздуха и погоде из открытых REST API с созданием витрины данных и дашборда.

## Описание проекта

Система собирает данные о качестве воздуха (PM2.5, PM10, NO₂, O₃) из OpenAQ API и метеорологические данные (температура, осадки, ветер) из Open-Meteo API для множества городов. Данные обрабатываются, проверяются на качество и загружаются в витрину данных PostgreSQL. Дашборд на Streamlit предоставляет визуализации трендов, корреляций и карту рисков.

## Структура проекта

```
air_quality/
├── src/                    # Исходный код
│   ├── api/               # API клиенты (OpenAQ, Open-Meteo)
│   ├── data/              # Модели данных, трансформации, проверки качества
│   ├── database/          # Работа с БД (схемы, загрузка)
│   ├── etl/               # ETL процессы
│   ├── ml/                # ML модели для прогноза PM2.5
│   ├── dashboard/          # Streamlit дашборд
│   ├── utils/              # Утилиты (логирование, кэш, rate limiting)
│   └── config.py          # Конфигурация
├── tests/                 # Тесты
├── dags/                  # Airflow DAGs
├── sql/                   # SQL скрипты и представления
├── docs/                  # Документация
├── requirements.txt       # Зависимости Python
├── Dockerfile            # Docker образ
├── docker-compose.yml    # Docker Compose конфигурация
└── pyproject.toml        # Конфигурация инструментов разработки
```

## Требования

- Python 3.11+
- PostgreSQL 15+
- Docker и Docker Compose (опционально)

## Установка

### Локальная установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd air_quality
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Создайте файл `.env` на основе `.env.example`:
```bash
cp .env.example .env
# Отредактируйте .env с вашими настройками
```

5. Настройте базу данных:
```bash
# Создайте БД PostgreSQL
createdb air_quality

# Инициализируйте схему
python -c "from src.database.schema import create_schema; create_schema()"
```

### Установка с Docker

1. Запустите все сервисы:
```bash
docker-compose up -d
```

2. Инициализируйте схему БД:
```bash
docker-compose exec app python -c "from src.database.schema import create_schema; create_schema()"
```

## Использование

### Запуск ETL процесса

```bash
# Запуск для всех городов за последние 7 дней
python -m src.main

# Запуск для конкретных городов
python -m src.main --cities Moscow London Paris --days 30

# Запуск с указанием дат
python -m src.main --date-from 2024-01-01 --date-to 2024-01-31
```

### Запуск дашборда

```bash
streamlit run src/dashboard/dashboard.py
```

Дашборд будет доступен по адресу: http://localhost:8501

### Запуск Airflow

```bash
# С Docker Compose
docker-compose up airflow-webserver airflow-scheduler

# Или локально (после настройки Airflow)
airflow webserver
airflow scheduler
```

Airflow UI будет доступен по адресу: http://localhost:8080

### Обучение ML модели

```python
from src.ml.predictor import PM25Predictor
from src.database.connection import get_engine
import pandas as pd
from sqlalchemy import text

# Загрузите данные из БД
engine = get_engine()
query = """
SELECT 
    w.temperature_c,
    w.humidity_percent,
    w.precipitation_mm,
    w.wind_speed_ms,
    w.wind_direction_deg,
    aq.value_ug_m3 as pm25
FROM fact_weather w
JOIN fact_air_quality aq ON w.city_id = aq.city_id AND w.timestamp = aq.timestamp
JOIN dim_metric m ON aq.metric_id = m.metric_id
WHERE m.metric_code = 'pm25'
"""
df = pd.read_sql(text(query), engine)

# Обучите модель
predictor = PM25Predictor()
metrics = predictor.train(df.drop('pm25', axis=1), df['pm25'])
predictor.save()
```

## Тестирование

```bash
# Запуск всех тестов
pytest

# С покрытием
pytest --cov=src --cov-report=html

# Конкретный тест
pytest tests/test_api.py
```

## Методы и инструменты

### API клиенты
- **OpenAQ Client**: Извлечение данных о качестве воздуха
- **Open-Meteo Client**: Извлечение метеорологических данных
- Кэширование ответов для снижения нагрузки на API
- Rate limiting для соблюдения лимитов API
- Retry с экспоненциальной задержкой

### ETL процессы
- Экстракция данных из REST API
- Трансформация и нормализация (единицы измерения, временные зоны)
- Агрегация данных по станциям и городам
- Загрузка в витрину данных

### Проверки качества данных
- Проверка пропущенных значений
- Валидация диапазонов значений
- Проверка временной монотонности
- Обнаружение внезапных скачков
- Проверка полноты данных

### Витрина данных
- **dim_city**: Справочник городов
- **dim_metric**: Справочник метрик
- **fact_air_quality**: Факты о качестве воздуха
- **fact_weather**: Факты о погоде
- Представления для аналитики (v_air_quality, v_weather, v_daily_aggregates, v_moving_averages, v_risk_map)

### Дашборд
- Тренды PM2.5/PM10 по городам
- Матрица корреляций с метеопоказателями
- Карта риска по дням/часам
- Статистика и метрики качества данных

### Оркестрация
- Airflow DAG для автоматического запуска ETL
- Ежечасный запуск
- Валидация качества данных
- Логирование и отчетность

## Результаты

Система предоставляет:
- Многогородовую витрину данных о качестве воздуха и погоде
- Автоматизированную загрузку данных по расписанию
- Контроль качества данных с предупреждениями
- Интерактивный дашборд для анализа
- ML модель для прогноза PM2.5

## Лицензия

MIT License

## Контакты

Для вопросов и предложений создайте issue в репозитории.

