#!/usr/bin/env python3
"""Script to check if data is being written to the database."""

import sys
sys.path.insert(0, '/opt/airflow/src')

from src.database.connection import get_engine
from src.utils.logger import setup_logger
from sqlalchemy import text
import pandas as pd

logger = setup_logger(__name__)

def check_database():
    """Check database for recent data."""
    engine = get_engine()
    
    queries = {
        "Cities": "SELECT COUNT(*) as count FROM dim_city",
        "Metrics": "SELECT COUNT(*) as count FROM dim_metric",
        "Air Quality Records (last 7 days)": """
            SELECT COUNT(*) as count 
            FROM fact_air_quality 
            WHERE timestamp >= NOW() - INTERVAL '7 days'
        """,
        "Air Quality Records (last 24 hours)": """
            SELECT COUNT(*) as count 
            FROM fact_air_quality 
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
        """,
        "Weather Records (last 7 days)": """
            SELECT COUNT(*) as count 
            FROM fact_weather 
            WHERE timestamp >= NOW() - INTERVAL '7 days'
        """,
        "Weather Records (last 24 hours)": """
            SELECT COUNT(*) as count 
            FROM fact_weather 
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
        """,
        "Latest Air Quality Record": """
            SELECT MAX(timestamp) as latest_timestamp 
            FROM fact_air_quality
        """,
        "Latest Weather Record": """
            SELECT MAX(timestamp) as latest_timestamp 
            FROM fact_weather
        """,
        "Records by City (last 7 days)": """
            SELECT 
                c.city_name,
                COUNT(aq.id) as aq_count,
                COUNT(w.id) as weather_count
            FROM dim_city c
            LEFT JOIN fact_air_quality aq ON c.city_id = aq.city_id 
                AND aq.timestamp >= NOW() - INTERVAL '7 days'
            LEFT JOIN fact_weather w ON c.city_id = w.city_id 
                AND w.timestamp >= NOW() - INTERVAL '7 days'
            GROUP BY c.city_id, c.city_name
            ORDER BY c.city_name
        """
    }
    
    print("\n" + "="*60)
    print("DATABASE STATUS CHECK")
    print("="*60 + "\n")
    
    with engine.connect() as conn:
        for name, query in queries.items():
            try:
                result = pd.read_sql(text(query), conn)
                if "latest_timestamp" in result.columns:
                    latest = result.iloc[0]["latest_timestamp"]
                    print(f"{name}: {latest if pd.notna(latest) else 'No data'}")
                elif "city_name" in result.columns:
                    print(f"\n{name}:")
                    for _, row in result.iterrows():
                        print(f"  {row['city_name']}: {row['aq_count']} AQ, {row['weather_count']} Weather")
                else:
                    count = result.iloc[0]["count"]
                    print(f"{name}: {count}")
            except Exception as e:
                print(f"{name}: ERROR - {e}")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    check_database()

