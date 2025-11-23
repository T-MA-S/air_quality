"""Configuration management for the project."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="air_quality", env="POSTGRES_DB")
    postgres_user: str = Field(default="air_quality_user", env="POSTGRES_USER")
    postgres_password: str = Field(default="air_quality_password", env="POSTGRES_PASSWORD")

    # OpenAQ API
    openaq_api_base_url: str = Field(
        default="https://api.openaq.org/v3", env="OPENAQ_API_BASE_URL"
    )
    openaq_api_key: Optional[str] = Field(default=None, env="OPENAQ_API_KEY")
    openaq_rate_limit: int = Field(default=100, env="OPENAQ_RATE_LIMIT")

    # Open-Meteo API
    open_meteo_base_url: str = Field(
        default="https://api.open-meteo.com/v1", env="OPEN_METEO_BASE_URL"
    )

    # Cache
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    cache_dir: str = Field(default="./cache", env="CACHE_DIR")

    # Data quality
    max_missing_ratio: float = Field(default=0.3, env="MAX_MISSING_RATIO")
    min_data_points_per_day: int = Field(default=20, env="MIN_DATA_POINTS_PER_DAY")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()

