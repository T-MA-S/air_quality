"""Database connection management."""

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session

from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Global engine instance
_engine: Engine = None
_SessionLocal = None


def get_engine() -> Engine:
    """Get or create database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info(f"Database engine created for {settings.postgres_host}")
    return _engine


def get_session() -> Session:
    """Get database session."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal()


def close_engine():
    """Close database engine."""
    global _engine
    if _engine:
        _engine.dispose()
        _engine = None
        logger.info("Database engine closed")

