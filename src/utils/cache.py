"""Caching utilities with TTL support."""

import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional
from cachetools import TTLCache

from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FileCache:
    """File-based cache with TTL."""

    def __init__(self, cache_dir: str = None, ttl_seconds: int = None):
        """Initialize file cache."""
        self.cache_dir = Path(cache_dir or settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds or settings.cache_ttl_seconds

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        # Sanitize key for filesystem
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self.cache_dir / f"{safe_key}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                timestamp, value = data

            if time.time() - timestamp > self.ttl_seconds:
                cache_path.unlink()
                return None

            return value
        except Exception as e:
            logger.warning(f"Error reading cache for {key}: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache with timestamp."""
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump((time.time(), value), f)
        except Exception as e:
            logger.warning(f"Error writing cache for {key}: {e}")


class MemoryCache:
    """In-memory TTL cache."""

    def __init__(self, maxsize: int = 1000, ttl_seconds: int = None):
        """Initialize memory cache."""
        self.ttl_seconds = ttl_seconds or settings.cache_ttl_seconds
        self.cache = TTLCache(maxsize=maxsize, ttl=self.ttl_seconds)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        self.cache[key] = value


# Global cache instance
file_cache = FileCache()
memory_cache = MemoryCache()

