"""Rate limiting utilities."""

import time
from collections import deque
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, max_requests: int, time_window: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds (default: 60 for per-minute)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()

    def acquire(self) -> bool:
        """
        Try to acquire permission for a request.

        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()

        # Remove old requests outside the time window
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()

        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True

        return False

    def wait_if_needed(self) -> None:
        """Wait if rate limit is exceeded."""
        if not self.acquire():
            # Calculate wait time
            oldest_request = self.requests[0]
            wait_time = self.time_window - (time.time() - oldest_request) + 1

            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                self.acquire()


class RetryHandler:
    """Handle retries with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        """
        Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            backoff_factor: Multiplier for delay on each retry
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def execute(self, func, *args, **kwargs):
        """
        Execute function with retries.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        delay = self.initial_delay

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                error_msg = str(e)
                
                if attempt < self.max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed ({error_type}): {error_msg}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                    delay *= self.backoff_factor
                else:
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed. Last error ({error_type}): {error_msg}"
                    )

        raise last_exception

