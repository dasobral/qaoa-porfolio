"""
Performance monitoring utilities for QAOA Portfolio Optimizer (QOPO)

This module provides timing and performance monitoring tools.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

import logging
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class PerformanceTimer:
    """Context manager for timing code execution"""

    def __init__(self, name: str = "Operation", log_result: bool = True):
        self.name = name
        self.log_result = log_result
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.debug(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

        if self.log_result:
            if exc_type is None:
                logger.info(f"✓ {self.name} completed in {self.duration:.4f}s")
            else:
                logger.error(
                    f"✗ {self.name} failed after {self.duration:.4f}s: {exc_val}"
                )

    def get_duration(self) -> Optional[float]:
        """Returns the duration of the timed operation in seconds."""
        return self.duration


def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor the execution time of a function."""

    def wrapper(*args, **kwargs):
        with PerformanceTimer(f"{func.__module__}.{func.__name__}"):
            return func(*args, **kwargs)

    return wrapper
