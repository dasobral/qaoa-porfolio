"""
Utilities and common functions for QAOA Portfolio Optimizer (QOPO)

This module provides general utility functions and helper classes
used throughout the portfolio optimization framework.

For data validation, see `validation.py`.
For performance monitoring, see `timing.py`.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

import logging
from typing import List, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Re-export from focused modules for backward compatibility
from .validation import DataValidator  # noqa: F401
from .timing import PerformanceTimer, performance_monitor  # noqa: F401

# ============================================================================
# Utility Functions
# ============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"

def validate_weights(weights: List[float], tolerance: float = 1e-6) -> bool:
    """Validate that portfolio weights sum to 1 and are non-negative."""
    if not weights or any(w < 0 for w in weights):
        return False
    return abs(sum(weights) - 1.0) <= tolerance

def normalize_weights(weights: List[float]) -> List[float]:
    """Normalize weights to sum to 1."""
    if not weights:
        return weights
    total = sum(weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]
