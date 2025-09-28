"""
Test configuration and fixtures for QAOA Portfolio Optimizer tests.

This module provides common fixtures, utilities, and configuration for all tests.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import shutil
from pathlib import Path

from qaoa_portfolio.config import ConfigManager
from qaoa_portfolio.data_loader import MarketDataLoader
from qaoa_portfolio.exceptions import MarketDataError, DataValidationError


# ============================================================================
# Event Loop Management for Async Tests
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for test configuration."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_config_dir):
    """Create a test configuration manager with isolated settings."""
    config_data = {
        "data_sources": {
            "cache_enabled": False,  # Disable cache for tests
            "cache_duration_days": 1,
            "free_tier_mode": True,
            "default": "yfinance"
        },
        "portfolio": {
            "default_size": 3
        },
        "performance": {
            "conservative_rate_limiting": False  # Disable for faster tests
        },
        "logging": {
            "show_free_tier_tips": False  # Reduce noise in tests
        },
        "free_tier": {
            "yahoo_finance": {
                "rate_limit_per_minute": 60
            }
        }
    }

    config_manager = ConfigManager()
    config_manager.config = config_data  # Use .config instead of ._config
    return config_manager


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def sample_stock_symbols():
    """Sample stock symbols for testing."""
    return ['AAPL', 'MSFT', 'GOOGL']


@pytest.fixture
def sample_crypto_symbols():
    """Sample crypto symbols for testing."""
    return ['BTC-USD', 'ETH-USD', 'ADA-USD']


@pytest.fixture
def sample_date_range():
    """Sample date range for testing."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


@pytest.fixture
def mock_price_data():
    """Create mock price data for testing."""
    def _create_mock_data(symbols: List[str], days: int = 30) -> pd.DataFrame:
        """Create realistic mock price data."""
        date_range = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )

        data = {}
        for symbol in symbols:
            # Generate realistic price movements
            base_price = np.random.uniform(50, 200)
            returns = np.random.normal(0.001, 0.02, len(date_range))
            prices = [base_price]

            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            # Create OHLCV data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col == 'volume':
                    data[(symbol, col)] = np.random.randint(1000000, 10000000, len(date_range))
                elif col == 'high':
                    data[(symbol, col)] = [p * np.random.uniform(1.00, 1.05) for p in prices]
                elif col == 'low':
                    data[(symbol, col)] = [p * np.random.uniform(0.95, 1.00) for p in prices]
                elif col == 'open':
                    data[(symbol, col)] = [p * np.random.uniform(0.98, 1.02) for p in prices]
                else:  # close
                    data[(symbol, col)] = prices

        df = pd.DataFrame(data, index=date_range)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['symbol', 'price_type'])
        return df

    return _create_mock_data


@pytest.fixture
def mock_invalid_data():
    """Create mock invalid data for testing error handling."""
    def _create_invalid_data(issue_type: str = "missing_data") -> pd.DataFrame:
        """Create data with specific issues for testing validation."""
        if issue_type == "missing_data":
            # Create data with missing values
            data = {
                ('AAPL', 'close'): [100, np.nan, 102, 103],
                ('AAPL', 'volume'): [1000, 1100, np.nan, 1300],
            }
        elif issue_type == "negative_prices":
            # Create data with negative prices
            data = {
                ('AAPL', 'close'): [100, -50, 102, 103],
                ('AAPL', 'volume'): [1000, 1100, 1200, 1300],
            }
        elif issue_type == "zero_volume":
            # Create data with zero volume
            data = {
                ('AAPL', 'close'): [100, 101, 102, 103],
                ('AAPL', 'volume'): [1000, 0, 1200, 1300],
            }
        else:
            raise ValueError(f"Unknown issue type: {issue_type}")

        date_range = pd.date_range('2023-01-01', periods=4, freq='D')
        df = pd.DataFrame(data, index=date_range)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['symbol', 'price_type'])
        return df

    return _create_invalid_data


# ============================================================================
# Market Data Loader Fixtures
# ============================================================================

@pytest.fixture
def market_loader(test_config):
    """Create a MarketDataLoader instance with test configuration."""
    with patch('qaoa_portfolio.data_loader.config', test_config):
        loader = MarketDataLoader()
        return loader


@pytest.fixture
def mock_yfinance():
    """Mock yfinance module for testing without network calls."""
    with patch('qaoa_portfolio.data_loader.yf') as mock_yf:
        yield mock_yf


# ============================================================================
# Test Utilities
# ============================================================================

class TestDataValidator:
    """Utility class for validating test data and results."""

    @staticmethod
    def validate_dataframe_structure(df: pd.DataFrame, expected_symbols: List[str]) -> bool:
        """Validate that DataFrame has expected structure."""
        if not isinstance(df.columns, pd.MultiIndex):
            return False

        if df.columns.names != ['symbol', 'price_type']:
            return False

        symbols = df.columns.get_level_values('symbol').unique().tolist()
        return set(symbols) == set(expected_symbols)

    @staticmethod
    def validate_price_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate price data quality and return metrics."""
        issues = {
            'missing_data': df.isnull().sum().sum(),
            'negative_prices': 0,
            'zero_volume': 0,
            'data_points': len(df)
        }

        # Check for negative prices
        price_columns = [col for col in df.columns if col[1] in ['open', 'high', 'low', 'close']]
        for col in price_columns:
            issues['negative_prices'] += (df[col] < 0).sum()

        # Check for zero volume
        volume_columns = [col for col in df.columns if col[1] == 'volume']
        for col in volume_columns:
            issues['zero_volume'] += (df[col] == 0).sum()

        return issues


@pytest.fixture
def test_validator():
    """Provide test validation utilities."""
    return TestDataValidator()


# ============================================================================
# Error Simulation Fixtures
# ============================================================================

@pytest.fixture
def network_error_simulator():
    """Simulate network errors for testing error handling."""
    def _simulate_error(error_type: str = "timeout"):
        if error_type == "timeout":
            return TimeoutError("Network timeout")
        elif error_type == "connection":
            return ConnectionError("Connection failed")
        elif error_type == "http":
            return Exception("HTTP 500 Internal Server Error")
        else:
            return Exception(f"Unknown error: {error_type}")

    return _simulate_error


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}

        def start_timer(self, name: str):
            import time
            self.metrics[name] = {'start': time.time()}

        def end_timer(self, name: str):
            import time
            if name in self.metrics:
                self.metrics[name]['end'] = time.time()
                self.metrics[name]['duration'] = (
                    self.metrics[name]['end'] - self.metrics[name]['start']
                )

        def get_duration(self, name: str) -> float:
            return self.metrics.get(name, {}).get('duration', 0.0)

    return PerformanceMonitor()


# ============================================================================
# Async Test Helpers
# ============================================================================

@pytest.fixture
def async_test_runner():
    """Helper for running async tests with proper error handling."""
    async def _run_async_test(coro, expected_exception=None):
        """Run async coroutine and handle exceptions gracefully."""
        try:
            result = await coro
            if expected_exception:
                pytest.fail(f"Expected {expected_exception.__name__} but no exception was raised")
            return result
        except Exception as e:
            if expected_exception and isinstance(e, expected_exception):
                return e  # Return the exception for further testing
            elif expected_exception:
                pytest.fail(f"Expected {expected_exception.__name__} but got {type(e).__name__}: {e}")
            else:
                # Re-raise unexpected exceptions with context
                raise AssertionError(f"Unexpected exception in async test: {type(e).__name__}: {e}") from e

    return _run_async_test