"""
Testing utilities and helper functions for QAOA Portfolio Optimizer tests.

This module provides common utilities, mock data generators, and test helpers
used across multiple test modules.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, MagicMock
import tempfile
import json
from pathlib import Path


# ============================================================================
# Mock Data Generators
# ============================================================================

class MockDataGenerator:
    """Generate realistic mock data for testing."""

    @staticmethod
    def create_realistic_price_data(
        symbols: List[str],
        days: int = 30,
        start_date: Optional[datetime] = None,
        volatility: float = 0.02,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create realistic mock price data with proper OHLCV structure.

        Args:
            symbols: List of asset symbols
            days: Number of trading days
            start_date: Start date (defaults to 30 days ago)
            volatility: Daily volatility (standard deviation of returns)
            seed: Random seed for reproducibility

        Returns:
            DataFrame with multi-level columns (symbol, price_type)
        """
        if seed is not None:
            np.random.seed(seed)

        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)

        date_range = pd.date_range(start=start_date, periods=days, freq='D')
        data = {}

        for symbol in symbols:
            # Generate price series with realistic characteristics
            base_price = np.random.uniform(50, 300)
            returns = np.random.normal(0.0005, volatility, days)

            # Create price series
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            # Generate OHLCV data
            for i, price in enumerate(prices):
                # Open: previous close + small gap
                if i == 0:
                    open_price = price
                else:
                    gap = np.random.normal(0, volatility * 0.5)
                    open_price = prices[i-1] * (1 + gap)

                # High and low based on intraday volatility
                intraday_vol = volatility * 0.7
                high_mult = 1 + abs(np.random.normal(0, intraday_vol))
                low_mult = 1 - abs(np.random.normal(0, intraday_vol))

                high_price = max(open_price, price) * high_mult
                low_price = min(open_price, price) * low_mult

                # Volume with realistic patterns
                base_volume = np.random.randint(1000000, 5000000)
                volume_mult = 1 + abs(np.random.normal(0, 0.3))
                volume = int(base_volume * volume_mult)

                # Store data
                data[(symbol, 'open')] = data.get((symbol, 'open'), []) + [open_price]
                data[(symbol, 'high')] = data.get((symbol, 'high'), []) + [high_price]
                data[(symbol, 'low')] = data.get((symbol, 'low'), []) + [low_price]
                data[(symbol, 'close')] = data.get((symbol, 'close'), []) + [price]
                data[(symbol, 'volume')] = data.get((symbol, 'volume'), []) + [volume]

        df = pd.DataFrame(data, index=date_range)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['symbol', 'price_type'])
        return df

    @staticmethod
    def create_data_with_issues(
        symbols: List[str],
        issue_type: str,
        days: int = 10
    ) -> pd.DataFrame:
        """
        Create mock data with specific data quality issues.

        Args:
            symbols: List of symbols
            issue_type: Type of issue ('missing', 'negative', 'zero_volume', 'outliers')
            days: Number of days

        Returns:
            DataFrame with the specified data issues
        """
        # Start with clean data
        df = MockDataGenerator.create_realistic_price_data(symbols, days, seed=42)

        if issue_type == 'missing':
            # Introduce missing values
            for symbol in symbols:
                # Make some random entries NaN
                mask = np.random.random(len(df)) < 0.1  # 10% missing
                df.loc[mask, (symbol, 'close')] = np.nan
                df.loc[mask[:len(mask)//2], (symbol, 'volume')] = np.nan

        elif issue_type == 'negative':
            # Introduce negative prices
            for symbol in symbols:
                idx = np.random.choice(df.index, size=2, replace=False)
                df.loc[idx, (symbol, 'close')] = -abs(df.loc[idx, (symbol, 'close')])

        elif issue_type == 'zero_volume':
            # Introduce zero volume
            for symbol in symbols:
                idx = np.random.choice(df.index, size=3, replace=False)
                df.loc[idx, (symbol, 'volume')] = 0

        elif issue_type == 'outliers':
            # Introduce extreme price outliers
            for symbol in symbols:
                idx = np.random.choice(df.index, size=1)[0]
                current_price = df.loc[idx, (symbol, 'close')]
                df.loc[idx, (symbol, 'close')] = current_price * 100  # 100x price spike

        return df

    @staticmethod
    def create_empty_data() -> pd.DataFrame:
        """Create empty DataFrame with proper structure."""
        df = pd.DataFrame()
        return df

    @staticmethod
    def create_minimal_valid_data(symbol: str = 'TEST') -> pd.DataFrame:
        """Create minimal valid data for a single symbol."""
        data = {
            (symbol, 'close'): [100.0, 101.0],
            (symbol, 'volume'): [1000, 1100]
        }
        df = pd.DataFrame(data, index=pd.date_range('2023-01-01', periods=2))
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['symbol', 'price_type'])
        return df


# ============================================================================
# Mock Configuration Utilities
# ============================================================================

class MockConfigManager:
    """Mock configuration manager for testing."""

    def __init__(self, config_data: Optional[Dict] = None):
        """Initialize with test configuration data."""
        self.config_data = config_data or self._default_test_config()

    def _default_test_config(self) -> Dict:
        """Create default test configuration."""
        return {
            "data_sources": {
                "cache_enabled": False,
                "cache_duration_days": 1,
                "free_tier_mode": True,
                "default": "yfinance"
            },
            "portfolio": {
                "default_size": 3
            },
            "performance": {
                "conservative_rate_limiting": False
            },
            "logging": {
                "show_free_tier_tips": False
            },
            "free_tier": {
                "yahoo_finance": {
                    "rate_limit_per_minute": 60
                }
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key."""
        keys = key.split('.')
        value = self.config_data

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-separated key."""
        keys = key.split('.')
        config = self.config_data

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value


# ============================================================================
# Test File Management
# ============================================================================

class TestFileManager:
    """Manage test files and temporary directories."""

    def __init__(self):
        self.temp_dirs = []
        self.temp_files = []

    def create_temp_directory(self) -> Path:
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def create_temp_file(self, content: str = "", suffix: str = ".txt") -> Path:
        """Create a temporary file with optional content."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
        temp_file.write(content)
        temp_file.close()

        temp_path = Path(temp_file.name)
        self.temp_files.append(temp_path)
        return temp_path

    def create_temp_json_file(self, data: Dict) -> Path:
        """Create a temporary JSON file."""
        json_content = json.dumps(data, indent=2)
        return self.create_temp_file(json_content, ".json")

    def cleanup(self):
        """Clean up all temporary files and directories."""
        import shutil

        for temp_file in self.temp_files:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass

        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except FileNotFoundError:
                pass

        self.temp_files.clear()
        self.temp_dirs.clear()


# ============================================================================
# Error Simulation
# ============================================================================

class ErrorSimulator:
    """Simulate various types of errors for testing."""

    @staticmethod
    def network_timeout():
        """Simulate network timeout error."""
        return TimeoutError("Network request timed out")

    @staticmethod
    def connection_error():
        """Simulate connection error."""
        return ConnectionError("Failed to establish connection")

    @staticmethod
    def http_error(status_code: int = 500):
        """Simulate HTTP error."""
        return Exception(f"HTTP {status_code} Error")

    @staticmethod
    def rate_limit_error():
        """Simulate rate limit error."""
        from qaoa_portfolio.exceptions import RateLimitError
        return RateLimitError("API rate limit exceeded")

    @staticmethod
    def data_validation_error():
        """Simulate data validation error."""
        from qaoa_portfolio.exceptions import DataValidationError
        return DataValidationError("Data validation failed")

    @staticmethod
    def market_data_error():
        """Simulate market data error."""
        from qaoa_portfolio.exceptions import MarketDataError
        return MarketDataError("Failed to load market data")


# ============================================================================
# Performance Testing Utilities
# ============================================================================

class PerformanceTracker:
    """Track performance metrics during tests."""

    def __init__(self):
        self.metrics = {}
        self.start_times = {}

    def start_timer(self, name: str):
        """Start timing a operation."""
        import time
        self.start_times[name] = time.perf_counter()

    def end_timer(self, name: str) -> float:
        """End timing and return duration."""
        import time
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")

        duration = time.perf_counter() - self.start_times[name]
        self.metrics[name] = duration
        del self.start_times[name]
        return duration

    def get_metric(self, name: str) -> Optional[float]:
        """Get a recorded metric."""
        return self.metrics.get(name)

    def assert_performance(self, name: str, max_duration: float):
        """Assert that an operation completed within time limit."""
        duration = self.get_metric(name)
        if duration is None:
            raise AssertionError(f"No metric recorded for '{name}'")
        if duration > max_duration:
            raise AssertionError(
                f"Operation '{name}' took {duration:.3f}s, "
                f"expected < {max_duration:.3f}s"
            )


# ============================================================================
# Assertion Helpers
# ============================================================================

class DataFrameAssertions:
    """Custom assertions for DataFrame testing."""

    @staticmethod
    def assert_has_multiindex_columns(df: pd.DataFrame, expected_levels: List[str]):
        """Assert DataFrame has MultiIndex columns with expected levels."""
        if not isinstance(df.columns, pd.MultiIndex):
            raise AssertionError("DataFrame does not have MultiIndex columns")

        if list(df.columns.names) != expected_levels:
            raise AssertionError(
                f"Column levels {df.columns.names} != expected {expected_levels}"
            )

    @staticmethod
    def assert_symbols_present(df: pd.DataFrame, expected_symbols: List[str]):
        """Assert all expected symbols are present in DataFrame."""
        if not isinstance(df.columns, pd.MultiIndex):
            raise AssertionError("DataFrame must have MultiIndex columns")

        actual_symbols = df.columns.get_level_values('symbol').unique().tolist()
        missing_symbols = set(expected_symbols) - set(actual_symbols)

        if missing_symbols:
            raise AssertionError(f"Missing symbols: {missing_symbols}")

    @staticmethod
    def assert_no_missing_data(df: pd.DataFrame):
        """Assert DataFrame has no missing values."""
        if df.isnull().any().any():
            missing_count = df.isnull().sum().sum()
            raise AssertionError(f"DataFrame has {missing_count} missing values")

    @staticmethod
    def assert_positive_prices(df: pd.DataFrame):
        """Assert all price columns have positive values."""
        price_columns = [col for col in df.columns if col[1] in ['open', 'high', 'low', 'close']]

        for col in price_columns:
            if (df[col] <= 0).any():
                negative_count = (df[col] <= 0).sum()
                raise AssertionError(f"Column {col} has {negative_count} non-positive values")

    @staticmethod
    def assert_valid_ohlc_relationships(df: pd.DataFrame):
        """Assert OHLC data has valid relationships (High >= Low, etc.)."""
        symbols = df.columns.get_level_values('symbol').unique()

        for symbol in symbols:
            try:
                high = df[(symbol, 'high')]
                low = df[(symbol, 'low')]
                open_price = df[(symbol, 'open')]
                close = df[(symbol, 'close')]

                # High should be >= Low
                if (high < low).any():
                    raise AssertionError(f"Symbol {symbol}: High < Low found")

                # High should be >= Open and Close
                if (high < open_price).any() or (high < close).any():
                    raise AssertionError(f"Symbol {symbol}: High < Open or Close found")

                # Low should be <= Open and Close
                if (low > open_price).any() or (low > close).any():
                    raise AssertionError(f"Symbol {symbol}: Low > Open or Close found")

            except KeyError:
                # Skip if OHLC columns don't exist
                continue


# ============================================================================
# Test Case Base Classes
# ============================================================================

class BaseTestCase:
    """Base class for test cases with common utilities."""

    def setup_method(self):
        """Setup method called before each test."""
        self.file_manager = TestFileManager()
        self.performance_tracker = PerformanceTracker()
        self.mock_data = MockDataGenerator()

    def teardown_method(self):
        """Teardown method called after each test."""
        self.file_manager.cleanup()

    def assert_dataframe_valid(self, df: pd.DataFrame, symbols: List[str]):
        """Common DataFrame validation."""
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        DataFrameAssertions.assert_has_multiindex_columns(df, ['symbol', 'price_type'])
        DataFrameAssertions.assert_symbols_present(df, symbols)


class AsyncTestCase(BaseTestCase):
    """Base class for async test cases."""

    async def run_with_timeout(self, coro, timeout: float = 5.0):
        """Run coroutine with timeout."""
        import asyncio
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise AssertionError(f"Operation timed out after {timeout}s")


# ============================================================================
# Mock Object Factories
# ============================================================================

class MockFactory:
    """Factory for creating common mock objects."""

    @staticmethod
    def create_mock_yfinance():
        """Create a mock yfinance module."""
        mock_yf = Mock()
        mock_yf.download = Mock()
        return mock_yf

    @staticmethod
    def create_mock_market_data_loader():
        """Create a mock MarketDataLoader."""
        mock_loader = Mock()
        mock_loader.load_portfolio_data = Mock()
        mock_loader.calculate_returns = Mock()
        mock_loader.get_market_data_summary = Mock()
        return mock_loader

    @staticmethod
    def create_mock_performance_timer():
        """Create a mock PerformanceTimer."""
        mock_timer = Mock()
        mock_timer.start = Mock()
        mock_timer.end = Mock()
        mock_timer.get_duration = Mock(return_value=0.1)
        return mock_timer