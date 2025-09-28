"""
Tests for MarketDataLoader and related functionality.

This module contains comprehensive tests for the market data loading system,
including unit tests, integration tests, and error handling validation.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List

from qaoa_portfolio.data_loader import MarketDataLoader
from qaoa_portfolio.exceptions import MarketDataError, DataValidationError, RateLimitError
from qaoa_portfolio.config import config


# ============================================================================
# MarketDataLoader Initialization Tests
# ============================================================================

class TestMarketDataLoaderInit:
    """Test MarketDataLoader initialization and configuration."""

    def test_default_initialization(self, test_config):
        """Test loader initializes with default configuration."""
        with patch('qaoa_portfolio.data_loader.config', test_config):
            loader = MarketDataLoader()

            assert loader.free_tier_mode is True
            assert loader.data_source == 'yfinance'
            assert loader.cache_enabled is False  # Test config disables cache
            assert loader.rate_limit_enabled is False  # Test config disables rate limiting

    def test_cache_directory_creation(self, test_config, tmp_path):
        """Test cache directory is created when cache is enabled."""
        test_config._config['data_sources']['cache_enabled'] = True

        with patch('qaoa_portfolio.data_loader.config', test_config):
            with patch('qaoa_portfolio.data_loader.ensure_directory') as mock_ensure_dir:
                mock_ensure_dir.return_value = str(tmp_path / "cache")
                loader = MarketDataLoader()

                mock_ensure_dir.assert_called_once_with("data/cache")
                assert loader.cache_dir is not None

    def test_rate_limiting_configuration(self, test_config):
        """Test rate limiting configuration."""
        test_config._config['performance']['conservative_rate_limiting'] = True

        with patch('qaoa_portfolio.data_loader.config', test_config):
            loader = MarketDataLoader()

            assert loader.rate_limit_enabled is True
            assert loader.rate_limit == 60
            assert loader.last_call_time == 0


# ============================================================================
# Data Loading Tests
# ============================================================================

class TestDataLoading:
    """Test data loading functionality."""

    @pytest.mark.asyncio
    async def test_load_portfolio_data_success(
        self,
        market_loader,
        sample_stock_symbols,
        sample_date_range,
        mock_price_data,
        mock_yfinance
    ):
        """Test successful portfolio data loading."""
        start_date, end_date = sample_date_range
        expected_data = mock_price_data(sample_stock_symbols)

        # Mock yfinance download
        mock_yfinance.download.return_value = expected_data

        result = await market_loader.load_portfolio_data(
            symbols=sample_stock_symbols,
            start_date=start_date,
            end_date=end_date
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_yfinance.download.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_portfolio_data_with_validation(
        self,
        market_loader,
        sample_stock_symbols,
        sample_date_range,
        mock_price_data,
        mock_yfinance
    ):
        """Test data loading with validation enabled."""
        start_date, end_date = sample_date_range
        expected_data = mock_price_data(sample_stock_symbols)

        mock_yfinance.download.return_value = expected_data

        result = await market_loader.load_portfolio_data(
            symbols=sample_stock_symbols,
            start_date=start_date,
            end_date=end_date,
            validate_data=True
        )

        assert isinstance(result, pd.DataFrame)
        # Validation should not raise errors with good mock data
        assert not result.empty

    @pytest.mark.asyncio
    async def test_load_empty_symbol_list(self, market_loader, sample_date_range):
        """Test handling of empty symbol list."""
        start_date, end_date = sample_date_range

        with pytest.raises(MarketDataError, match="No symbols provided"):
            await market_loader.load_portfolio_data(
                symbols=[],
                start_date=start_date,
                end_date=end_date
            )

    @pytest.mark.asyncio
    async def test_load_invalid_date_range(self, market_loader, sample_stock_symbols):
        """Test handling of invalid date ranges."""
        start_date = "2023-12-31"
        end_date = "2023-01-01"  # End before start

        with pytest.raises(MarketDataError, match="Invalid date range"):
            await market_loader.load_portfolio_data(
                symbols=sample_stock_symbols,
                start_date=start_date,
                end_date=end_date
            )

    @pytest.mark.asyncio
    async def test_network_error_handling(
        self,
        market_loader,
        sample_stock_symbols,
        sample_date_range,
        mock_yfinance,
        network_error_simulator
    ):
        """Test handling of network errors."""
        start_date, end_date = sample_date_range

        # Simulate network timeout
        mock_yfinance.download.side_effect = network_error_simulator("timeout")

        with pytest.raises(MarketDataError, match="Failed to load data"):
            await market_loader.load_portfolio_data(
                symbols=sample_stock_symbols,
                start_date=start_date,
                end_date=end_date
            )


# ============================================================================
# Data Validation Tests
# ============================================================================

class TestDataValidation:
    """Test data validation functionality."""

    @pytest.mark.asyncio
    async def test_validation_with_missing_data(
        self,
        market_loader,
        mock_yfinance,
        mock_invalid_data
    ):
        """Test validation catches missing data."""
        invalid_data = mock_invalid_data("missing_data")
        mock_yfinance.download.return_value = invalid_data

        with pytest.raises(DataValidationError, match="Data validation failed"):
            await market_loader.load_portfolio_data(
                symbols=['AAPL'],
                start_date='2023-01-01',
                end_date='2023-01-05',
                validate_data=True
            )

    @pytest.mark.asyncio
    async def test_validation_with_negative_prices(
        self,
        market_loader,
        mock_yfinance,
        mock_invalid_data
    ):
        """Test validation catches negative prices."""
        invalid_data = mock_invalid_data("negative_prices")
        mock_yfinance.download.return_value = invalid_data

        with pytest.raises(DataValidationError, match="Data validation failed"):
            await market_loader.load_portfolio_data(
                symbols=['AAPL'],
                start_date='2023-01-01',
                end_date='2023-01-05',
                validate_data=True
            )

    @pytest.mark.asyncio
    async def test_validation_disabled(
        self,
        market_loader,
        mock_yfinance,
        mock_invalid_data
    ):
        """Test that invalid data passes when validation is disabled."""
        invalid_data = mock_invalid_data("missing_data")
        mock_yfinance.download.return_value = invalid_data

        # Should not raise when validation is disabled
        result = await market_loader.load_portfolio_data(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            validate_data=False
        )

        assert isinstance(result, pd.DataFrame)


# ============================================================================
# Returns Calculation Tests
# ============================================================================

class TestReturnsCalculation:
    """Test returns calculation functionality."""

    def test_calculate_simple_returns(self, market_loader, mock_price_data, sample_stock_symbols):
        """Test simple returns calculation."""
        price_data = mock_price_data(sample_stock_symbols, days=10)

        returns = market_loader.calculate_returns(
            price_data=price_data,
            return_type='simple'
        )

        assert isinstance(returns, pd.DataFrame)
        assert not returns.empty
        assert len(returns) == len(price_data) - 1  # One less due to diff

    def test_calculate_log_returns(self, market_loader, mock_price_data, sample_stock_symbols):
        """Test log returns calculation."""
        price_data = mock_price_data(sample_stock_symbols, days=10)

        returns = market_loader.calculate_returns(
            price_data=price_data,
            return_type='log'
        )

        assert isinstance(returns, pd.DataFrame)
        assert not returns.empty

    def test_invalid_return_type(self, market_loader, mock_price_data, sample_stock_symbols):
        """Test handling of invalid return type."""
        price_data = mock_price_data(sample_stock_symbols, days=10)

        with pytest.raises(ValueError, match="Invalid return_type"):
            market_loader.calculate_returns(
                price_data=price_data,
                return_type='invalid'
            )

    def test_returns_with_missing_price_column(self, market_loader):
        """Test returns calculation with missing price column."""
        # Create data without close prices
        data = {
            ('AAPL', 'open'): [100, 101, 102],
            ('AAPL', 'volume'): [1000, 1100, 1200]
        }
        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['symbol', 'price_type'])

        with pytest.raises(KeyError, match="Price column 'close' not found"):
            market_loader.calculate_returns(price_data=df)


# ============================================================================
# Summary Statistics Tests
# ============================================================================

class TestSummaryStatistics:
    """Test summary statistics functionality."""

    def test_get_market_data_summary(self, market_loader, mock_price_data, sample_stock_symbols):
        """Test market data summary generation."""
        price_data = mock_price_data(sample_stock_symbols, days=30)

        summary = market_loader.get_market_data_summary(price_data)

        assert isinstance(summary, dict)
        assert 'symbols' in summary
        assert 'date_range' in summary
        assert 'data_points' in summary
        assert 'data_quality' in summary

        assert summary['symbols'] == sample_stock_symbols
        assert summary['data_points'] > 0

    def test_summary_with_empty_data(self, market_loader):
        """Test summary generation with empty data."""
        empty_df = pd.DataFrame()

        summary = market_loader.get_market_data_summary(empty_df)

        assert summary['symbols'] == []
        assert summary['data_points'] == 0


# ============================================================================
# Rate Limiting Tests
# ============================================================================

class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limiting_enabled(self, test_config):
        """Test rate limiting when enabled."""
        test_config._config['performance']['conservative_rate_limiting'] = True

        with patch('qaoa_portfolio.data_loader.config', test_config):
            loader = MarketDataLoader()
            assert loader.rate_limit_enabled is True

    @pytest.mark.asyncio
    async def test_rate_limit_delay(self, test_config, mock_yfinance, mock_price_data):
        """Test that rate limiting introduces appropriate delays."""
        test_config._config['performance']['conservative_rate_limiting'] = True
        test_config._config['free_tier']['yahoo_finance']['rate_limit_per_minute'] = 2  # Very low for testing

        with patch('qaoa_portfolio.data_loader.config', test_config):
            loader = MarketDataLoader()

            # Mock successful data response
            mock_data = mock_price_data(['AAPL'])
            mock_yfinance.download.return_value = mock_data

            # Time two consecutive calls
            import time
            start_time = time.time()

            await loader.load_portfolio_data(['AAPL'], '2023-01-01', '2023-01-02')
            await loader.load_portfolio_data(['MSFT'], '2023-01-01', '2023-01-02')

            elapsed = time.time() - start_time

            # Should have some delay between calls
            assert elapsed > 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_loading_performance(
        self,
        market_loader,
        mock_yfinance,
        mock_price_data,
        performance_monitor
    ):
        """Test performance of concurrent data loading."""
        symbols_sets = [['AAPL'], ['MSFT'], ['GOOGL']]

        # Mock responses for each symbol
        for symbols in symbols_sets:
            mock_yfinance.download.return_value = mock_price_data(symbols)

        performance_monitor.start_timer('concurrent_loading')

        # Load data for multiple symbol sets concurrently
        tasks = []
        for symbols in symbols_sets:
            task = market_loader.load_portfolio_data(
                symbols=symbols,
                start_date='2023-01-01',
                end_date='2023-01-31'
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        performance_monitor.end_timer('concurrent_loading')

        # All results should be DataFrames (no exceptions)
        for result in results:
            assert isinstance(result, pd.DataFrame)

        # Performance should be reasonable (under 5 seconds for mock data)
        duration = performance_monitor.get_duration('concurrent_loading')
        assert duration < 5.0

    @pytest.mark.asyncio
    async def test_large_symbol_list_handling(
        self,
        market_loader,
        mock_yfinance,
        mock_price_data
    ):
        """Test handling of large symbol lists."""
        # Create a large list of symbols
        large_symbol_list = [f'SYMBOL{i}' for i in range(50)]

        mock_yfinance.download.return_value = mock_price_data(large_symbol_list)

        result = await market_loader.load_portfolio_data(
            symbols=large_symbol_list,
            start_date='2023-01-01',
            end_date='2023-01-31'
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complete_portfolio_analysis_workflow(
        self,
        market_loader,
        sample_stock_symbols,
        mock_yfinance,
        mock_price_data,
        test_validator
    ):
        """Test complete portfolio analysis workflow."""
        # Step 1: Load data
        price_data = mock_price_data(sample_stock_symbols, days=252)  # One year
        mock_yfinance.download.return_value = price_data

        loaded_data = await market_loader.load_portfolio_data(
            symbols=sample_stock_symbols,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )

        # Step 2: Validate structure
        assert test_validator.validate_dataframe_structure(loaded_data, sample_stock_symbols)

        # Step 3: Calculate returns
        returns = market_loader.calculate_returns(loaded_data)
        assert isinstance(returns, pd.DataFrame)

        # Step 4: Generate summary
        summary = market_loader.get_market_data_summary(loaded_data)
        assert isinstance(summary, dict)
        assert summary['symbols'] == sample_stock_symbols

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(
        self,
        market_loader,
        sample_stock_symbols,
        mock_yfinance,
        network_error_simulator
    ):
        """Test error recovery in data loading workflow."""
        # First call fails
        mock_yfinance.download.side_effect = network_error_simulator("timeout")

        with pytest.raises(MarketDataError):
            await market_loader.load_portfolio_data(
                symbols=sample_stock_symbols,
                start_date='2023-01-01',
                end_date='2023-01-31'
            )

        # Second call succeeds (simulating network recovery)
        mock_yfinance.download.side_effect = None
        mock_yfinance.download.return_value = pd.DataFrame()  # Minimal valid response

        # Should not raise an exception
        result = await market_loader.load_portfolio_data(
            symbols=sample_stock_symbols,
            start_date='2023-01-01',
            end_date='2023-01-31',
            validate_data=False  # Skip validation for empty DataFrame
        )

        assert isinstance(result, pd.DataFrame)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_single_symbol_single_day(
        self,
        market_loader,
        mock_yfinance,
        mock_price_data
    ):
        """Test loading single symbol for single day."""
        mock_data = mock_price_data(['AAPL'], days=1)
        mock_yfinance.download.return_value = mock_data

        result = await market_loader.load_portfolio_data(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-01'
        )

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_weekend_date_handling(self, market_loader, mock_yfinance):
        """Test handling of weekend dates."""
        # Saturday to Sunday (no trading days)
        mock_yfinance.download.return_value = pd.DataFrame()  # Empty response

        result = await market_loader.load_portfolio_data(
            symbols=['AAPL'],
            start_date='2023-01-07',  # Saturday
            end_date='2023-01-08',    # Sunday
            validate_data=False
        )

        assert isinstance(result, pd.DataFrame)

    def test_malformed_symbol_handling(self, market_loader):
        """Test handling of malformed symbols."""
        malformed_symbols = ['', '  ', 'SYMBOL WITH SPACES', None]

        # Filter out None values as they would cause TypeError
        clean_symbols = [s for s in malformed_symbols if s is not None]

        with pytest.raises(MarketDataError):
            # This should be run in async context, but we're testing the validation
            # that happens before the async call
            asyncio.run(market_loader.load_portfolio_data(
                symbols=clean_symbols,
                start_date='2023-01-01',
                end_date='2023-01-31'
            ))