"""
Tests for portfolio utilities and functions.

This module tests portfolio creation, symbol loading, and related utilities.
"""

import pytest
import pandas as pd
from unittest.mock import patch, Mock
from typing import List

from qaoa_portfolio.portfolios import (
    load_sp500_symbols,
    create_sample_portfolio,
    create_sample_crypto_portfolio,
    create_mixed_portfolio,
    get_preset_portfolio,
    list_portfolio_presets,
    quick_portfolio_load
)
from qaoa_portfolio.exceptions import MarketDataError


# ============================================================================
# S&P 500 Symbol Loading Tests
# ============================================================================

class TestSP500Symbols:
    """Test S&P 500 symbol loading functionality."""

    @patch('qaoa_portfolio.portfolios.pd.read_html')
    def test_load_sp500_symbols_success(self, mock_read_html):
        """Test successful S&P 500 symbol loading."""
        # Mock Wikipedia table data
        mock_table = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'Security': ['Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.', 'Amazon.com Inc.', 'Tesla Inc.']
        })
        mock_read_html.return_value = [mock_table]

        symbols = load_sp500_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) == 5
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols

    @patch('qaoa_portfolio.portfolios.pd.read_html')
    def test_load_sp500_symbols_with_dots(self, mock_read_html):
        """Test S&P 500 symbol loading with dot replacement."""
        # Mock table with symbols containing dots
        mock_table = pd.DataFrame({
            'Symbol': ['BRK.A', 'BRK.B', 'BF.A', 'BF.B'],
            'Security': ['Berkshire Hathaway A', 'Berkshire Hathaway B', 'Brown-Forman A', 'Brown-Forman B']
        })
        mock_read_html.return_value = [mock_table]

        symbols = load_sp500_symbols()

        assert 'BRK-A' in symbols
        assert 'BRK-B' in symbols
        assert 'BF-A' in symbols
        assert 'BF-B' in symbols
        assert 'BRK.A' not in symbols  # Should be replaced

    @patch('qaoa_portfolio.portfolios.pd.read_html')
    def test_load_sp500_symbols_fallback(self, mock_read_html):
        """Test fallback to default symbols when Wikipedia fails."""
        mock_read_html.side_effect = Exception("Network error")

        symbols = load_sp500_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) == 10  # Fallback list size
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols


# ============================================================================
# Portfolio Creation Tests
# ============================================================================

class TestPortfolioCreation:
    """Test portfolio creation functions."""

    @patch('qaoa_portfolio.portfolios.load_sp500_symbols')
    def test_create_sample_portfolio_default_size(self, mock_load_symbols, test_config):
        """Test creating sample portfolio with default size."""
        mock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        mock_load_symbols.return_value = mock_symbols

        with patch('qaoa_portfolio.portfolios.config', test_config):
            portfolio = create_sample_portfolio()

            assert isinstance(portfolio, list)
            assert len(portfolio) == 3  # Test config default size
            assert all(symbol in mock_symbols for symbol in portfolio)

    @patch('qaoa_portfolio.portfolios.load_sp500_symbols')
    def test_create_sample_portfolio_custom_size(self, mock_load_symbols):
        """Test creating sample portfolio with custom size."""
        mock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        mock_load_symbols.return_value = mock_symbols

        portfolio = create_sample_portfolio(size=2)

        assert len(portfolio) == 2
        assert all(symbol in mock_symbols for symbol in portfolio)

    @patch('qaoa_portfolio.portfolios.load_sp500_symbols')
    def test_create_sample_portfolio_size_exceeds_available(self, mock_load_symbols):
        """Test creating portfolio when requested size exceeds available symbols."""
        mock_symbols = ['AAPL', 'MSFT']  # Only 2 symbols available
        mock_load_symbols.return_value = mock_symbols

        portfolio = create_sample_portfolio(size=5)

        assert len(portfolio) == 2  # Should return all available
        assert portfolio == mock_symbols

    def test_create_sample_crypto_portfolio(self):
        """Test creating sample crypto portfolio."""
        crypto_portfolio = create_sample_crypto_portfolio(size=3)

        assert isinstance(crypto_portfolio, list)
        assert len(crypto_portfolio) == 3
        assert all(symbol.endswith('-USD') for symbol in crypto_portfolio)
        assert 'BTC-USD' in crypto_portfolio
        assert 'ETH-USD' in crypto_portfolio

    def test_create_sample_crypto_portfolio_large_size(self):
        """Test creating crypto portfolio with size larger than available."""
        # Test with a size larger than the hardcoded crypto list
        crypto_portfolio = create_sample_crypto_portfolio(size=100)

        assert isinstance(crypto_portfolio, list)
        assert len(crypto_portfolio) <= 20  # Should not exceed available symbols
        assert all(symbol.endswith('-USD') for symbol in crypto_portfolio)

    @patch('qaoa_portfolio.portfolios.create_sample_portfolio')
    @patch('qaoa_portfolio.portfolios.create_sample_crypto_portfolio')
    def test_create_mixed_portfolio(self, mock_crypto, mock_stocks):
        """Test creating mixed portfolio with stocks and crypto."""
        mock_stocks.return_value = ['AAPL', 'MSFT', 'GOOGL']
        mock_crypto.return_value = ['BTC-USD', 'ETH-USD']

        mixed_portfolio = create_mixed_portfolio(stocks=3, crypto=2)

        assert isinstance(mixed_portfolio, list)
        assert len(mixed_portfolio) == 5
        assert 'AAPL' in mixed_portfolio
        assert 'BTC-USD' in mixed_portfolio

        mock_stocks.assert_called_once_with(size=3)
        mock_crypto.assert_called_once_with(size=2)

    def test_create_mixed_portfolio_zero_stocks(self):
        """Test creating mixed portfolio with zero stocks."""
        mixed_portfolio = create_mixed_portfolio(stocks=0, crypto=2)

        assert len(mixed_portfolio) == 2
        assert all(symbol.endswith('-USD') for symbol in mixed_portfolio)

    def test_create_mixed_portfolio_zero_crypto(self):
        """Test creating mixed portfolio with zero crypto."""
        with patch('qaoa_portfolio.portfolios.create_sample_portfolio') as mock_stocks:
            mock_stocks.return_value = ['AAPL', 'MSFT']

            mixed_portfolio = create_mixed_portfolio(stocks=2, crypto=0)

            assert len(mixed_portfolio) == 2
            assert not any(symbol.endswith('-USD') for symbol in mixed_portfolio)


# ============================================================================
# Portfolio Presets Tests
# ============================================================================

class TestPortfolioPresets:
    """Test portfolio preset functionality."""

    def test_list_portfolio_presets(self):
        """Test listing available portfolio presets."""
        presets = list_portfolio_presets()

        assert isinstance(presets, dict)
        assert 'conservative_stocks' in presets
        assert 'growth_stocks' in presets
        assert 'major_crypto' in presets
        assert 'defi_crypto' in presets
        assert 'balanced_mixed' in presets

        # Check that descriptions are provided
        for preset_name, description in presets.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_get_preset_portfolio_conservative_stocks(self):
        """Test getting conservative stocks preset."""
        portfolio = get_preset_portfolio('conservative_stocks')

        assert isinstance(portfolio, list)
        assert len(portfolio) > 0
        # Should contain large-cap defensive stocks
        expected_stocks = ['JNJ', 'PG', 'KO', 'PEP', 'WMT']
        assert any(stock in portfolio for stock in expected_stocks)

    def test_get_preset_portfolio_growth_stocks(self):
        """Test getting growth stocks preset."""
        portfolio = get_preset_portfolio('growth_stocks')

        assert isinstance(portfolio, list)
        assert len(portfolio) > 0
        # Should contain tech growth stocks
        expected_stocks = ['GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        assert any(stock in portfolio for stock in expected_stocks)

    def test_get_preset_portfolio_major_crypto(self):
        """Test getting major crypto preset."""
        portfolio = get_preset_portfolio('major_crypto')

        assert isinstance(portfolio, list)
        assert len(portfolio) > 0
        assert all(symbol.endswith('-USD') for symbol in portfolio)
        assert 'BTC-USD' in portfolio
        assert 'ETH-USD' in portfolio

    def test_get_preset_portfolio_defi_crypto(self):
        """Test getting DeFi crypto preset."""
        portfolio = get_preset_portfolio('defi_crypto')

        assert isinstance(portfolio, list)
        assert len(portfolio) > 0
        assert all(symbol.endswith('-USD') for symbol in portfolio)
        # Should contain DeFi tokens
        expected_tokens = ['ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD', 'AVAX-USD']
        assert any(token in portfolio for token in expected_tokens)

    def test_get_preset_portfolio_balanced_mixed(self):
        """Test getting balanced mixed preset."""
        portfolio = get_preset_portfolio('balanced_mixed')

        assert isinstance(portfolio, list)
        assert len(portfolio) > 0

        # Should contain both stocks and crypto
        has_stocks = any(not symbol.endswith('-USD') for symbol in portfolio)
        has_crypto = any(symbol.endswith('-USD') for symbol in portfolio)
        assert has_stocks and has_crypto

    def test_get_preset_portfolio_invalid_name(self):
        """Test handling of invalid preset name."""
        with pytest.raises(ValueError, match="Unknown portfolio preset"):
            get_preset_portfolio('invalid_preset_name')


# ============================================================================
# Quick Portfolio Load Tests
# ============================================================================

class TestQuickPortfolioLoad:
    """Test quick portfolio loading functionality."""

    @pytest.mark.asyncio
    @patch('qaoa_portfolio.portfolios.MarketDataLoader')
    async def test_quick_portfolio_load_default(self, mock_loader_class):
        """Test quick portfolio load with defaults."""
        # Mock the loader instance and its methods
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        # Mock return values
        mock_price_data = pd.DataFrame({'test': [1, 2, 3]})
        mock_returns_data = pd.DataFrame({'returns': [0.1, 0.2]})

        mock_loader.load_portfolio_data = Mock(return_value=mock_price_data)
        mock_loader.calculate_returns = Mock(return_value=mock_returns_data)

        # Mock create_sample_portfolio
        with patch('qaoa_portfolio.portfolios.create_sample_portfolio') as mock_create:
            mock_create.return_value = ['AAPL', 'MSFT', 'GOOGL']

            price_data, returns_data = await quick_portfolio_load()

            assert price_data is mock_price_data
            assert returns_data is mock_returns_data
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    @patch('qaoa_portfolio.portfolios.MarketDataLoader')
    async def test_quick_portfolio_load_crypto(self, mock_loader_class):
        """Test quick portfolio load for crypto."""
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        mock_price_data = pd.DataFrame({'test': [1, 2, 3]})
        mock_returns_data = pd.DataFrame({'returns': [0.1, 0.2]})

        mock_loader.load_portfolio_data = Mock(return_value=mock_price_data)
        mock_loader.calculate_returns = Mock(return_value=mock_returns_data)

        with patch('qaoa_portfolio.portfolios.create_sample_crypto_portfolio') as mock_create:
            mock_create.return_value = ['BTC-USD', 'ETH-USD']

            price_data, returns_data = await quick_portfolio_load(
                portfolio_type='crypto',
                days_back=180
            )

            assert price_data is mock_price_data
            assert returns_data is mock_returns_data
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    @patch('qaoa_portfolio.portfolios.MarketDataLoader')
    async def test_quick_portfolio_load_custom_symbols(self, mock_loader_class):
        """Test quick portfolio load with custom symbols."""
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        mock_price_data = pd.DataFrame({'test': [1, 2, 3]})
        mock_returns_data = pd.DataFrame({'returns': [0.1, 0.2]})

        mock_loader.load_portfolio_data = Mock(return_value=mock_price_data)
        mock_loader.calculate_returns = Mock(return_value=mock_returns_data)

        custom_symbols = ['AAPL', 'BTC-USD', 'ETH-USD']

        price_data, returns_data = await quick_portfolio_load(
            symbols=custom_symbols,
            days_back=365
        )

        assert price_data is mock_price_data
        assert returns_data is mock_returns_data

        # Verify the correct symbols were passed
        call_args = mock_loader.load_portfolio_data.call_args
        assert call_args[1]['symbols'] == custom_symbols

    @pytest.mark.asyncio
    @patch('qaoa_portfolio.portfolios.MarketDataLoader')
    async def test_quick_portfolio_load_preset(self, mock_loader_class):
        """Test quick portfolio load with preset."""
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        mock_price_data = pd.DataFrame({'test': [1, 2, 3]})
        mock_returns_data = pd.DataFrame({'returns': [0.1, 0.2]})

        mock_loader.load_portfolio_data = Mock(return_value=mock_price_data)
        mock_loader.calculate_returns = Mock(return_value=mock_returns_data)

        with patch('qaoa_portfolio.portfolios.get_preset_portfolio') as mock_preset:
            mock_preset.return_value = ['GOOGL', 'AMZN', 'TSLA']

            price_data, returns_data = await quick_portfolio_load(
                preset='growth_stocks'
            )

            assert price_data is mock_price_data
            assert returns_data is mock_returns_data
            mock_preset.assert_called_once_with('growth_stocks')

    @pytest.mark.asyncio
    @patch('qaoa_portfolio.portfolios.MarketDataLoader')
    async def test_quick_portfolio_load_data_loading_error(self, mock_loader_class):
        """Test quick portfolio load with data loading error."""
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        # Simulate data loading error
        mock_loader.load_portfolio_data = Mock(side_effect=MarketDataError("Network error"))

        with pytest.raises(MarketDataError, match="Network error"):
            await quick_portfolio_load(symbols=['AAPL'])

    def test_quick_portfolio_load_invalid_portfolio_type(self):
        """Test quick portfolio load with invalid portfolio type."""
        with pytest.raises(ValueError, match="Invalid portfolio_type"):
            # This would be async, but the validation should happen before async call
            import asyncio
            asyncio.run(quick_portfolio_load(portfolio_type='invalid'))


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling in portfolio utilities."""

    def test_create_sample_portfolio_zero_size(self):
        """Test creating portfolio with zero size."""
        with patch('qaoa_portfolio.portfolios.load_sp500_symbols') as mock_load:
            mock_load.return_value = ['AAPL', 'MSFT']

            portfolio = create_sample_portfolio(size=0)

            assert portfolio == []

    def test_create_sample_portfolio_negative_size(self):
        """Test creating portfolio with negative size."""
        with patch('qaoa_portfolio.portfolios.load_sp500_symbols') as mock_load:
            mock_load.return_value = ['AAPL', 'MSFT']

            portfolio = create_sample_portfolio(size=-1)

            assert portfolio == []

    def test_create_sample_crypto_portfolio_zero_size(self):
        """Test creating crypto portfolio with zero size."""
        portfolio = create_sample_crypto_portfolio(size=0)
        assert portfolio == []

    def test_create_mixed_portfolio_negative_values(self):
        """Test creating mixed portfolio with negative values."""
        portfolio = create_mixed_portfolio(stocks=-1, crypto=-1)
        assert portfolio == []

    @patch('qaoa_portfolio.portfolios.load_sp500_symbols')
    def test_create_sample_portfolio_empty_symbol_list(self, mock_load_symbols):
        """Test creating portfolio when no symbols are available."""
        mock_load_symbols.return_value = []

        portfolio = create_sample_portfolio(size=5)

        assert portfolio == []


# ============================================================================
# Integration Tests
# ============================================================================

class TestPortfolioIntegration:
    """Integration tests for portfolio functionality."""

    @pytest.mark.asyncio
    async def test_full_portfolio_workflow(self):
        """Test complete portfolio creation and data loading workflow."""
        # This test uses mocks to avoid network calls
        with patch('qaoa_portfolio.portfolios.MarketDataLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader

            # Mock successful data loading
            mock_data = pd.DataFrame({
                ('AAPL', 'close'): [150, 151, 152],
                ('MSFT', 'close'): [250, 251, 252]
            })
            mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)

            mock_loader.load_portfolio_data = Mock(return_value=mock_data)
            mock_loader.calculate_returns = Mock(return_value=pd.DataFrame())

            # Test the full workflow
            symbols = get_preset_portfolio('conservative_stocks')
            assert len(symbols) > 0

            price_data, returns_data = await quick_portfolio_load(symbols=symbols[:3])

            assert isinstance(price_data, pd.DataFrame)
            assert isinstance(returns_data, pd.DataFrame)