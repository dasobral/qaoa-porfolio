"""
Focused tests for portfolio helpers and quick load wrapper.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from qaoa_portfolio.portfolios import (
    create_mixed_portfolio,
    create_sample_crypto_portfolio,
    create_sample_portfolio,
    get_preset_portfolio,
    list_portfolio_presets,
    quick_portfolio_load,
)

pytestmark = pytest.mark.unit


def test_create_sample_portfolio_size():
    portfolio = create_sample_portfolio(size=3)
    assert len(portfolio) == 3


def test_create_sample_crypto_portfolio_size():
    portfolio = create_sample_crypto_portfolio(size=2)
    assert len(portfolio) == 2
    assert all(symbol.endswith("-USD") for symbol in portfolio)


def test_create_mixed_portfolio_counts():
    portfolio = create_mixed_portfolio(stocks=2, crypto=2)
    assert len(portfolio) == 4


def test_list_portfolio_presets():
    presets = list_portfolio_presets()
    assert "conservative_stocks" in presets
    assert "growth_stocks" in presets


def test_get_preset_portfolio_invalid_name():
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset_portfolio("invalid_preset_name")


@patch("qaoa_portfolio.portfolios.MarketDataLoader")
def test_quick_portfolio_load_preset(mock_loader_class):
    mock_loader = Mock()
    mock_loader_class.return_value = mock_loader
    mock_loader.load_portfolio_data = AsyncMock(
        return_value=pd.DataFrame({("AAPL", "close"): [100, 101]})
    )
    mock_loader.calculate_returns = Mock(return_value=pd.DataFrame({"AAPL": [0.01]}))

    with patch(
        "qaoa_portfolio.portfolios.get_preset_portfolio", return_value=["AAPL", "MSFT"]
    ) as mock_preset:
        price_data, returns_data = asyncio.run(
            quick_portfolio_load(preset="growth_stocks")
        )

    mock_preset.assert_called_once_with("growth_stocks")
    assert isinstance(price_data, pd.DataFrame)
    assert isinstance(returns_data, pd.DataFrame)


def test_quick_portfolio_load_invalid_portfolio_type():
    with pytest.raises(ValueError, match="Invalid portfolio_type"):
        asyncio.run(quick_portfolio_load(portfolio_type="invalid"))
