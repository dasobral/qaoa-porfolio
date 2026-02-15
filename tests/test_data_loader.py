"""
Focused tests for MarketDataLoader.
"""

import asyncio
from datetime import datetime, timedelta

import pandas as pd
import pytest
from unittest.mock import AsyncMock

from qaoa_portfolio.data_loader import MarketDataLoader
from qaoa_portfolio.exceptions import MarketDataError


def _single_symbol_frame(symbol: str = "AAPL", days: int = 5) -> pd.DataFrame:
    idx = pd.date_range(datetime.now() - timedelta(days=days), periods=days, freq="D")
    return pd.DataFrame(
        {
            "open": [100 + i for i in range(days)],
            "high": [101 + i for i in range(days)],
            "low": [99 + i for i in range(days)],
            "close": [100 + i for i in range(days)],
            "volume": [1000 + i for i in range(days)],
        },
        index=idx,
    )


def test_load_portfolio_data_success(monkeypatch, test_config):
    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)

    loader = MarketDataLoader()
    loader._load_single_asset_async = AsyncMock(return_value=_single_symbol_frame())
    result = asyncio.run(
        loader.load_portfolio_data(
            symbols=["AAPL", "MSFT"],
            start_date=(datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
        )
    )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.columns.names == ["symbol", "price_type"]


def test_calculate_simple_returns(market_loader, mock_price_data, sample_stock_symbols):
    price_data = mock_price_data(sample_stock_symbols, days=10)
    returns = market_loader.calculate_returns(price_data=price_data, return_type="simple")
    assert isinstance(returns, pd.DataFrame)
    assert not returns.empty


def test_missing_adj_close_does_not_fail_validation(market_loader):
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000, 1100, 1200],
        }
    )
    market_loader._validate_basic_asset_data(data, "AAPL")


def test_rate_limit_calls_per_minute_fallback(monkeypatch, test_config):
    test_config.config["performance"]["conservative_rate_limiting"] = True
    test_config.config["free_tier"]["yahoo_finance"].pop("rate_limit_per_minute", None)
    test_config.config["free_tier"]["yahoo_finance"]["rate_limit_calls_per_minute"] = 15

    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)
    loader = MarketDataLoader()
    assert loader.rate_limit == 15


def test_missing_yfinance_is_actionable(monkeypatch, test_config):
    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)
    monkeypatch.setattr("qaoa_portfolio.data_loader.yf", None)

    loader = MarketDataLoader()
    with pytest.raises(MarketDataError, match="yfinance is required"):
        loader._load_from_yfinance(
            symbol="AAPL",
            start_date=datetime.now() - timedelta(days=10),
            end_date=datetime.now(),
            include_volume=True,
        )
