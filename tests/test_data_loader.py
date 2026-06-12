"""
Focused tests for MarketDataLoader.
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
import pytest
from unittest.mock import AsyncMock

from qaoa_portfolio.data_loader import (
    MarketDataLoader,
    get_free_tier_recommendations,
    setup_free_tier_environment,
)
from qaoa_portfolio.exceptions import MarketDataError

pytestmark = pytest.mark.unit


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
    returns = market_loader.calculate_returns(
        price_data=price_data, return_type="simple"
    )
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


def test_yfinance_history_call_omits_threads_keyword(monkeypatch, test_config):
    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)

    history_calls = []

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            history_calls.append(kwargs)
            if "threads" in kwargs:
                raise TypeError("unexpected keyword argument 'threads'")
            return _single_symbol_frame(self.symbol)

    class FakeYFinance:
        Ticker = FakeTicker

    monkeypatch.setattr("qaoa_portfolio.data_loader.yf", FakeYFinance)

    loader = MarketDataLoader()
    data = loader._load_from_yfinance(
        symbol="AAPL",
        start_date=datetime.now() - timedelta(days=10),
        end_date=datetime.now(),
        include_volume=True,
    )

    assert not data.empty
    assert history_calls
    assert "threads" not in history_calls[0]


def test_combine_aligns_mixed_timezone_indexes_on_calendar_dates(market_loader):
    """Regression test: equity bars (New York midnight) and crypto bars (UTC
    midnight) must align on calendar date, not raw timestamp, or mixed
    portfolios collapse to all-NaN rows."""
    stock_idx = pd.date_range("2022-01-03", periods=5, freq="B", tz="America/New_York")
    crypto_idx = pd.date_range("2022-01-01", periods=9, freq="D", tz="UTC")

    def frame(idx):
        return pd.DataFrame(
            {
                "open": [100.0] * len(idx),
                "high": [101.0] * len(idx),
                "low": [99.0] * len(idx),
                "close": [100.0] * len(idx),
            },
            index=idx,
        )

    combined = market_loader._combine_portfolio_data(
        {"AAPL": frame(stock_idx), "BTC-USD": frame(crypto_idx)},
        include_volume=False,
    )

    weekday_rows = combined.dropna()
    assert len(weekday_rows) == 5
    assert weekday_rows.index.tz is None
    assert (weekday_rows.index == weekday_rows.index.normalize()).all()


def test_yfinance_uses_explicit_start_end_for_historical_windows(
    monkeypatch, test_config
):
    """Regression test: past-anchored windows must fetch by start/end, not by a
    today-anchored period bucket (which returns no data for historical ranges)."""
    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)

    history_calls = []

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            history_calls.append(kwargs)
            start = pd.Timestamp(kwargs["start"])
            end = pd.Timestamp(kwargs["end"])
            idx = pd.date_range(start, end - timedelta(days=1), freq="D")
            return pd.DataFrame(
                {
                    "Open": [100.0] * len(idx),
                    "High": [101.0] * len(idx),
                    "Low": [99.0] * len(idx),
                    "Close": [100.0] * len(idx),
                    "Volume": [1000] * len(idx),
                },
                index=idx,
            )

    class FakeYFinance:
        Ticker = FakeTicker

    monkeypatch.setattr("qaoa_portfolio.data_loader.yf", FakeYFinance)

    loader = MarketDataLoader()
    data = loader._load_from_yfinance(
        symbol="AAPL",
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 6, 30),
        include_volume=True,
    )

    assert history_calls, "history() was never called"
    call = history_calls[0]
    assert "period" not in call
    assert pd.Timestamp(call["start"]) == pd.Timestamp("2020-01-01")
    # yfinance treats `end` as exclusive, so the loader must extend it by a day
    assert pd.Timestamp(call["end"]) == pd.Timestamp("2020-07-01")

    assert not data.empty
    assert data.index.min() == pd.Timestamp("2020-01-01")
    assert data.index.max() == pd.Timestamp("2020-06-30")


def test_yfinance_timezone_aware_index_accepts_naive_date_bounds(
    monkeypatch, test_config
):
    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            idx = pd.date_range(
                "2026-01-01", periods=5, freq="D", tz="America/New_York"
            )
            return pd.DataFrame(
                {
                    "Open": [100 + i for i in range(5)],
                    "High": [101 + i for i in range(5)],
                    "Low": [99 + i for i in range(5)],
                    "Close": [100 + i for i in range(5)],
                    "Volume": [1000 + i for i in range(5)],
                },
                index=idx,
            )

    class FakeYFinance:
        Ticker = FakeTicker

    monkeypatch.setattr("qaoa_portfolio.data_loader.yf", FakeYFinance)

    loader = MarketDataLoader()
    data = loader._load_from_yfinance(
        symbol="AAPL",
        start_date=datetime(2026, 1, 2),
        end_date=datetime(2026, 1, 4),
        include_volume=True,
    )

    assert len(data) == 3
    assert data.index.tz is not None
    assert list(data["close"]) == [101, 102, 103]


def test_parse_date_accepts_strings_and_datetimes(monkeypatch, test_config):
    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)
    loader = MarketDataLoader()

    assert loader._parse_date("2026-01-02") == datetime(2026, 1, 2)
    assert loader._parse_date("2026-01-02 13:30:00") == datetime(2026, 1, 2, 13, 30)
    moment = datetime(2026, 5, 1, 9, 0)
    assert loader._parse_date(moment) is moment

    with pytest.raises(ValueError, match="Invalid date format"):
        loader._parse_date("02/01/2026")
    with pytest.raises(TypeError, match="string or datetime"):
        loader._parse_date(20260102)


def test_cache_round_trip_and_expiry(monkeypatch, test_config, tmp_path):
    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)
    loader = MarketDataLoader()
    loader.cache_enabled = True
    loader.cache_dir = tmp_path

    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 10)
    frame = _single_symbol_frame("AAPL")

    loader._save_to_cache(frame, "AAPL", start, end)
    cached = loader._load_from_cache("AAPL", start, end)
    assert cached is not None
    assert list(cached.columns) == list(frame.columns)
    assert len(cached) == len(frame)

    # A negative retention forces the freshly written file to be expired
    loader.cache_duration_days = -1
    assert loader._load_from_cache("AAPL", start, end) is None
    assert not list(tmp_path.glob("AAPL_*.csv"))


def test_cache_miss_when_disabled_or_absent(monkeypatch, test_config, tmp_path):
    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)
    loader = MarketDataLoader()

    assert loader.cache_dir is None  # test_config disables caching
    start, end = datetime(2026, 1, 1), datetime(2026, 1, 10)
    assert loader._load_from_cache("AAPL", start, end) is None

    loader.cache_dir = tmp_path  # enabled but empty
    assert loader._load_from_cache("AAPL", start, end) is None
    loader._save_to_cache(_single_symbol_frame(), "AAPL", start, end)
    loader.cache_dir = None  # disabled save is a no-op
    loader._save_to_cache(_single_symbol_frame(), "MSFT", start, end)
    assert not list(tmp_path.glob("MSFT_*.csv"))


def test_load_single_asset_sync_prefers_cache(monkeypatch, test_config, tmp_path):
    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)
    loader = MarketDataLoader()
    loader.cache_enabled = True
    loader.cache_dir = tmp_path

    start, end = datetime(2026, 1, 1), datetime(2026, 1, 10)
    loader._save_to_cache(_single_symbol_frame("AAPL"), "AAPL", start, end)

    def explode(*args, **kwargs):
        raise AssertionError("network path must not be hit on cache hit")

    monkeypatch.setattr(loader, "_load_from_yfinance", explode)
    data = loader._load_single_asset_sync("AAPL", start, end, True, False)
    assert not data.empty


def test_rate_limiter_spaces_concurrent_calls(monkeypatch, test_config):
    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)
    loader = MarketDataLoader()
    loader.rate_limit_enabled = True
    loader.rate_limit = 1200  # min interval of 0.05s keeps the test fast
    loader.last_call_time = 0.0

    timestamps = []

    def hit():
        loader._apply_rate_limit()
        timestamps.append(time.time())

    threads = [threading.Thread(target=hit) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    timestamps.sort()
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    assert all(gap >= 0.04 for gap in gaps), f"calls not spaced: {gaps}"


def test_load_portfolio_data_raises_when_all_symbols_fail(monkeypatch, test_config):
    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)
    loader = MarketDataLoader()
    loader._load_single_asset_async = AsyncMock(side_effect=MarketDataError("down"))

    with pytest.raises(MarketDataError, match="No data could be loaded"):
        asyncio.run(
            loader.load_portfolio_data(
                symbols=["AAPL", "MSFT"],
                start_date=(datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d"),
            )
        )


def test_get_market_data_summary(monkeypatch, test_config):
    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)
    loader = MarketDataLoader()

    empty_summary = loader.get_market_data_summary(pd.DataFrame())
    assert empty_summary["symbols"] == []
    assert empty_summary["data_points"] == 0

    frame = _single_symbol_frame("AAPL")
    combined = loader._combine_portfolio_data({"AAPL": frame, "MSFT": frame}, True)
    summary = loader.get_market_data_summary(combined)

    assert sorted(summary["symbols"]) == ["AAPL", "MSFT"]
    assert summary["symbols_count"] == 2
    assert summary["date_range"]["total_days"] == len(frame)
    assert summary["columns_per_symbol"] == frame.shape[1]


def test_free_tier_helpers(monkeypatch, test_config, capsys):
    monkeypatch.setattr("qaoa_portfolio.data_loader.config", test_config)

    recommendations = get_free_tier_recommendations()
    assert recommendations["source"] == "Yahoo Finance"
    assert "configuration" in recommendations

    test_config.set("logging.show_free_tier_tips", True)
    setup_free_tier_environment()
    out = capsys.readouterr().out
    assert "FREE TIER" in out

    test_config.set("logging.show_free_tier_tips", False)
    setup_free_tier_environment()
    assert capsys.readouterr().out == ""
