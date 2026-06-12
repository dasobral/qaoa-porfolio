"""
Unit tests for DataValidator.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qaoa_portfolio.exceptions import DataValidationError
from qaoa_portfolio.validation import DataValidator

pytestmark = pytest.mark.unit


def make_ohlc_frame(days=5, base=100.0):
    idx = pd.date_range("2026-01-01", periods=days, freq="D")
    close = base + np.arange(days)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        },
        index=idx,
    )


class TestValSymbols:
    def test_cleans_and_uppercases(self):
        assert DataValidator.val_symbols([" aapl ", "msft"]) == ["AAPL", "MSFT"]

    def test_skips_blank_entries(self):
        assert DataValidator.val_symbols(["AAPL", "   "]) == ["AAPL"]

    def test_rejects_empty_list(self):
        with pytest.raises(DataValidationError, match="cannot be empty"):
            DataValidator.val_symbols([])

    def test_rejects_non_string_symbols(self):
        with pytest.raises(DataValidationError, match="must be a string"):
            DataValidator.val_symbols(["AAPL", 42])

    def test_rejects_all_blank_list(self):
        with pytest.raises(DataValidationError, match="No valid symbols"):
            DataValidator.val_symbols(["  ", ""])

    def test_warns_on_suspicious_symbol(self, caplog):
        with caplog.at_level("WARNING"):
            symbols = DataValidator.val_symbols(["AA$PL"])
        assert symbols == ["AA$PL"]
        assert "Potentially invalid symbol" in caplog.text


class TestValDateRange:
    def test_accepts_valid_range(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 6, 1)
        assert DataValidator.val_date_range(start, end) == (start, end)

    def test_rejects_inverted_range(self):
        with pytest.raises(DataValidationError, match="before end date"):
            DataValidator.val_date_range(datetime(2026, 6, 1), datetime(2026, 1, 1))

    def test_rejects_future_end_date(self):
        with pytest.raises(DataValidationError, match="future"):
            DataValidator.val_date_range(
                datetime(2026, 1, 1), datetime.now() + timedelta(days=30)
            )

    def test_warns_on_ranges_over_ten_years(self, caplog):
        start = datetime(2010, 1, 1)
        end = datetime(2025, 1, 1)
        with caplog.at_level("WARNING"):
            DataValidator.val_date_range(start, end)
        assert "exceeds 10 years" in caplog.text


class TestValPriceData:
    def test_accepts_clean_ohlc(self):
        assert DataValidator.val_price_data(make_ohlc_frame(), "AAPL") is True

    def test_rejects_empty_frame(self):
        with pytest.raises(DataValidationError, match="Empty data"):
            DataValidator.val_price_data(pd.DataFrame(), "AAPL")

    def test_rejects_missing_close_column(self):
        data = make_ohlc_frame().drop(columns=["close"])
        with pytest.raises(DataValidationError, match="Missing required columns"):
            DataValidator.val_price_data(data, "AAPL")

    def test_rejects_non_positive_prices(self):
        data = make_ohlc_frame()
        data.loc[data.index[0], "close"] = 0.0
        with pytest.raises(DataValidationError, match="Non-positive prices"):
            DataValidator.val_price_data(data, "AAPL")

    def test_warns_on_inconsistent_ohlc(self, caplog):
        data = make_ohlc_frame()
        data.loc[data.index[0], "high"] = data.loc[data.index[0], "low"] - 5.0
        with caplog.at_level("WARNING"):
            DataValidator.val_price_data(data, "AAPL")
        assert "Inconsistent OHLC" in caplog.text

    def test_rejects_excessive_missing_data(self):
        data = make_ohlc_frame(days=10)
        data.iloc[0:4] = np.nan
        with pytest.raises(DataValidationError, match="Excessive missing data"):
            DataValidator.val_price_data(data, "AAPL")


class TestValReturnsData:
    def test_accepts_clean_returns(self):
        returns = pd.DataFrame({"AAPL": [0.01, -0.02, 0.005]})
        assert DataValidator.val_returns_data(returns, "AAPL") is True

    def test_rejects_empty_returns(self):
        with pytest.raises(DataValidationError, match="Empty returns"):
            DataValidator.val_returns_data(pd.DataFrame(), "AAPL")

    def test_rejects_infinite_returns(self):
        returns = pd.DataFrame({"AAPL": [0.01, np.inf]})
        with pytest.raises(DataValidationError, match="Infinite values"):
            DataValidator.val_returns_data(returns, "AAPL")

    def test_warns_on_extreme_returns(self, caplog):
        returns = pd.DataFrame({"AAPL": [0.01, 0.95]})
        with caplog.at_level("WARNING"):
            DataValidator.val_returns_data(returns, "AAPL")
        assert "Extreme returns" in caplog.text
