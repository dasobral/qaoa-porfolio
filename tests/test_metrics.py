"""
Unit tests for FinancialMetrics — the correctness foundation for the
Phase 5 benchmark reporting.
"""

import numpy as np
import pandas as pd
import pytest

from qaoa_portfolio.exceptions import DataValidationError
from qaoa_portfolio.metrics import FinancialMetrics

pytestmark = pytest.mark.unit


def make_price_frame(symbols=("AAPL", "MSFT"), days=10, start=100.0, step=1.0):
    idx = pd.date_range("2026-01-01", periods=days, freq="D")
    columns = pd.MultiIndex.from_product(
        [symbols, ["close"]], names=["symbol", "price_type"]
    )
    prices = np.column_stack(
        [start + step * np.arange(days) + 10 * i for i in range(len(symbols))]
    )
    return pd.DataFrame(prices, index=idx, columns=columns)


class TestPointReturns:
    def test_simple_return(self):
        assert FinancialMetrics.simple_return(100.0, 110.0) == pytest.approx(0.10)
        assert FinancialMetrics.simple_return(100.0, 90.0) == pytest.approx(-0.10)

    def test_simple_return_guards_non_positive_start(self):
        assert FinancialMetrics.simple_return(0.0, 110.0) == 0.0
        assert FinancialMetrics.simple_return(-5.0, 110.0) == 0.0

    def test_log_return(self):
        assert FinancialMetrics.log_return(100.0, 110.0) == pytest.approx(np.log(1.1))

    def test_log_return_guards_non_positive_prices(self):
        assert FinancialMetrics.log_return(0.0, 110.0) == 0.0
        assert FinancialMetrics.log_return(100.0, 0.0) == 0.0


class TestAnnualization:
    def test_annualized_return_compounds_daily_mean(self):
        returns = pd.Series([0.001] * 100)
        assert FinancialMetrics.annualized_return(returns) == pytest.approx(
            1.001**252 - 1
        )

    def test_annualized_volatility_scales_by_sqrt_periods(self):
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
        expected = returns.std() * np.sqrt(252)
        assert FinancialMetrics.annualized_volatility(returns) == pytest.approx(
            expected
        )

    def test_empty_series_return_zero(self):
        empty = pd.Series([], dtype=float)
        assert FinancialMetrics.annualized_return(empty) == 0.0
        assert FinancialMetrics.annualized_volatility(empty) == 0.0


class TestRiskAdjustedRatios:
    def test_sharpe_ratio_matches_manual_computation(self):
        returns = pd.Series([0.002, -0.001, 0.003, 0.001, -0.002, 0.002])
        ann_ret = FinancialMetrics.annualized_return(returns)
        ann_vol = FinancialMetrics.annualized_volatility(returns)
        expected = (ann_ret - 0.02) / ann_vol
        assert FinancialMetrics.sharpe_ratio(returns) == pytest.approx(expected)

    def test_sharpe_ratio_zero_for_empty_or_constant(self):
        assert FinancialMetrics.sharpe_ratio(pd.Series([], dtype=float)) == 0.0
        assert FinancialMetrics.sharpe_ratio(pd.Series([0.01] * 5)) == 0.0

    def test_sortino_uses_downside_deviation_only(self):
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        ann_ret = FinancialMetrics.annualized_return(returns)
        downside = returns[returns < 0]
        expected = (ann_ret - 0.02) / (downside.std() * np.sqrt(252))
        assert FinancialMetrics.sortino_ratio(returns) == pytest.approx(expected)

    def test_sortino_with_no_downside_returns(self):
        gains = pd.Series([0.01, 0.02, 0.015])
        assert FinancialMetrics.sortino_ratio(gains) == np.inf
        assert FinancialMetrics.sortino_ratio(gains, risk_free_rate=1_000_000.0) == 0.0
        assert FinancialMetrics.sortino_ratio(pd.Series([], dtype=float)) == 0.0


class TestDrawdownAndTailRisk:
    def test_max_drawdown_known_path(self):
        # 100 -> 110 -> 99 -> 108.9: worst drop is 10% from the 110 peak
        returns = pd.Series([0.10, -0.10, 0.10])
        assert FinancialMetrics.max_drawdown(returns) == pytest.approx(-0.10)

    def test_max_drawdown_zero_for_monotonic_growth(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        assert FinancialMetrics.max_drawdown(returns) == pytest.approx(0.0)

    def test_max_drawdown_empty(self):
        assert FinancialMetrics.max_drawdown(pd.Series([], dtype=float)) == 0.0

    def test_value_at_risk_is_lower_percentile(self):
        returns = pd.Series(np.linspace(-0.05, 0.05, 101))
        assert FinancialMetrics.value_at_risk(returns, 0.05) == pytest.approx(
            np.percentile(returns, 5)
        )
        assert FinancialMetrics.value_at_risk(pd.Series([], dtype=float)) == 0.0

    def test_conditional_var_is_mean_of_tail(self):
        returns = pd.Series(np.linspace(-0.05, 0.05, 101))
        var = FinancialMetrics.value_at_risk(returns, 0.05)
        expected = returns[returns <= var].mean()
        assert FinancialMetrics.conditional_var(returns, 0.05) == pytest.approx(
            expected
        )
        assert FinancialMetrics.conditional_var(pd.Series([], dtype=float)) == 0.0


class TestBetaAndCorrelation:
    def test_beta_of_market_with_itself_is_one(self):
        market = pd.Series([0.01, -0.02, 0.03, 0.005, -0.01])
        assert FinancialMetrics.beta(market, market) == pytest.approx(1.0)

    def test_beta_scales_with_leverage(self):
        market = pd.Series([0.01, -0.02, 0.03, 0.005, -0.01])
        levered = 2.0 * market
        assert FinancialMetrics.beta(levered, market) == pytest.approx(2.0)

    def test_beta_guards(self):
        empty = pd.Series([], dtype=float)
        series = pd.Series([0.01, 0.02])
        assert FinancialMetrics.beta(empty, series) == 0.0
        assert FinancialMetrics.beta(series, empty) == 0.0
        constant_market = pd.Series([0.01, 0.01, 0.01])
        asset = pd.Series([0.01, 0.02, 0.03])
        assert FinancialMetrics.beta(asset, constant_market) == 0.0

    def test_correlation_of_identical_series_is_one(self):
        series = pd.Series([0.01, -0.02, 0.03, 0.005])
        assert FinancialMetrics.correlation(series, series) == pytest.approx(1.0)

    def test_correlation_of_opposite_series_is_minus_one(self):
        series = pd.Series([0.01, -0.02, 0.03, 0.005])
        assert FinancialMetrics.correlation(series, -series) == pytest.approx(-1.0)

    def test_correlation_guards(self):
        empty = pd.Series([], dtype=float)
        series = pd.Series([0.01, 0.02])
        assert FinancialMetrics.correlation(empty, series) == 0.0
        assert FinancialMetrics.correlation(series, empty) == 0.0


class TestCalculateReturns:
    def test_simple_returns_from_multiindex_prices(self):
        prices = make_price_frame()
        returns = FinancialMetrics.calculate_returns(prices, "simple")

        assert list(returns.columns) == ["AAPL", "MSFT"]
        assert len(returns) == len(prices) - 1
        expected_first = (101.0 - 100.0) / 100.0
        assert returns["AAPL"].iloc[0] == pytest.approx(expected_first)

    def test_log_returns_from_multiindex_prices(self):
        prices = make_price_frame()
        returns = FinancialMetrics.calculate_returns(prices, "log")
        assert returns["AAPL"].iloc[0] == pytest.approx(np.log(101.0 / 100.0))

    def test_rejects_unknown_return_type(self):
        with pytest.raises(ValueError, match="Unsupported return type"):
            FinancialMetrics.calculate_returns(make_price_frame(), "weird")

    def test_rejects_missing_price_column(self):
        with pytest.raises(KeyError, match="not found for any symbols"):
            FinancialMetrics.calculate_returns(
                make_price_frame(), price_column="adj_close"
            )

    def test_infinite_returns_are_rejected_by_validation(self):
        prices = make_price_frame(symbols=("AAPL",), days=3)
        prices.iloc[1] = 0.0  # 0 -> positive price produces an infinite return
        with pytest.raises(DataValidationError, match="Infinite values"):
            FinancialMetrics.calculate_returns(prices, "simple")
