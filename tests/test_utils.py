"""
Unit tests for utility helpers and performance timing.
"""

import logging

import pytest

from qaoa_portfolio.portfolios import (
    analyze_portfolio_composition,
    classify_asset_type,
)
from qaoa_portfolio.timing import PerformanceTimer, performance_monitor
from qaoa_portfolio.utils import (
    ensure_directory,
    format_percentage,
    normalize_weights,
    safe_divide,
    validate_weights,
)

pytestmark = pytest.mark.unit


class TestHelpers:
    def test_ensure_directory_creates_nested_path(self, tmp_path):
        target = tmp_path / "a" / "b"
        result = ensure_directory(target)
        assert result == target
        assert target.is_dir()
        # Idempotent on existing directories
        assert ensure_directory(target) == target

    def test_safe_divide(self):
        assert safe_divide(10.0, 4.0) == pytest.approx(2.5)
        assert safe_divide(10.0, 0.0) == 0.0
        assert safe_divide(10.0, 0.0, default=-1.0) == -1.0

    def test_format_percentage(self):
        assert format_percentage(0.1234) == "12.34%"
        assert format_percentage(0.1234, decimals=0) == "12%"

    def test_validate_weights(self):
        assert validate_weights([0.5, 0.3, 0.2]) is True
        assert validate_weights([0.5, 0.6]) is False
        assert validate_weights([0.5, -0.5, 1.0]) is False
        assert validate_weights([]) is False

    def test_normalize_weights(self):
        assert normalize_weights([2.0, 2.0]) == [0.5, 0.5]
        assert normalize_weights([]) == []
        # All-zero weights fall back to equal allocation
        assert normalize_weights([0.0, 0.0]) == [0.5, 0.5]


class TestTiming:
    def test_performance_timer_records_duration(self):
        with PerformanceTimer("test-op", log_result=False) as timer:
            pass
        assert timer.get_duration() is not None
        assert timer.get_duration() >= 0.0

    def test_performance_timer_duration_none_before_use(self):
        timer = PerformanceTimer("unused", log_result=False)
        assert timer.get_duration() is None

    def test_performance_monitor_decorator_passes_through(self, caplog):
        @performance_monitor
        def add(a, b):
            return a + b

        with caplog.at_level(logging.DEBUG):
            assert add(2, 3) == 5


class TestAssetClassification:
    def test_classify_asset_type(self):
        assert classify_asset_type("BTC-USD") == "crypto"
        assert classify_asset_type("EURUSD=X") == "forex"
        assert classify_asset_type("CHF.FX") == "forex"
        assert classify_asset_type("^GSPC") == "index"
        assert classify_asset_type("AAPL") == "stock"

    def test_analyze_portfolio_composition(self):
        composition = analyze_portfolio_composition(
            ["AAPL", "MSFT", "BTC-USD", "^GSPC"]
        )
        assert composition == {
            "stock": 2,
            "crypto": 1,
            "forex": 0,
            "index": 1,
            "other": 0,
        }
