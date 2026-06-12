"""
Unit tests for the qaoa-portfolio CLI.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from qaoa_portfolio.cli import _parse_symbols, _run, build_parser, main
from qaoa_portfolio.exceptions import MarketDataError

pytestmark = pytest.mark.unit


def make_load_result():
    prices = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    returns = pd.DataFrame({"a": [0.1, 0.2], "b": [0.25, 0.2]})
    return prices, returns


class TestParseSymbols:
    def test_splits_strips_and_uppercases(self):
        assert _parse_symbols(" aapl, msft ,btc-usd") == ["AAPL", "MSFT", "BTC-USD"]

    def test_empty_input_returns_none(self):
        assert _parse_symbols(None) is None
        assert _parse_symbols("") is None
        assert _parse_symbols(" , ,") is None


class TestParser:
    def test_defaults(self):
        args = build_parser().parse_args([])
        assert args.symbols is None
        assert args.portfolio_type == "stock"
        assert args.days_back == 252
        assert args.preset is None

    def test_rejects_unknown_portfolio_type(self, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--portfolio-type", "bonds"])
        assert "invalid choice" in capsys.readouterr().err


class TestRun:
    def test_successful_load_prints_summary_and_returns_zero(self, capsys):
        args = build_parser().parse_args(["--symbols", "AAPL,MSFT"])
        with patch(
            "qaoa_portfolio.cli.quick_portfolio_load",
            new=AsyncMock(return_value=make_load_result()),
        ) as loader:
            code = asyncio.run(_run(args))

        assert code == 0
        loader.assert_awaited_once_with(
            symbols=["AAPL", "MSFT"],
            portfolio_type="stock",
            days_back=252,
            preset=None,
        )
        out = capsys.readouterr().out
        assert "QAOA Portfolio load summary" in out
        assert "- rows: 3" in out

    def test_market_data_error_reports_and_returns_one(self, capsys):
        args = build_parser().parse_args([])
        with patch(
            "qaoa_portfolio.cli.quick_portfolio_load",
            new=AsyncMock(side_effect=MarketDataError("no data")),
        ):
            code = asyncio.run(_run(args))

        assert code == 1
        assert "Error: no data" in capsys.readouterr().err


class TestMain:
    def test_main_exits_with_run_return_code(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["qaoa-portfolio"])
        with patch(
            "qaoa_portfolio.cli.quick_portfolio_load",
            new=AsyncMock(return_value=make_load_result()),
        ):
            with pytest.raises(SystemExit) as excinfo:
                main()

        assert excinfo.value.code == 0
        assert "QAOA Portfolio load summary" in capsys.readouterr().out
