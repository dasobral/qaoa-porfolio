"""
Tests for QAOA Portfolio Optimizer (QOPO)

Test infrastructure is provided by pytest fixtures in conftest.py
and helper utilities in tests/utils.py.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

# Configure test logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ============================================================================
# Test Data Helpers
# ============================================================================

_TEST_DIR = Path(__file__).parent
_DATA_DIR = _TEST_DIR / "data"
_PORTFOLIOS_CACHE = None


def get_test_data_directory() -> Path:
    """Return the path to the test data directory."""
    return _DATA_DIR


def load_sample_portfolios() -> dict:
    """Load sample portfolio definitions from JSON."""
    global _PORTFOLIOS_CACHE
    if _PORTFOLIOS_CACHE is None:
        with open(_DATA_DIR / "sample_portfolios.json") as f:
            _PORTFOLIOS_CACHE = json.load(f)
    return _PORTFOLIOS_CACHE


def list_test_portfolios() -> Dict[str, str]:
    """List available test portfolios with descriptions."""
    data = load_sample_portfolios()
    return {
        name: info.get("description", "")
        for name, info in data.get("test_portfolios", {}).items()
    }


def get_test_portfolio(name: str) -> List[str]:
    """Get symbols for a named test portfolio."""
    data = load_sample_portfolios()
    portfolios = data.get("test_portfolios", {})
    if name not in portfolios:
        raise ValueError(
            f"Unknown portfolio '{name}'. Available: {list(portfolios.keys())}"
        )
    return portfolios[name]["symbols"]


def get_sample_symbols(asset_type: str = "stocks", count: int = 5) -> List[str]:
    """Get sample symbols of a given type."""
    data = load_sample_portfolios()
    symbols = data.get("test_symbols", {})

    if asset_type == "stocks":
        pool = symbols.get("stocks", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
    elif asset_type == "crypto":
        pool = symbols.get(
            "crypto", ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]
        )
    elif asset_type == "mixed":
        stocks = symbols.get("stocks", ["AAPL", "MSFT", "GOOGL"])
        crypto = symbols.get("crypto", ["BTC-USD", "ETH-USD"])
        pool = stocks + crypto
    else:
        pool = symbols.get(asset_type, [])

    return pool[:count]


def get_test_config() -> dict:
    """Get test-specific configuration overrides."""
    return {
        "data_sources": {
            "cache_enabled": False,
            "primary": "yfinance",
        },
        "performance": {
            "conservative_rate_limiting": False,
            "max_concurrent_requests": 5,
        },
        "portfolio": {
            "default_size": 3,
        },
        "testing": {
            "mock_data_seed": 42,
            "timeout_seconds": 30,
        },
    }
