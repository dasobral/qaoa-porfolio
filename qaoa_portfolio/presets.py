"""
Portfolio presets and symbol lists for QAOA Portfolio Optimizer (QOPO)

This module provides predefined portfolio configurations, symbol lists,
and sample portfolios for different asset classes.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from .config import config

logger = logging.getLogger(__name__)

# ============================================================================
# Stock Symbol Lists
# ============================================================================

LARGE_CAP_STOCKS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "JPM",
    "JNJ",
    "V",
    "PG",
    "HD",
    "BAC",
    "MA",
    "DIS",
    "ADBE",
    "CRM",
    "NFLX",
    "KO",
    "PEP",
    "INTC",
    "VZ",
    "CMCSA",
    "PFE",
    "T",
    "WMT",
    "MRK",
    "ABT",
    "CVX",
    "XOM",
]

CRYPTO_SYMBOLS = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "XRP-USD",
    "ADA-USD",
    "SOL-USD",
    "DOGE-USD",
    "DOT-USD",
    "MATIC-USD",
    "AVAX-USD",
    "SHIB-USD",
    "LTC-USD",
    "UNI-USD",
    "ATOM-USD",
    "LINK-USD",
    "XLM-USD",
    "ALGO-USD",
    "VET-USD",
    "ICP-USD",
    "FIL-USD",
]

# ============================================================================
# Portfolio Presets
# ============================================================================

PORTFOLIO_PRESETS = {
    "conservative_stocks": {
        "symbols": ["AAPL", "MSFT", "JNJ", "PG", "KO"],
        "description": "Large-cap defensive stocks",
    },
    "growth_stocks": {
        "symbols": ["GOOGL", "AMZN", "TSLA", "NVDA", "META"],
        "description": "High-growth technology stocks",
    },
    "major_crypto": {
        "symbols": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"],
        "description": "Top 5 cryptocurrencies by market cap",
    },
    "defi_crypto": {
        "symbols": ["ETH-USD", "UNI-USD", "LINK-USD", "MATIC-USD", "AVAX-USD"],
        "description": "DeFi and smart contract platforms",
    },
    "balanced_mixed": {
        "symbols": ["AAPL", "MSFT", "BTC-USD", "ETH-USD", "SPY"],
        "description": "Balanced mix of stocks, crypto, and index",
    },
}


# ============================================================================
# Preset Access Functions
# ============================================================================


def get_preset_portfolio(preset_name: str) -> List[str]:
    """Get a predefined portfolio by name."""
    if preset_name not in PORTFOLIO_PRESETS:
        available = list(PORTFOLIO_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    preset = PORTFOLIO_PRESETS[preset_name]
    logger.info(f"Loading preset '{preset_name}': {preset['description']}")
    return preset["symbols"]


def list_portfolio_presets() -> Dict[str, str]:
    """List all available portfolio presets."""
    return {name: data["description"] for name, data in PORTFOLIO_PRESETS.items()}


# ============================================================================
# Sample Portfolio Creators
# ============================================================================


def load_sp500_symbols() -> List[str]:
    """Load S&P 500 symbol list."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        symbols = sp500_table["Symbol"].tolist()

        # Clean symbols (remove dots for Yahoo Finance compatibility)
        symbols = [symbol.replace(".", "-") for symbol in symbols]

        logger.info(f"Loaded {len(symbols)} S&P 500 symbols")
        return symbols

    except Exception as e:
        logger.error(f"Error loading S&P 500 symbols: {e}")
        # Fallback to a small set of common symbols
        return [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "JPM",
            "JNJ",
            "V",
        ]


def create_sample_portfolio(size: Optional[int] = None) -> List[str]:
    """Create a sample stock portfolio of given size."""
    if size is None:
        size = config.get("portfolio.default_size", 5)

    if size > len(LARGE_CAP_STOCKS):
        logger.warning(
            f"Requested size {size} larger than "
            f"available symbols {len(LARGE_CAP_STOCKS)}"
        )
        size = len(LARGE_CAP_STOCKS)

    return LARGE_CAP_STOCKS[:size]


def load_crypto_symbols() -> List[str]:
    """Load top cryptocurrency symbols (Yahoo Finance format)."""
    logger.info(f"Loaded {len(CRYPTO_SYMBOLS)} cryptocurrency symbols")
    return list(CRYPTO_SYMBOLS)


def create_sample_crypto_portfolio(size: Optional[int] = None) -> List[str]:
    """Create a sample cryptocurrency portfolio of given size."""
    if size is None:
        size = config.get("portfolio.default_size", 5)

    if size > len(CRYPTO_SYMBOLS):
        logger.warning(
            f"Requested size {size} larger than "
            f"available crypto assets {len(CRYPTO_SYMBOLS)}"
        )
        size = len(CRYPTO_SYMBOLS)

    return CRYPTO_SYMBOLS[:size]


def create_mixed_portfolio(stocks: int = 3, crypto: int = 2) -> List[str]:
    """Create a mixed portfolio of large-cap stocks and major crypto assets."""
    portfolio = []

    if stocks > 0:
        portfolio.extend(create_sample_portfolio(stocks))

    if crypto > 0:
        portfolio.extend(create_sample_crypto_portfolio(crypto))

    logger.info(
        f"Created mixed portfolio: {stocks} stocks + {crypto} crypto "
        f"= {len(portfolio)} assets"
    )
    return portfolio
