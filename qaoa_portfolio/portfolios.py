"""
Portfolio loading and analysis utilities for QAOA Portfolio Optimizer (QOPO)

This module provides portfolio loading orchestration and asset classification.
For portfolio presets and symbol lists, see `presets.py`.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .exceptions import MarketDataError
from .data_loader import MarketDataLoader

# Re-export from presets for backward compatibility
from .presets import (  # noqa: F401
    load_sp500_symbols,
    create_sample_portfolio,
    load_crypto_symbols,
    create_sample_crypto_portfolio,
    create_mixed_portfolio,
    get_preset_portfolio,
    list_portfolio_presets,
    PORTFOLIO_PRESETS,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Portfolio Analysis Utilities
# ============================================================================

def classify_asset_type(symbol: str) -> str:
    """Classify asset type based on symbol format."""
    if symbol.endswith('-USD'):
        return 'crypto'
    elif '=' in symbol or symbol.endswith('.FX'):
        return 'forex'
    elif symbol.startswith('^'):
        return 'index'
    else:
        return 'stock'


def analyze_portfolio_composition(symbols: List[str]) -> Dict[str, int]:
    """Analyze portfolio composition by asset type."""
    composition = {'stock': 0, 'crypto': 0, 'forex': 0, 'index': 0, 'other': 0}

    for symbol in symbols:
        asset_type = classify_asset_type(symbol)
        if asset_type in composition:
            composition[asset_type] += 1
        else:
            composition['other'] += 1

    return composition


# ============================================================================
# Quick Portfolio Loading Wrapper
# ============================================================================

async def quick_portfolio_load(
    symbols: Optional[List[str]] = None,
    portfolio_type: str = 'stock',
    days_back: int = 252,
    preset: Optional[str] = None
) -> Tuple:
    """
    Quick utility to load portfolio data and calculate returns.

    Args:
        symbols: List of symbols (if None, creates sample portfolio)
        portfolio_type: 'stock', 'crypto', or 'mixed'
        days_back: Number of days of historical data
        preset: Optional preset name from PORTFOLIO_PRESETS

    Returns:
        Tuple of (price_data, returns_data)
    """
    if preset is not None:
        symbols = get_preset_portfolio(preset)
    elif symbols is None:
        if portfolio_type == 'stock':
            symbols = create_sample_portfolio()
        elif portfolio_type == 'crypto':
            symbols = create_sample_crypto_portfolio()
        elif portfolio_type == 'mixed':
            symbols = create_mixed_portfolio()
        else:
            raise ValueError(f"Invalid portfolio_type: {portfolio_type}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    loader = MarketDataLoader()

    try:
        # Load price data
        price_data = await loader.load_portfolio_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )

        # Calculate returns
        returns_data = loader.calculate_returns(price_data, return_type='simple')

        # Analyze composition
        composition = analyze_portfolio_composition(symbols)
        logger.info(f"Quick load completed: {composition}, {len(price_data)} days")

        return price_data, returns_data

    except Exception as e:
        logger.error(f"Quick portfolio load failed: {e}")
        raise MarketDataError(f"Failed to load portfolio data: {e}")
