"""
Portfolio utilities for QAOA Portfolio Optimizer (QOPO)

This module provides utilities for creating and managing different types of portfolios
including stocks, crypto, and other asset classes.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import config
from .exceptions import MarketDataError

logger = logging.getLogger(__name__)

# ============================================================================
# Stock Portfolio Utilities
# ============================================================================

def load_sp500_symbols() -> List[str]:
    """Load S&P 500 symbol list."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        symbols = sp500_table['Symbol'].tolist()
        
        # Clean symbols (remove dots for Yahoo Finance compatibility)
        symbols = [symbol.replace('.', '-') for symbol in symbols]
        
        logger.info(f"Loaded {len(symbols)} S&P 500 symbols")
        return symbols
        
    except Exception as e:
        logger.error(f"Error loading S&P 500 symbols: {e}")
        # Fallback to a small set of common symbols
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']


def create_sample_portfolio(size: Optional[int] = None) -> List[str]:
    """Create a sample stock portfolio of given size."""
    # Get default size from config if not specified
    if size is None:
        size = config.get('portfolio.default_size', 5)
    
    large_cap_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V',
        'PG', 'HD', 'BAC', 'MA', 'DIS', 'ADBE', 'CRM', 'NFLX', 'KO', 'PEP',
        'INTC', 'VZ', 'CMCSA', 'PFE', 'T', 'WMT', 'MRK', 'ABT', 'CVX', 'XOM'
    ]
    
    if size > len(large_cap_stocks):
        logger.warning(f"Requested size {size} larger than available symbols {len(large_cap_stocks)}")
        size = len(large_cap_stocks)
    
    return large_cap_stocks[:size]


# ============================================================================
# Crypto Portfolio Utilities
# ============================================================================

def load_crypto_symbols() -> List[str]:
    """Load top cryptocurrency symbols (Yahoo Finance format)."""
    try:
        # Top cryptocurrencies by market cap with Yahoo Finance suffixes
        crypto_symbols = [
            'BTC-USD',   # Bitcoin
            'ETH-USD',   # Ethereum
            'BNB-USD',   # Binance Coin
            'XRP-USD',   # Ripple
            'ADA-USD',   # Cardano
            'SOL-USD',   # Solana
            'DOGE-USD',  # Dogecoin
            'DOT-USD',   # Polkadot
            'MATIC-USD', # Polygon
            'AVAX-USD',  # Avalanche
            'SHIB-USD',  # Shiba Inu
            'LTC-USD',   # Litecoin
            'UNI-USD',   # Uniswap
            'ATOM-USD',  # Cosmos
            'LINK-USD',  # Chainlink
            'XLM-USD',   # Stellar
            'ALGO-USD',  # Algorand
            'VET-USD',   # VeChain
            'ICP-USD',   # Internet Computer
            'FIL-USD'    # Filecoin
        ]
        
        logger.info(f"Loaded {len(crypto_symbols)} cryptocurrency symbols")
        return crypto_symbols
        
    except Exception as e:
        logger.error(f"Error loading crypto symbols: {e}")
        # Fallback to major cryptocurrencies
        return ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']


def create_sample_crypto_portfolio(size: Optional[int] = None) -> List[str]:
    """Create a sample cryptocurrency portfolio of given size."""
    # Get default size from config if not specified
    if size is None:
        size = config.get('portfolio.default_size', 5)
    
    crypto_assets = [
        'BTC-USD',   # Bitcoin - Digital gold
        'ETH-USD',   # Ethereum - Smart contracts platform
        'BNB-USD',   # Binance Coin - Exchange token
        'SOL-USD',   # Solana - High-performance blockchain
        'ADA-USD',   # Cardano - Research-driven blockchain
        'DOT-USD',   # Polkadot - Interoperability protocol
        'MATIC-USD', # Polygon - Ethereum scaling
        'AVAX-USD',  # Avalanche - Decentralized platform
        'LINK-USD',  # Chainlink - Oracle network
        'UNI-USD'    # Uniswap - Decentralized exchange
    ]
    
    if size > len(crypto_assets):
        logger.warning(f"Requested size {size} larger than available crypto assets {len(crypto_assets)}")
        size = len(crypto_assets)
    
    return crypto_assets[:size]


# ============================================================================
# Mixed Portfolio Utilities
# ============================================================================

def create_mixed_portfolio(
    stocks: int = 3, 
    crypto: int = 2, 
    stock_type: str = 'large_cap', 
    crypto_type: str = 'major'
) -> List[str]:
    """Create a mixed portfolio with stocks and crypto assets."""
    portfolio = []
    
    # Add stock component
    if stocks > 0:
        if stock_type == 'large_cap':
            stock_symbols = create_sample_portfolio(stocks)
        else:
            stock_symbols = create_sample_portfolio(stocks)  # Could add small_cap, etc.
        portfolio.extend(stock_symbols)
    
    # Add crypto component
    if crypto > 0:
        if crypto_type == 'major':
            crypto_symbols = create_sample_crypto_portfolio(crypto)
        else:
            crypto_symbols = create_sample_crypto_portfolio(crypto)  # Could add altcoins, etc.
        portfolio.extend(crypto_symbols)
    
    logger.info(f"Created mixed portfolio: {stocks} stocks + {crypto} crypto = {len(portfolio)} assets")
    return portfolio


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
    days_back: int = 252
) -> Tuple:
    """
    Quick utility to load portfolio data and calculate returns.
    
    Args:
        symbols: List of symbols (if None, creates sample portfolio)
        portfolio_type: 'stock', 'crypto', or 'mixed'
        days_back: Number of days of historical data
        
    Returns:
        Tuple of (price_data, returns_data)
    """
    # Import here to avoid circular import
    from .data_loader import MarketDataLoader
    
    # Create symbols if not provided
    if symbols is None:
        if portfolio_type == 'stock':
            symbols = create_sample_portfolio()
        elif portfolio_type == 'crypto':
            symbols = create_sample_crypto_portfolio()
        elif portfolio_type == 'mixed':
            symbols = create_mixed_portfolio()
        else:
            raise ValueError(f"Unsupported portfolio type: {portfolio_type}")
    
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


# ============================================================================
# Portfolio Presets
# ============================================================================

PORTFOLIO_PRESETS = {
    'conservative_stocks': {
        'symbols': ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO'],
        'description': 'Large-cap defensive stocks'
    },
    'growth_stocks': {
        'symbols': ['GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
        'description': 'High-growth technology stocks'
    },
    'major_crypto': {
        'symbols': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD'],
        'description': 'Top 5 cryptocurrencies by market cap'
    },
    'defi_crypto': {
        'symbols': ['ETH-USD', 'UNI-USD', 'LINK-USD', 'MATIC-USD', 'AVAX-USD'],
        'description': 'DeFi and smart contract platforms'
    },
    'balanced_mixed': {
        'symbols': ['AAPL', 'MSFT', 'BTC-USD', 'ETH-USD', 'SPY'],
        'description': 'Balanced mix of stocks, crypto, and index'
    }
}


def get_preset_portfolio(preset_name: str) -> List[str]:
    """Get a predefined portfolio by name."""
    if preset_name not in PORTFOLIO_PRESETS:
        available = list(PORTFOLIO_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    preset = PORTFOLIO_PRESETS[preset_name]
    logger.info(f"Loading preset '{preset_name}': {preset['description']}")
    return preset['symbols']


def list_portfolio_presets() -> Dict[str, str]:
    """List all available portfolio presets."""
    return {name: data['description'] for name, data in PORTFOLIO_PRESETS.items()}