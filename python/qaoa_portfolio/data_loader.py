"""
Market Data Loader for the QAOA Portfolio Optimizer (QOPO)

This module provides comprehensive market data loading capabilities 
for portfolio optimization, supporting multiple data sources and 
real-time/historical data retrieval.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

import logging
import warnings
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import yfinance as yf

from .exceptions import DataValidationError, MarketDataError, RateLimitError
from .utils import DataValidator, PerformanceTimer, ensure_directory
from .params import MarketDataParams
from .config import config 

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")

# ============================================================================
# Market Data Loader
# ============================================================================

class MarketDataLoader:
    """
    Market Data Loader with configuration integration.
    
    Features:
    - Yahoo Finance only (100% free, no API keys)
    - Configuration-driven settings
    - Async data loading for performance
    - Comprehensive data validation
    - Optional caching system
    """

    def __init__(self):
        """Initialize the MarketDataLoader with configuration settings."""
        # Get settings from config
        self.cache_enabled = config.get('data_sources.cache_enabled', True)
        self.cache_duration_days = config.get('data_sources.cache_duration_days', 7)
        self.free_tier_mode = config.get('data_sources.free_tier_mode', True)
        self.data_source = config.get('data_sources.default', 'yfinance')

        # Setup cache directory
        if self.cache_enabled:
            self.cache_dir = ensure_directory("data/cache")
        else:
            self.cache_dir = None

        # Get rate limiting settings
        self.rate_limit_enabled = config.get('performance.conservative_rate_limiting', True)
        if self.rate_limit_enabled:
            self.rate_limit = config.get('free_tier.yahoo_finance.rate_limit_per_minute', 60)
            self.last_call_time = 0

        # Validate configuration
        if self.data_source != 'yfinance':
            logger.warning(f"Only Yahoo Finance is supported in free tier version. Igoring configured source '{self.data_source}'.")

        logger.info(f"MarketDataLoader initialized with data source: {self.data_source}, cache enabled: {self.cache_enabled}")
        if config.get('logging.show_free_tier_tips', True):
            self._show_free_tier_info()

    def _show_free_tier_info(self) -> None:
        """Display free tier information."""
        logger.info("🆓 Using Yahoo Finance - completely free, no API key required")
        logger.info(f"📁 Caching: {'enabled' if self.cache_enabled else 'disabled'}")
        logger.info(f"⏱️ Rate limiting: {'enabled' if self.rate_limit_enabled else 'disabled'}")

    async def load_portfolio_data(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        include_volume: bool = True,
        validate_data: bool = True
    ) -> pd.DataFrame:
        """
        Load historical price data for a portfolio of assets. 

        Args:
            symbols: List of asset symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])
            start_date: Start date for historical data
            end_date: End date for historical data
            include_volume: Whether to include trading volume data
            validate_data: Whether to perform data validation
            
        Returns:
            DataFrame with multi-level columns (symbol, price_type)
        """
        try:
            # Validate imputs
            symbols = DataValidator.val_symbols(symbols)
            start_date = self._parse_date(start_date)
            end_date = self._parse_date(end_date)
            start_date, end_date = DataValidator.val_date_range(start_date, end_date)

            logger.info(f"Loading data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")

            with PerformanceTimer(f"Portfolio data loading (Yahoo Finance)", log_result=True):
                # Load data asynchronously
                data_tasks = [
                    self._load_single_asset_async(symbol, start_date, end_date, include_volume, validate_data)
                    for symbol in symbols
                ]
                results = await asyncio.gather(*data_tasks, return_exceptions=True)

            # Process results
            portfolio_data = {}
            failed_symbols = []

            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.error(f"Error loading data for {symbol}: {result}")
                    failed_symbols.append(symbol)
                else:
                    portfolio_data[symbol] = result

            if failed_symbols:
                logger.warning(f"Failed to load data for symbols: {', '.join(failed_symbols)}")
                    
            if not portfolio_data:
                raise MarketDataError("No data could be loaded for any symbols.")
            
            # Combine into a single DataFrame
            complete_data = self._combine_portfolio_data(portfolio_data, include_volume)

            if validate_data:
                complete_data = self._validate_and_clean_data(portfolio_data, include_volume)

            logger.info(f"Successfully loaded data for {len(complete_data.columns.levels[0])} symbols.")

            return complete_data
        
        except Exception as e:
            logger.error(f"Error loading portfolio data: {e}")
            raise MarketDataError(f"Failed to load portfolio data: {e}")
        
    async def _load_single_asset_async(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        include_volume: bool,
        validate_data: bool
    ) -> pd.DataFrame:
        """Load data for a single asset asynchronously."""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as pool_exec:
            future = pool_exec.submit(
                self._load_single_asset_sync,
                symbol, start_date, end_date, include_volume, validate_data
            )
            return await loop.run_in_executor(pool_exec, future.result())
        
    def _load_single_asset_sync(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        include_volume: bool,
        validate_data: bool
    ) -> pd.DataFrame:
        """Load data for a single asset synchronously."""
        # Check cache first
        if self.cache_enabled:
            cached_data = self._load_from_cache(symbol, start_date, end_date)
            if cached_data is not None:
                logger.debug(f"Loaded {symbol} data from cache.")
                return cached_data

        # Rate limiting
        if self.rate_limit_enabled:
            self._apply_rate_limit()

        # Fetch data from Yahoo Finance
        try:
            data = self._load_from_yfinance(symbol, start_date, end_date, include_volume)

            if validate_data:
                self._validate_basic_asset_data(data, symbol)

            # Cache the data
            if self.cache_enabled and data is not None:
                self._save_to_cache(data, symbol, start_date, end_date)

            return data

        except Exception as e:
            logger.error(f"Error loading {symbol} from Yahoo Finance: {e}")
            raise MarketDataError(f"Failed to load {symbol}: {e}")
        
    def _load_from_yfinance(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        include_volume: bool
    ) -> pd.DataFrame:
        """Load historical data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)

            # Use period
            total_days = (end_date - start_date).days
            if total_days <= 7:
                period = "7d"
            elif total_days <= 30:
                period = "1mo"
            elif total_days <= 90:
                period = "3mo"
            elif total_days <= 180:
                period = "6mo"
            elif total_days <= 365:
                period = "1y"
            elif total_days <= 365 * 2:
                period = "2y"
            elif total_days <= 365 * 5:
                period = "5y"
            else:
                period = "max"

            # Download with optimized settings
            data = ticker.history(
                period=period,
                interval="1d",
                auto_adjust=True,
                prepost=False,
                threads=True
            )

            # Filter to exact date range
            if not data.empty:
                data = data.loc[start_date:end_date]

            if data.empty:
                raise MarketDataError(f"No data returned for {symbol} in the specified date range.")
            
            # Standardize columns names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]

            if not include_volume and 'volume' in data.columns:
                data = data.drop('volume', axis=1)

            logger.debug(f"Yahoo Finance: Loaded {len(data)} days for {symbol}")
            return data
        
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            raise MarketDataError(f"Yahoo Finance failed for {symbol}: {e}")
        
    def _validate_basic_asset_data(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Fast basic validation for individual asset data.
        Validates essential data integrity without expensive operations.
        """
        # Check data exists and not empty
        if data.empty:
            raise DataValidationError(f"Empty data for {symbol}")
        
        # Check ALL expected columns exist (Yahoo Finance standard)
        expected_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in expected_cols if col not in data.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns for {symbol}: {missing_cols}")
        
        # Check for non-positive prices
        price_cols = expected_cols + ['adj_close']
        for col in price_cols:
            if (data[col] <= 0).any():
                raise DataValidationError(f"Non-positive prices found in {col} for {symbol}")
        
        # Check data types are numeric
        numeric_cols = [col for col in price_cols if col in data.columns]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise DataValidationError(f"Non-numeric data in {col} for {symbol}")
        
        # Check for reasonable data size
        if len(data) < MarketDataParams.MIN_TRADING_DAYS_REQUIRED:
            logger.warning(f"Limited data for {symbol}: {len(data)} days")
        
        logger.debug(f"Basic validation passed for {symbol}")
            
    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the loaded data."""
        try:
            logger.info("Validating and cleaning data...")
            
            # Check for missing data
            missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100
            if missing_pct > MarketDataParams.MAX_MISSING_DATA_PCT:
                logger.warning(f"High percentage of missing data: {missing_pct:.2f}%")
            
            # Remove rows with all NaN values
            data = data.dropna(how='all')
            
            # Forward fill missing values (common in financial data)
            data = data.fillna(method='ffill')
            
            # Remove any remaining NaN values
            data = data.dropna()
            
            logger.info(f"Data validation completed. Final shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            raise DataValidationError(f"Data validation failed: {e}")
        
    def _combine_portfolio_data(
        self,
        portfolio_data: Dict[str, pd.DataFrame],
        include_volume: bool
    ) -> pd.DataFrame:
        """Combine individual asset DataFrames into a single multi-index DataFrame."""
        try:
            # Select common columns
            all_cols = set()
            for data in portfolio_data.values():
                all_cols.update(data.columns)

            # Multi-level DF
            complete_data = pd.DataFrame()
            for symbol, data in portfolio_data.items():
                for column in all_cols:
                    if column in data.columns:
                        complete_data[(symbol, column)] = data[column]
                    else:
                        # Fill missing columns with NaN
                        complete_data[(symbol, column)] = np.nan

            # Set multi-level columns names
            complete_data.columns.names = ['symbol', 'price_type']

            return complete_data
        
        except Exception as e:
            logger.error(f"Error combining portfolio data: {e}")
            raise MarketDataError(f"Failed to combine portfolio data: {e}")
        
    def _apply_rate_limit(self) -> None:
        """Apply rate limiting for API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        min_interval = 60 / self.rate_limit
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()

    def _load_from_cache(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available and not expired."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{symbol}_yfinance_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        
        if cache_file.exists():
            try:
                # Check if cache is expired
                file_age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
                if file_age_days > self.cache_duration_days:
                    logger.debug(f"Cache expired for {symbol}, age: {file_age_days} days")
                    cache_file.unlink()  # Delete expired cache
                    return None
                
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.debug(f"Loaded {symbol} from cache: {cache_file}")
                return data
                
            except Exception as e:
                logger.warning(f"Error loading from cache: {e}")
                try:
                    cache_file.unlink()  # Delete corrupted cache
                except:
                    pass
        
        return None
    
    def _save_to_cache(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> None:
        """Save data to cache."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{symbol}_yfinance_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        
        try:
            data.to_csv(cache_file)
            logger.debug(f"Saved {symbol} to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def _parse_date(self, date: Union[str, datetime]) -> datetime:
        """Parse date input into datetime object."""
        if isinstance(date, str):
            try:
                return datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                try:
                    return datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    raise ValueError(f"Invalid date format: {date}. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'")
        elif isinstance(date, datetime):
            return date
        else:
            raise TypeError(f"Date must be string or datetime, got {type(date)}")
        
    def calculate_returns(
        self,
        price_data: pd.DataFrame,
        return_type: str = 'simple',
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """Calculate returns from price data."""
        try:
            returns_data = pd.DataFrame()

            # Extract symbols
            symbols = price_data.columns.get_level_values(0).unique()
            for symbol in symbols:
                if (symbol, price_column) in price_data.columns:
                    prices = price_data[(symbol, price_column)].dropna()

                    if return_type == 'simple':
                        returns = prices.pct_change().dropna()
                    elif return_type == 'log':
                        returns = np.log(prices / prices.shift(1)).dropna()
                    else:
                        raise ValueError(f"Unsupported return type: {return_type}")

                    returns_data[symbol] = returns

            logger.info(f"Calculated {return_type} returns for {len(symbols)} symbols")
            return returns_data

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            raise MarketDataError(f"Failed to calculate returns: {e}")
        
    def get_market_data_summary(self, data: pd.DataFrame) -> Dict:
        """Generate a summary of the loaded market data."""
        try:
            symbols = data.columns.get_level_values(0).unique()

            summary = {
                'symbols': list(symbols),
                'date_range': {
                    'start': data.index.min().strftime('%Y-%m-%d'),
                    'end': data.index.max().strftime('%Y-%m-%d'),
                    'total_days': len(data)
                },
                'data_quality': {
                    'missing_values': data.isnull().sum().sum(),
                    'missing_percentage': (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
                },
                'symbols_count': len(symbols),
                'columns_per_symbol': len(data.columns.get_level_values(1).unique())
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating data summary: {e}")
            return {}
        
# ============================================================================
# Utility Functions
# ============================================================================
        
def get_free_tier_recommendations() -> Dict:
    """Get recommendations for optimal free-tier usage."""
    return {
        "source": "Yahoo Finance",
        "benefits": [
            "Completely free with no API key required",
            "No rate limits for reasonable usage",
            "Supports stocks, ETFs, indices, forex, crypto",
            "Historical data available up to decades back",
            "Most reliable free source"
        ],
        "best_practices": [
            "Use caching to avoid repeated API calls",
            f"Keep portfolio sizes reasonable (default: {config.get('portfolio.default_size', 5)} symbols)",
            "Use shorter time periods for initial testing",
            "Enable rate limiting for stability"
        ],
        "configuration": {
            "cache_enabled": config.get('data_sources.cache_enabled'),
            "cache_duration_days": config.get('data_sources.cache_duration_days'),
            "rate_limiting": config.get('performance.conservative_rate_limiting'),
            "default_portfolio_size": config.get('portfolio.default_size')
        }
    }


def setup_free_tier_environment() -> None:
    """Setup optimal configuration for free-tier usage."""
    if not config.get('logging.show_free_tier_tips', True):
        return
    
    recommendations = get_free_tier_recommendations()
    
    print("\n" + "="*50)
    print("🆓 QAOA PORTFOLIO OPTIMIZER - FREE TIER")
    print("="*50)
    print(f"📊 Data Source: {recommendations['source']}")
    print(f"💾 Caching: {'Enabled' if recommendations['configuration']['cache_enabled'] else 'Disabled'}")
    print(f"⚡ Rate Limiting: {'Enabled' if recommendations['configuration']['rate_limiting'] else 'Disabled'}")
    print(f"📈 Default Portfolio Size: {recommendations['configuration']['default_portfolio_size']} symbols")
    print("="*50)
    
    logger.info("Free-tier environment configured successfully!")