"""
Data validation utilities for QAOA Portfolio Optimizer (QOPO)

This module provides data validation classes and functions
for financial data integrity checking.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

import logging
from typing import List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .exceptions import DataValidationError

logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation utilities for financial data."""

    @staticmethod
    def val_symbols(symbols: List[str]) -> List[str]:
        """Validate and clean ticker symbols."""
        if not symbols:
            raise DataValidationError("Symbol list cannot be empty.")

        clean_symbols = []
        for symbol in symbols:
            if not isinstance(symbol, str):
                raise DataValidationError(
                    f"Symbol must be a string, got {type(symbol)}"
                )

            clean_symbol = symbol.strip().upper()
            if not clean_symbol:
                continue

            if (
                not clean_symbol.replace("-", "")
                .replace(".", "")
                .replace("_", "")
                .isalnum()
            ):
                logger.warning(f"Potentially invalid symbol: {clean_symbol}")

            clean_symbols.append(clean_symbol)

        if not clean_symbols:
            raise DataValidationError("No valid symbols found after cleaning.")

        return clean_symbols

    @staticmethod
    def val_date_range(start_date: datetime, end_date: datetime) -> tuple:
        """Validate date range for data loading"""
        if start_date >= end_date:
            raise DataValidationError("Start date must be before end date.")

        if end_date > datetime.now() + timedelta(days=1):
            raise DataValidationError("End date is in the future!")

        date_diff = end_date - start_date
        if date_diff.days < 1:
            raise DataValidationError("Date range must be at least 1 day.")

        if date_diff.days > 365 * 10:
            logger.warning("Date range exceeds 10 years.")

        return start_date, end_date

    @staticmethod
    def val_price_data(data: pd.DataFrame, symbol: str = "Unknown") -> bool:
        """Validate price data DataFrame."""
        if data.empty:
            raise DataValidationError(f"Empty data for {symbol}")

        # Check for required columns
        required_cols = ["close"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns for {symbol}: {missing_cols}"
            )

        # Check for negative prices
        price_cols = ["open", "high", "low", "close", "adj_close"]
        for col in price_cols:
            if col in data.columns:
                if (data[col] <= 0).any():
                    raise DataValidationError(
                        f"Non-positive prices found in {col} for {symbol}"
                    )

        # Check for reasonable price relationships (if OHLC available)
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            if not (
                (data["high"] >= data["open"])
                & (data["high"] >= data["low"])
                & (data["high"] >= data["close"])
            ).all():
                logger.warning(f"Inconsistent OHLC relationships found for {symbol}")

            if not (
                (data["low"] <= data["open"])
                & (data["low"] <= data["high"])
                & (data["low"] <= data["close"])
            ).all():
                logger.warning(f"Inconsistent OHLC relationships found for {symbol}")

        # Check for excessive missing data
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100
        if missing_pct > 20:
            raise DataValidationError(
                f"Excessive missing data for {symbol}: {missing_pct:.1f}%"
            )

        return True

    @staticmethod
    def val_returns_data(returns: pd.DataFrame, symbol: str = "Unknown") -> bool:
        """Validate returns data DataFrame."""
        if returns.empty:
            raise DataValidationError(f"Empty returns data for {symbol}")

        if (
            np.isinf(returns).any().any()
            if isinstance(returns, pd.DataFrame)
            else np.isinf(returns).any()
        ):
            raise DataValidationError(
                f"Infinite values found in returns data for {symbol}"
            )

        extreme_threshold = 0.9  # 90% daily return is extreme
        if isinstance(returns, pd.DataFrame):
            extreme_ret = (returns.abs() > extreme_threshold).any().any()
        else:
            extreme_ret = (returns.abs() > extreme_threshold).any()

        if extreme_ret:
            logger.warning(
                f"Extreme returns (>{extreme_threshold*100}%) detected for {symbol}"
            )

        return True
