"""
Custom exceptions for QAOA Portfolio Optimizer (QOPO)

This module defines custom exceptions used throughout the portfolio optimization framework.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

# ============================================================================
# Custom Exceptions
# ============================================================================

class QAOAPortfolioError(Exception):
    """Base exception for QAOA Portfolio Optimizer."""
    pass


class MarketDataError(QAOAPortfolioError):
    """Exception raised for market data loading/processing errors."""
    pass


class DataValidationError(QAOAPortfolioError):
    """Exception raised for data validation errors."""
    pass


class OptimizationError(QAOAPortfolioError):
    """Exception raised for optimization algorithm errors."""
    pass


class QuantumBackendError(QAOAPortfolioError):
    """Exception raised for quantum backend errors."""
    pass


class RateLimitError(QAOAPortfolioError):
    """Exception raised when API rate limits are exceeded."""
    pass


class ConfigurationError(QAOAPortfolioError):
    """Exception raised for configuration errors."""
    pass
