"""
QAOA Portfolio Optimizer

A high-performance implementation of the Quantum Approximate Optimization Algorithm (QAOA)
for portfolio optimization problems, demonstrating quantum-inspired solutions for 
real-world financial applications.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

__version__ = "0.1.0"
__author__ = "Daniel Sobral Blanco"
__email__ = "dasobral93@gmail.com"
__license__ = "CC BY-NC-ND 4.0"

# Import exceptions
from .exceptions import (
    QAOAPortfolioError,
    MarketDataError,
    DataValidationError,
    OptimizationError,
    QuantumBackendError,
    RateLimitError,
    ConfigurationError
)

# Import configuration
from .config import ConfigManager, config

# Import utilities
from .utils import (
    PerformanceTimer,
    performance_monitor,
    DataValidator,
    ensure_directory,
    safe_divide,
    format_percentage,
    validate_weights,
    normalize_weights,
    initialize_qaoa_portfolio
)

# Import metrics
from .metrics import FinancialMetrics

# Import parameters
from .params import PortfolioParams, MarketDataParams, QAOAParams

# Import data loader
from .data_loader import (
    MarketDataLoader,
    get_free_tier_recommendations,
    setup_free_tier_environment
)

# Import portfolio utilities
from .portfolios import (
    load_sp500_symbols,
    create_sample_portfolio,
    quick_portfolio_load
)

# Package-level exports
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Exceptions
    "QAOAPortfolioError",
    "MarketDataError", 
    "DataValidationError",
    "OptimizationError",
    "QuantumBackendError",
    "RateLimitError",
    "ConfigurationError",
    
    # Configuration
    "ConfigManager",
    "config",
    
    # Utilities
    "PerformanceTimer",
    "performance_monitor",
    "DataValidator",
    "ensure_directory",
    "safe_divide",
    "format_percentage",
    "validate_weights",
    "normalize_weights",
    "initialize_qaoa_portfolio",
    
    # Metrics
    "FinancialMetrics",
    
    # Parameters
    "PortfolioParams",
    "MarketDataParams", 
    "QAOAParams",
    
    # Data loading
    "MarketDataLoader",
    "load_sp500_symbols",
    "create_sample_portfolio", 
    "quick_portfolio_load",
    "get_free_tier_recommendations",
    "setup_free_tier_environment"
]

# Initialize the package
initialize_qaoa_portfolio()