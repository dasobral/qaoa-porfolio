"""
Test Package for QAOA Portfolio Optimizer

This module provides centralized test configuration, utilities, and shared resources
for the entire test suite. It serves as the entry point for test discovery and
provides common testing infrastructure.

Author: Generated for QAOA Portfolio Optimizer
License: CC BY-NC-ND 4.0
"""

import pytest
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import test utilities for easy access
from .utils import (
    MockDataGenerator,
    ErrorSimulator,
    DataFrameAssertions,
    PerformanceTracker,
    MockConfigManager,
    TestFileManager,
    BaseTestCase,
    AsyncTestCase,
    MockFactory
)

# Test configuration
__version__ = "1.0.0"
__test_author__ = "QAOA Portfolio Optimizer Test Suite"

# ============================================================================
# Test Package Configuration
# ============================================================================

def configure_test_logging(level: str = "WARNING") -> None:
    """
    Configure logging for tests to reduce noise.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True
    )

    # Suppress specific warnings that are noisy in tests
    warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


def get_test_data_directory() -> Path:
    """Get the test data directory path."""
    return Path(__file__).parent / "data"


def ensure_test_data_directory() -> Path:
    """Ensure test data directory exists and return its path."""
    data_dir = get_test_data_directory()
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_test_config() -> Dict[str, Any]:
    """
    Get standard test configuration that can be used across tests.

    Returns:
        Dictionary with test-specific configuration settings
    """
    return {
        "data_sources": {
            "cache_enabled": False,  # Disable cache for tests
            "cache_duration_days": 1,
            "free_tier_mode": True,
            "default": "yfinance"
        },
        "portfolio": {
            "default_size": 3  # Smaller size for faster tests
        },
        "performance": {
            "conservative_rate_limiting": False  # Disable for faster tests
        },
        "logging": {
            "show_free_tier_tips": False,  # Reduce noise in tests
            "level": "WARNING"
        },
        "free_tier": {
            "yahoo_finance": {
                "rate_limit_per_minute": 60
            }
        },
        "testing": {
            "mock_data_seed": 42,  # For reproducible tests
            "timeout_seconds": 30,
            "max_test_symbols": 5
        }
    }


# ============================================================================
# Test Discovery and Collection
# ============================================================================

def get_test_modules() -> List[str]:
    """
    Get list of all test modules in the package.

    Returns:
        List of test module names
    """
    test_dir = Path(__file__).parent
    test_modules = []

    for test_file in test_dir.glob("test_*.py"):
        module_name = test_file.stem
        test_modules.append(module_name)

    return sorted(test_modules)


def collect_test_categories() -> Dict[str, List[str]]:
    """
    Collect tests by category based on pytest markers.

    Returns:
        Dictionary mapping test categories to test module lists
    """
    categories = {
        "unit": [],
        "integration": [],
        "performance": [],
        "slow": [],
        "network": []
    }

    # This would be populated by actual test discovery
    # For now, provide a basic mapping
    test_modules = get_test_modules()

    for module in test_modules:
        if "integration" in module:
            categories["integration"].append(module)
        elif "performance" in module:
            categories["performance"].append(module)
        else:
            categories["unit"].append(module)

    return categories


# ============================================================================
# Test Fixtures and Utilities Registry
# ============================================================================

class TestRegistry:
    """Central registry for test utilities and fixtures."""

    def __init__(self):
        self._mock_generators = {}
        self._error_simulators = {}
        self._performance_trackers = {}

    def register_mock_generator(self, name: str, generator_class):
        """Register a custom mock data generator."""
        self._mock_generators[name] = generator_class

    def get_mock_generator(self, name: str = "default") -> MockDataGenerator:
        """Get a mock data generator by name."""
        if name == "default" or name not in self._mock_generators:
            return MockDataGenerator()
        return self._mock_generators[name]()

    def register_error_simulator(self, name: str, simulator_class):
        """Register a custom error simulator."""
        self._error_simulators[name] = simulator_class

    def get_error_simulator(self, name: str = "default") -> ErrorSimulator:
        """Get an error simulator by name."""
        if name == "default" or name not in self._error_simulators:
            return ErrorSimulator()
        return self._error_simulators[name]()


# Global test registry
test_registry = TestRegistry()


# ============================================================================
# Test Environment Setup
# ============================================================================

def setup_test_environment(verbose: bool = False) -> None:
    """
    Set up the test environment with proper configuration.

    Args:
        verbose: Whether to enable verbose logging
    """
    # Configure logging
    log_level = "INFO" if verbose else "WARNING"
    configure_test_logging(log_level)

    # Ensure test data directory exists
    ensure_test_data_directory()

    # Set up any environment variables needed for tests
    import os
    os.environ["TESTING"] = "true"
    os.environ["QAOA_PORTFOLIO_TEST_MODE"] = "true"

    if verbose:
        print("Test environment configured successfully")
        print(f"Test data directory: {get_test_data_directory()}")
        print(f"Available test modules: {', '.join(get_test_modules())}")


def teardown_test_environment() -> None:
    """Clean up test environment."""
    import os

    # Clean up environment variables
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if "QAOA_PORTFOLIO_TEST_MODE" in os.environ:
        del os.environ["QAOA_PORTFOLIO_TEST_MODE"]


# ============================================================================
# Test Execution Helpers
# ============================================================================

def run_test_category(category: str, verbose: bool = False) -> int:
    """
    Run tests for a specific category.

    Args:
        category: Test category ('unit', 'integration', 'performance', etc.)
        verbose: Whether to run with verbose output

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    categories = collect_test_categories()

    if category not in categories:
        print(f"Unknown test category: {category}")
        print(f"Available categories: {', '.join(categories.keys())}")
        return 1

    test_modules = categories[category]
    if not test_modules:
        print(f"No test modules found for category: {category}")
        return 0

    # Construct pytest arguments
    args = []
    if verbose:
        args.extend(["-v", "--tb=short"])
    else:
        args.extend(["-q", "--tb=line"])

    args.extend([f"tests/{module}.py" for module in test_modules])

    # Run pytest programmatically
    return pytest.main(args)


def run_fast_tests() -> int:
    """Run only fast tests (excludes slow and network tests)."""
    args = [
        "-v",
        "--tb=short",
        "-m", "not slow and not network",
        "tests/"
    ]
    return pytest.main(args)


def run_all_tests(verbose: bool = False) -> int:
    """Run all tests in the suite."""
    args = []
    if verbose:
        args.extend(["-v", "--tb=short"])
    else:
        args.extend(["-q"])

    args.append("tests/")
    return pytest.main(args)


# ============================================================================
# Test Data Management
# ============================================================================

def load_sample_portfolios() -> Dict[str, Any]:
    """
    Load sample portfolios from the test data file.

    Returns:
        Dictionary containing sample portfolios and symbols
    """
    import json
    data_file = get_test_data_directory() / "sample_portfolios.json"

    if not data_file.exists():
        # Fallback to hardcoded data if file doesn't exist
        return {
            "test_portfolios": {
                "small_stocks": {"symbols": ["AAPL", "MSFT", "GOOGL"]}
            },
            "test_symbols": {
                "reliable_stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            }
        }

    with open(data_file, 'r') as f:
        return json.load(f)


def get_sample_symbols(asset_type: str = "stocks", count: int = 3) -> List[str]:
    """
    Get sample symbols for testing from the sample portfolios data.

    Args:
        asset_type: Type of assets ('stocks', 'crypto', 'mixed')
        count: Number of symbols to return

    Returns:
        List of asset symbols
    """
    portfolios_data = load_sample_portfolios()

    if asset_type == "stocks":
        available_symbols = portfolios_data["test_symbols"]["reliable_stocks"]
    elif asset_type == "crypto":
        available_symbols = portfolios_data["test_symbols"]["reliable_crypto"]
    else:  # mixed
        stocks = portfolios_data["test_symbols"]["reliable_stocks"][:3]
        crypto = portfolios_data["test_symbols"]["reliable_crypto"][:2]
        available_symbols = stocks + crypto

    return available_symbols[:count]


def get_test_portfolio(portfolio_name: str) -> List[str]:
    """
    Get a specific test portfolio by name.

    Args:
        portfolio_name: Name of the portfolio ('small_stocks', 'small_crypto', etc.)

    Returns:
        List of symbols in the portfolio
    """
    portfolios_data = load_sample_portfolios()

    if portfolio_name not in portfolios_data["test_portfolios"]:
        available = list(portfolios_data["test_portfolios"].keys())
        raise ValueError(f"Unknown portfolio '{portfolio_name}'. Available: {available}")

    return portfolios_data["test_portfolios"][portfolio_name]["symbols"]


def list_test_portfolios() -> Dict[str, str]:
    """
    List all available test portfolios with their descriptions.

    Returns:
        Dictionary mapping portfolio names to descriptions
    """
    portfolios_data = load_sample_portfolios()
    result = {}

    for name, portfolio in portfolios_data["test_portfolios"].items():
        result[name] = portfolio.get("description", f"Test portfolio: {name}")

    return result


def create_test_portfolio(size: int = 3, asset_type: str = "stocks") -> List[str]:
    """
    Create a test portfolio with specified characteristics.

    Args:
        size: Number of assets in portfolio
        asset_type: Type of assets ('stocks', 'crypto', 'mixed')

    Returns:
        List of asset symbols for testing
    """
    return get_sample_symbols(asset_type, size)


# ============================================================================
# Package Exports
# ============================================================================

# Make key utilities available at package level
__all__ = [
    # Test utilities
    "MockDataGenerator",
    "ErrorSimulator",
    "DataFrameAssertions",
    "PerformanceTracker",
    "MockConfigManager",
    "TestFileManager",
    "BaseTestCase",
    "AsyncTestCase",
    "MockFactory",

    # Configuration and setup
    "get_test_config",
    "setup_test_environment",
    "teardown_test_environment",
    "configure_test_logging",

    # Test execution
    "run_test_category",
    "run_fast_tests",
    "run_all_tests",

    # Test data
    "get_sample_symbols",
    "create_test_portfolio",
    "get_test_data_directory",
    "load_sample_portfolios",
    "get_test_portfolio",
    "list_test_portfolios",

    # Registry
    "test_registry",

    # Discovery
    "get_test_modules",
    "collect_test_categories"
]

# ============================================================================
# Package Initialization
# ============================================================================

# Automatically configure test environment when package is imported
setup_test_environment(verbose=False)

# Print information only if running tests directly
if __name__ == "__main__":
    print("QAOA Portfolio Optimizer Test Suite")
    print(f"Version: {__version__}")
    print(f"Test modules: {len(get_test_modules())}")
    print(f"Test categories: {list(collect_test_categories().keys())}")

    # Run fast tests by default
    exit_code = run_fast_tests()
    exit(exit_code)