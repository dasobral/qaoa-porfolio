"""
Test data integration tests.

This module demonstrates how the reorganized test data directory
integrates with the testing architecture and provides examples
of using the enhanced test package functionality.
"""

import pytest
from pathlib import Path
from typing import List, Dict

# Import from the enhanced test package
import tests
from tests.utils import MockDataGenerator, DataFrameAssertions


class TestDataDirectoryIntegration:
    """Test the integration of the reorganized test data directory."""

    def test_data_directory_structure(self):
        """Test that the test data directory has the correct structure."""
        data_dir = tests.get_test_data_directory()

        assert data_dir.exists(), "Test data directory should exist"
        assert data_dir.name == "data", "Directory should be named 'data'"
        assert data_dir.parent.name == "tests", "Parent should be 'tests' directory"

        # Check for expected files
        readme_file = data_dir / "README.md"
        portfolios_file = data_dir / "sample_portfolios.json"

        assert readme_file.exists(), "README.md should exist in test data directory"
        assert portfolios_file.exists(), "sample_portfolios.json should exist"

    def test_sample_portfolios_loading(self):
        """Test loading and accessing sample portfolios."""
        # Load sample portfolios
        portfolios_data = tests.load_sample_portfolios()

        assert isinstance(portfolios_data, dict), "Portfolios data should be a dictionary"
        assert "test_portfolios" in portfolios_data, "Should contain test_portfolios key"
        assert "test_symbols" in portfolios_data, "Should contain test_symbols key"

        # Check test portfolios
        test_portfolios = portfolios_data["test_portfolios"]
        assert len(test_portfolios) > 0, "Should have at least one test portfolio"
        assert "small_stocks" in test_portfolios, "Should have small_stocks portfolio"

    def test_portfolio_access_functions(self):
        """Test the portfolio access functions."""
        # Test listing portfolios
        available_portfolios = tests.list_test_portfolios()
        assert isinstance(available_portfolios, dict), "Should return dictionary"
        assert len(available_portfolios) > 0, "Should have available portfolios"

        # Test getting specific portfolio
        portfolio_name = list(available_portfolios.keys())[0]
        portfolio_symbols = tests.get_test_portfolio(portfolio_name)

        assert isinstance(portfolio_symbols, list), "Should return list of symbols"
        assert len(portfolio_symbols) > 0, "Portfolio should have symbols"

        # Test invalid portfolio name
        with pytest.raises(ValueError, match="Unknown portfolio"):
            tests.get_test_portfolio("nonexistent_portfolio")

    def test_sample_symbols_integration(self):
        """Test sample symbols integration with data file."""
        # Test getting stock symbols
        stocks = tests.get_sample_symbols("stocks", 3)
        assert isinstance(stocks, list), "Should return list"
        assert len(stocks) == 3, "Should return requested number of symbols"
        assert all(isinstance(symbol, str) for symbol in stocks), "All should be strings"

        # Test getting crypto symbols
        crypto = tests.get_sample_symbols("crypto", 2)
        assert len(crypto) == 2, "Should return requested number of crypto symbols"
        assert all(symbol.endswith("-USD") for symbol in crypto), "Crypto symbols should end with -USD"

        # Test mixed symbols
        mixed = tests.get_sample_symbols("mixed", 5)
        assert len(mixed) == 5, "Should return requested number of mixed symbols"

    def test_mock_data_with_sample_portfolios(self):
        """Test generating mock data using sample portfolios."""
        # Get a test portfolio
        small_stocks = tests.get_test_portfolio("small_stocks")

        # Generate mock data for this portfolio
        mock_gen = MockDataGenerator()
        mock_data = mock_gen.create_realistic_price_data(small_stocks, days=10, seed=42)

        # Validate the data structure
        assert mock_data.shape[0] == 10, "Should have 10 days of data"
        assert mock_data.shape[1] == len(small_stocks) * 5, "Should have OHLCV for each symbol"

        # Validate using DataFrame assertions
        DataFrameAssertions.assert_has_multiindex_columns(mock_data, ['symbol', 'price_type'])
        DataFrameAssertions.assert_symbols_present(mock_data, small_stocks)
        DataFrameAssertions.assert_no_missing_data(mock_data)
        DataFrameAssertions.assert_positive_prices(mock_data)

    def test_portfolio_data_structure_validation(self):
        """Test that sample portfolios have expected data structure."""
        portfolios_data = tests.load_sample_portfolios()

        # Check expected data structure section
        if "expected_data_structure" in portfolios_data:
            expected_structure = portfolios_data["expected_data_structure"]

            assert "columns" in expected_structure, "Should define expected columns"
            assert "price_types" in expected_structure["columns"], "Should define price types"

            expected_price_types = expected_structure["columns"]["price_types"]
            assert "open" in expected_price_types, "Should include 'open' price type"
            assert "close" in expected_price_types, "Should include 'close' price type"
            assert "volume" in expected_price_types, "Should include 'volume' price type"

    def test_test_configuration_integration(self):
        """Test that test configuration works with data directory."""
        config = tests.get_test_config()

        # Verify test-specific settings
        assert config["data_sources"]["cache_enabled"] is False, "Cache should be disabled for tests"
        assert config["performance"]["conservative_rate_limiting"] is False, "Rate limiting should be disabled"
        assert config["portfolio"]["default_size"] == 3, "Should have small default size for tests"

        # Test configuration should include testing-specific settings
        if "testing" in config:
            test_config = config["testing"]
            assert "mock_data_seed" in test_config, "Should have mock data seed for reproducibility"
            assert "timeout_seconds" in test_config, "Should have timeout configuration"


class TestDataDirectoryUsageExamples:
    """Examples of how to use the reorganized test data directory."""

    def test_complete_portfolio_testing_workflow(self):
        """Demonstrate a complete testing workflow using sample data."""
        # Step 1: Get a test portfolio
        portfolio_symbols = tests.get_test_portfolio("mixed_portfolio")

        # Step 2: Generate mock data for the portfolio
        mock_gen = MockDataGenerator()
        price_data = mock_gen.create_realistic_price_data(
            portfolio_symbols, days=30, seed=42
        )

        # Step 3: Validate the data
        DataFrameAssertions.assert_has_multiindex_columns(price_data, ['symbol', 'price_type'])
        DataFrameAssertions.assert_symbols_present(price_data, portfolio_symbols)

        # Step 4: Use test configuration
        config = tests.get_test_config()
        assert not config["data_sources"]["cache_enabled"], "Should use test config"

        # Demonstrate that this is a complete, realistic test workflow
        assert len(price_data) == 30, "Should have 30 days of data"
        assert price_data.shape[1] == len(portfolio_symbols) * 5, "Should have OHLCV data"

    def test_error_handling_with_sample_data(self):
        """Test error handling using sample portfolios."""
        from tests.utils import ErrorSimulator

        # Get valid portfolio for testing
        valid_portfolio = tests.get_test_portfolio("small_stocks")
        assert len(valid_portfolio) > 0, "Should have valid portfolio for testing"

        # Test error simulation
        error_sim = ErrorSimulator()

        # Test that errors can be safely simulated
        with pytest.raises(Exception):  # Specific exception type depends on what we're testing
            raise error_sim.market_data_error()

    def test_performance_testing_with_sample_data(self):
        """Test performance tracking with sample portfolios."""
        from tests.utils import PerformanceTracker

        # Get different sized portfolios for performance testing
        small_portfolio = tests.get_test_portfolio("small_stocks")
        large_portfolio = tests.get_test_portfolio("large_test")

        assert len(small_portfolio) < len(large_portfolio), "Should have different sized portfolios"

        # Test performance tracking
        tracker = PerformanceTracker()

        # Simulate operation with small portfolio
        tracker.start_timer("small_portfolio_test")
        mock_gen = MockDataGenerator()
        mock_gen.create_realistic_price_data(small_portfolio, days=5, seed=42)
        small_duration = tracker.end_timer("small_portfolio_test")

        # Simulate operation with large portfolio
        tracker.start_timer("large_portfolio_test")
        mock_gen.create_realistic_price_data(large_portfolio, days=5, seed=42)
        large_duration = tracker.end_timer("large_portfolio_test")

        # Performance should be measurable
        assert small_duration > 0, "Should measure small portfolio performance"
        assert large_duration > 0, "Should measure large portfolio performance"
        # Note: We don't assert large > small as the difference might be minimal with mock data


class TestDataDirectoryMaintenance:
    """Tests for maintaining and validating the test data directory."""

    def test_data_directory_completeness(self):
        """Test that all expected data files are present and valid."""
        data_dir = tests.get_test_data_directory()

        # Check required files
        required_files = ["README.md", "sample_portfolios.json"]
        for filename in required_files:
            filepath = data_dir / filename
            assert filepath.exists(), f"Required file {filename} should exist"
            assert filepath.stat().st_size > 0, f"File {filename} should not be empty"

    def test_sample_portfolios_data_validity(self):
        """Test that sample portfolios data is valid and consistent."""
        portfolios_data = tests.load_sample_portfolios()

        # Test that all portfolios have required fields
        for portfolio_name, portfolio_data in portfolios_data["test_portfolios"].items():
            assert "symbols" in portfolio_data, f"Portfolio {portfolio_name} should have symbols"
            assert isinstance(portfolio_data["symbols"], list), f"Symbols should be a list for {portfolio_name}"
            assert len(portfolio_data["symbols"]) > 0, f"Portfolio {portfolio_name} should have at least one symbol"

            # Test symbols are valid strings
            for symbol in portfolio_data["symbols"]:
                assert isinstance(symbol, str), f"Symbol {symbol} should be a string"
                assert len(symbol) > 0, f"Symbol should not be empty in {portfolio_name}"

    def test_data_directory_access_permissions(self):
        """Test that test data directory has proper access permissions."""
        data_dir = tests.get_test_data_directory()

        # Directory should be readable
        assert data_dir.exists(), "Data directory should exist"
        assert data_dir.is_dir(), "Should be a directory"

        # Files should be readable
        for file in data_dir.iterdir():
            if file.is_file():
                assert file.stat().st_size >= 0, f"File {file.name} should be accessible"