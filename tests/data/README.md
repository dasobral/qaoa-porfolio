# Test Data Directory

This directory contains test data files and fixtures used by the test suite.

## Contents

- `sample_portfolios.json` - Predefined test portfolios
- `mock_price_data/` - Sample price data for testing (when needed)
- `config_samples/` - Sample configuration files for testing

## Usage

Test data is automatically managed by the test utilities. Use the test package's
`get_test_data_directory()` function to access this directory programmatically.

## Guidelines

- Keep test data files small and focused
- Use representative but synthetic data when possible
- Document any real market data sources and ensure compliance with usage terms
- Clean up temporary test data in test teardown methods
