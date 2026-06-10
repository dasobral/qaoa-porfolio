# QAOA Portfolio Optimizer (QOPO)

A high-performance implementation of the Quantum Approximate Optimization Algorithm (QAOA) for portfolio optimization problems, demonstrating quantum-inspired solutions for real-world financial applications.

## 🎯 Overview

This project showcases how quantum-inspired algorithms can solve complex portfolio optimization problems that are challenging for classical methods. By implementing QAOA with classical simulation, we bridge the gap between current optimization capabilities and future quantum computing advantages.

## Quick Start

### Environment Setup

It is strongly recommended to work within a Python virtual environment to manage dependencies and avoid conflicts with system packages.

```bash
# Create and activate virtual environment
python -m venv qaoa-env
source qaoa-env/bin/activate  

# On Windows: qaoa-env\Scripts\activate

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel
```

### Prerequisites

```bash
# For Rust implementation
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# For Python quantum backend and market data components
pip install pennylane numpy pandas matplotlib yfinance
pip install pytest black flake8 jupyter ipykernel
```

### Installation

```bash
git clone https://github.com/your-username/qaoa-portfolio.git
cd qaoa-portfolio

# Build classical core (Rust)
cargo build --release

# Install Python components in development mode
pip install -e ".[dev]"

# Build the optional Rust/Python extension wheel
python -m pip install maturin
python -m maturin build --features python-bindings
```

## Current Implementation Status

The QAOA Portfolio Optimizer is currently in active development with the following components implemented:

### ✅ Market Data Loader (Completed)

**Status:** Production-ready

- Yahoo Finance integration (100% free, no API keys required)
- Async data loading with performance monitoring
- Comprehensive data validation (dual-level: per-asset + portfolio-wide)
- Smart caching system with configurable duration
- Support for stocks, cryptocurrencies, and mixed portfolios
- Configuration-driven behavior with free-tier optimization

**Features:**

- `MarketDataLoader` class for async data loading
- Portfolio utilities for stocks, crypto, and mixed assets
- Predefined portfolio presets (conservative, growth, DeFi, etc.)
- Quick-start functions for rapid prototyping
- Professional error handling and logging

For detailed API documentation and usage examples, see [Market Data Loader Documentation](docs/dataloader.md).

### ✅ Rust QUBO Core (Completed)

- Portfolio and return-series data structures in Rust
- Covariance/correlation statistics and Markowitz-to-QUBO formulation
- Budget, position, and diversification penalty builders
- Brute-force, simulated annealing, and Markowitz baseline solvers
- PyO3 bridge module: `qaoa_portfolio_core`

See [Rust Core API](docs/rust_core.md) for usage and build details.

### 🚧 In Development

- **Quantum Backend:** QAOA algorithm implementation using PennyLane
- **Visualization:** Portfolio analysis and optimization result visualization

### Current CLI

After installing the Python package in editable mode:

```bash
cd python
qaoa-portfolio --help
```

Example with a preset portfolio:

```bash
cd python
qaoa-portfolio --preset growth_stocks --days-back 180
```

### Current Limits

- `qaoa_portfolio/quantum_backend.py` is currently a placeholder.
- `qaoa_portfolio/visualization.py` is currently a placeholder.
- Phase 3 will consume Rust-generated QUBO matrices in the PennyLane quantum backend.

### 📋 Planned Components

- Advanced portfolio optimization algorithms
- Risk analysis and stress testing
- Backtesting framework
- Web-based dashboard
- Quantum readiness consulting tools

## Related Work

- Quantum machine learning for finance
- Variational quantum algorithms
- Portfolio optimization with quantum computing

## 📄 License

This project is licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives) - see the [LICENSE](LICENSE) file for details.
For commercial use, please contact the author to discuss licensing terms.

## 🤝 Acknowledgments

PennyLane Team for excellent quantum computing framework
Yahoo Finance for free market data access
