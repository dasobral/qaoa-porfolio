# QAOA Portfolio Optimizer (QOPO)

A high-performance implementation of the Quantum Approximate Optimization Algorithm (QAOA) for portfolio optimization problems, demonstrating quantum-inspired solutions for real-world financial applications.

## 🎯 Overview

This project showcases how quantum-inspired algorithms can solve complex portfolio optimization problems that are challenging for classical methods. By implementing QAOA with classical simulation, we bridge the gap between current optimization capabilities and future quantum computing advantages.

## Quick Start

### Prerequisites

Install Rust and `uv` before setting up the project:

```bash
# Rust toolchain for the QUBO core and PyO3 extension
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# uv manages the Python environment and builds the maturin extension
python -m pip install uv
```

### Installation

```bash
git clone https://github.com/your-username/qaoa-portfolio.git
cd qaoa-portfolio
export UV_PROJECT_ENVIRONMENT=qaoa-env
uv sync --extra dev
source qaoa-env/bin/activate
qaoa-portfolio --help
```

`UV_PROJECT_ENVIRONMENT=qaoa-env` makes uv use `qaoa-env/` as the project environment instead of `.venv/`. Keep that variable exported in shells where you run `uv sync` or `uv run`; otherwise uv will fall back to `.venv` and may warn that `VIRTUAL_ENV=qaoa-env` does not match the project environment.

Build the Rust core directly when working on Rust internals:

```bash
cargo build --release
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

### ✅ PennyLane QAOA Quantum Backend (Completed)

- Cost Hamiltonian construction from Phase 2 QUBO matrices
- Standard X-mixer and configurable QAOA layer count
- Variational optimization with Adam, gradient descent, COBYLA, and Nelder-Mead
- Deterministic statevector runs for tests and optional shot-based sampling
- Ranked bitstring decoding into selected portfolio assets
- Rust QUBO bridge integration through `qaoa_portfolio_core.PyQUBOMatrix`

See [Quantum Backend API](docs/quantum_backend.md) for usage and configuration details.

### 🚧 In Development

- **Visualization:** Portfolio analysis and optimization result visualization

### Current CLI

After installing with `UV_PROJECT_ENVIRONMENT=qaoa-env uv sync --extra dev`:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv run qaoa-portfolio --help
```

Example with a preset portfolio:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv run qaoa-portfolio --preset growth_stocks --days-back 180
```

### Current Limits

- `qaoa_portfolio/visualization.py` is currently a placeholder.
- Quantum backend tests use small simulator problems; large scaling studies are deferred to benchmarking phases.
- Phase 4 will consume QAOA result probabilities, convergence history, and ranked portfolios for visualization.

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
