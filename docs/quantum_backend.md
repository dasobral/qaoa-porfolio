# Quantum Backend

The Phase 3 quantum backend implements a PennyLane QAOA solver for QUBO matrices produced by the Rust core or supplied as NumPy-compatible arrays. It lives in `qaoa_portfolio/quantum_backend.py` and is exported from the package root.

## Public API

```python
from qaoa_portfolio import QAOAConfig, QAOAQuantumBackend, solve_qubo_qaoa
```

Primary objects:

- `QAOAConfig` validates QAOA layers, optimizer, backend, iteration limits, shots, seed, and restart count.
- `QAOAQuantumBackend.solve(qubo, labels=None)` runs QAOA and returns a `QAOAResult`.
- `solve_qubo_qaoa(qubo, labels=None, config=None)` is a convenience wrapper around `QAOAQuantumBackend`.
- `QAOAResult.to_dict()` returns JSON-safe values for reporting or later visualization.

Helper functions:

- `build_cost_hamiltonian(qubo, offset=0.0)`
- `build_mixer_hamiltonian(num_wires)`
- `evaluate_qubo_bitstring(qubo, bitstring, offset=0.0)`
- `bitstring_to_solution(bitstring)`
- `decode_solution(bitstring, labels=None)`

## Configuration

```python
config = QAOAConfig(
    layers=1,
    optimizer="gradient_descent",
    max_iterations=20,
    convergence_threshold=1e-8,
    shots=None,
    seed=42,
    backend="default.qubit",
    num_restarts=2,
    max_stored_solutions=64,
)
```

Supported optimizers are `adam`, `gradient_descent`, `cobyla`, and `nelder_mead`. Supported PennyLane devices are `default.qubit` and `lightning.qubit`. Use `shots=None` for deterministic statevector probabilities in tests and small local experiments.

`max_stored_solutions` (default 64) caps how many of the most probable basis states are kept in `QAOAResult.probabilities` and considered for ranking — the full 2ⁿ distribution grows exponentially and is never needed downstream. For n ≤ 6 the default keeps every state, so small-instance behavior is exact.

## QUBO Input

The solver accepts either:

1. `qaoa_portfolio_core.PyQUBOMatrix`
2. A square symmetric NumPy-compatible matrix

`PyQUBOMatrix` inputs use `.to_numpy()` and `.offset`. Labels are not stored by the Rust QUBO object, so pass the original symbols when asset names are needed:

```python
result = solve_qubo_qaoa(rust_qubo, labels=["AAPL", "MSFT", "NVDA"], config=config)
```

Matrices must be non-empty, square, finite, and symmetric within `1e-9`.

## Hamiltonian Mapping

The backend maps QUBO variables with `x_i = (1 - Z_i) / 2`. QUBO evaluation follows the Phase 2 upper-triangle convention:

```text
offset + sum_i Q[i, i] x_i + sum_i<j Q[i, j] x_i x_j
```

The mixer Hamiltonian is the standard X-mixer with one Pauli-X term per wire.

## Result Fields

`QAOAResult` includes:

- `best_bitstring` and `best_solution`
- `selected_indices` and `selected_assets`
- `objective_value`
- `probabilities`
- `top_solutions`
- `optimal_parameters`
- `convergence_history`
- `iterations`, `elapsed_ms`, and `metadata`

Top solutions are ranked by original QUBO objective value ascending, among the `max_stored_solutions` most probable basis states (`probabilities` carries exactly that capped set; the cap is echoed in `metadata["max_stored_solutions"]`).

## End-to-End Example

```python
import numpy as np
import qaoa_portfolio_core
from qaoa_portfolio import QAOAConfig, solve_qubo_qaoa

prices = np.full((70, 4), 100.0)
symbols = ["A", "B", "C", "D"]

qubo = qaoa_portfolio_core.build_qubo(prices, symbols, 0.5, 2)
result = solve_qubo_qaoa(
    qubo,
    labels=symbols,
    config=QAOAConfig(layers=1, max_iterations=10, num_restarts=1),
)

print(result.best_bitstring)
print(result.selected_assets)
print(result.objective_value)
```

## Verification

Run the Phase 3 tests:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv run pytest tests/test_quantum_backend.py
UV_PROJECT_ENVIRONMENT=qaoa-env uv run pytest tests/test_qaoa_integration.py
```

Run the full project checks before merging Phase 3:

```bash
UV_PROJECT_ENVIRONMENT=qaoa-env uv run pytest
cargo test
cargo clippy -- -D warnings
python -m maturin build --features python-bindings
```

## Current Limits

The backend targets simulator-backed QAOA and binary include/exclude portfolio selection. Visualization, large benchmark studies, and alternative mixers are deferred to later roadmap phases.
