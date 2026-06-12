import numpy as np
import pytest

from qaoa_portfolio.quantum_backend import (
    QAOAConfig,
    QAOAQuantumBackend,
    solve_qubo_qaoa,
)
from tests.utils import MockDataGenerator

qaoa_portfolio_core = pytest.importorskip("qaoa_portfolio_core")

pytestmark = pytest.mark.integration


def trend_prices(periods=70):
    base = np.arange(periods, dtype=np.float64)
    return np.column_stack(
        [
            100.0 + base * 0.12,
            120.0 + base * 0.09,
            80.0 + base * 0.06,
            95.0 + base * 0.03,
        ]
    )


def tiny_config(seed=7):
    return QAOAConfig(
        layers=1,
        optimizer="gradient_descent",
        max_iterations=6,
        convergence_threshold=1e-12,
        seed=seed,
        num_restarts=1,
    )


def test_rust_qubo_solves_with_qaoa_backend_and_matches_brute_force():
    symbols = ["A", "B", "C", "D"]
    qubo = qaoa_portfolio_core.build_qubo(trend_prices(), symbols, 0.5, 2)
    brute_force = qaoa_portfolio_core.solve_brute_force(qubo)

    backend = QAOAQuantumBackend(tiny_config())
    result = backend.solve(qubo, labels=symbols)

    assert result.objective_value == pytest.approx(brute_force.objective_value)
    assert result.best_solution == list(brute_force.solution)
    assert result.selected_assets == list(brute_force.selected_assets)
    assert set(result.selected_assets).issubset(symbols)
    assert result.metadata["source"] == "PyQUBOMatrix"


def test_qaoa_result_ranks_top_solutions_for_small_rust_qubo():
    symbols = ["A", "B", "C"]
    qubo = qaoa_portfolio_core.build_qubo(trend_prices()[:, :3], symbols, 0.4, 2)

    result = solve_qubo_qaoa(qubo, labels=symbols, config=tiny_config(seed=9))
    objectives = [entry["objective_value"] for entry in result.top_solutions]

    assert len(result.top_solutions) == 8
    assert objectives == sorted(objectives)
    assert result.top_solutions[0]["bitstring"] == result.best_bitstring
    assert all(
        set(entry["selected_assets"]).issubset(symbols)
        for entry in result.top_solutions
    )


def test_mock_market_data_to_rust_qubo_to_qaoa_workflow():
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN"]
    market_data = MockDataGenerator().create_realistic_price_data(
        symbols, days=75, seed=42
    )
    prices = np.column_stack(
        [
            market_data[(symbol, "close")].to_numpy(dtype=np.float64)
            for symbol in symbols
        ]
    )

    qubo = qaoa_portfolio_core.build_qubo(prices, symbols, 0.6, 2)
    result = solve_qubo_qaoa(qubo, labels=symbols, config=tiny_config(seed=11))

    assert len(result.best_solution) == len(symbols)
    assert len(result.best_bitstring) == len(symbols)
    assert np.isfinite(result.objective_value)
    assert set(result.selected_assets).issubset(symbols)
    assert result.probabilities
    assert result.metadata["num_variables"] == len(symbols)
