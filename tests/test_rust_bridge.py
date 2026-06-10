import numpy as np
import pytest

qaoa_portfolio_core = pytest.importorskip("qaoa_portfolio_core")


def constant_prices(periods=70, assets=4):
    return np.full((periods, assets), 100.0, dtype=np.float64)


def trend_prices(periods=70):
    base = np.arange(periods, dtype=np.float64)
    return np.column_stack(
        [
            100.0 + base * 0.10,
            120.0 + base * 0.08,
            80.0 + base * 0.05,
        ]
    )


def test_build_qubo_returns_numpy_compatible_matrix():
    qubo = qaoa_portfolio_core.build_qubo(
        constant_prices(),
        ["A", "B", "C", "D"],
        0.5,
        2,
    )

    matrix = qubo.to_numpy()
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T)
    assert np.isfinite(qubo.evaluate([True, True, False, False]))


def test_brute_force_bridge_returns_feasible_result_dict():
    qubo = qaoa_portfolio_core.build_qubo(
        constant_prices(),
        ["A", "B", "C", "D"],
        0.5,
        2,
    )

    result = qaoa_portfolio_core.solve_brute_force(qubo)
    payload = result.to_dict()

    assert payload["solver_name"] == "brute-force"
    assert len(payload["solution"]) == 4
    assert sum(payload["solution"]) == 2
    assert len(payload["selected_assets"]) == 2
    assert np.isfinite(payload["objective_value"])


def test_simulated_annealing_bridge_accepts_seeded_parameters():
    qubo = qaoa_portfolio_core.build_qubo(
        constant_prices(),
        ["A", "B", "C", "D"],
        0.5,
        2,
    )

    result = qaoa_portfolio_core.solve_simulated_annealing(
        qubo,
        initial_temperature=10.0,
        cooling_rate=0.98,
        max_iterations=1_000,
        seed=11,
    )

    assert len(result.solution) == 4
    assert sum(result.solution) == 2
    assert result.solver_name == "simulated-annealing"


def test_markowitz_bridge_weights_sum_to_one():
    result = qaoa_portfolio_core.solve_markowitz(
        trend_prices(),
        ["A", "B", "C"],
    )

    assert set(result) >= {
        "weights",
        "expected_return",
        "volatility",
        "sharpe_ratio",
        "symbols",
    }
    assert np.isclose(sum(result["weights"]), 1.0)
    assert result["symbols"] == ["A", "B", "C"]


def test_bridge_maps_invalid_inputs_to_python_exceptions():
    prices = constant_prices()
    prices[0, 0] = -1.0

    with pytest.raises(ValueError):
        qaoa_portfolio_core.build_qubo(prices, ["A", "B", "C", "D"], 0.5, 2)
