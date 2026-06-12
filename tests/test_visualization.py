"""
Tests for qaoa_portfolio.visualization (Phase 4).

Chart-data helpers are validated without rendering; figure-producing
functions are checked headlessly with the Matplotlib "Agg" backend.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from qaoa_portfolio.exceptions import VisualizationError  # noqa: E402
from qaoa_portfolio.metrics import FinancialMetrics  # noqa: E402
from qaoa_portfolio.quantum_backend import (  # noqa: E402
    QAOAConfig,
    QAOAResult,
    solve_qubo_qaoa,
)
from qaoa_portfolio.visualization import (  # noqa: E402
    VisualizationConfig,
    normalize_qaoa_result,
    plot_correlation_heatmap,
    plot_efficient_frontier,
    plot_portfolio_composition,
    plot_qaoa_convergence,
    plot_risk_return_scatter,
    plot_solution_probabilities,
    plot_solver_comparison,
    plot_top_solutions,
    prepare_composition_data,
    prepare_probability_data,
    prepare_risk_return_data,
    prepare_solver_comparison_data,
    prepare_top_solutions_data,
    render_qaoa_circuit_summary,
)
from tests.utils import MockDataGenerator  # noqa: E402

# ============================================================================
# Fixtures and helpers
# ============================================================================


def make_result_dict():
    """Deterministic 3-qubit QAOA result payload matching Phase 3 fields."""
    return {
        "best_bitstring": "101",
        "best_solution": [True, False, True],
        "selected_indices": [0, 2],
        "selected_assets": ["AAPL", "NVDA"],
        "objective_value": -1.5,
        "probabilities": {"101": 0.4, "010": 0.3, "111": 0.2, "000": 0.1},
        "top_solutions": [
            {
                "bitstring": "101",
                "selected_assets": ["AAPL", "NVDA"],
                "objective_value": -1.5,
                "probability": 0.4,
            },
            {
                "bitstring": "111",
                "selected_assets": ["AAPL", "MSFT", "NVDA"],
                "objective_value": -1.0,
                "probability": 0.2,
            },
            {
                "bitstring": "010",
                "selected_assets": ["MSFT"],
                "objective_value": -0.5,
                "probability": 0.3,
            },
            {
                "bitstring": "000",
                "selected_assets": [],
                "objective_value": 0.0,
                "probability": 0.1,
            },
        ],
        "optimal_parameters": {"gammas": [0.5], "betas": [0.3]},
        "convergence_history": [-0.2, -0.8, -1.2, -1.5],
        "iterations": 4,
        "elapsed_ms": 10,
        "metadata": {
            "layers": 1,
            "optimizer": "adam",
            "backend": "default.qubit",
            "num_variables": 3,
        },
    }


def make_result_instance():
    payload = make_result_dict()
    return QAOAResult(**payload)


@pytest.fixture
def sample_returns():
    """Deterministic per-asset periodic returns."""
    return pd.DataFrame(
        {
            "AAA": [0.010, -0.020, 0.015, 0.005, 0.002],
            "BBB": [0.020, 0.010, -0.010, 0.000, 0.012],
            "CCC": [-0.005, 0.004, 0.008, -0.002, 0.001],
        }
    )


@pytest.fixture
def frontier_points():
    return pd.DataFrame(
        {
            "return": [0.05, 0.08, 0.12, 0.15],
            "volatility": [0.10, 0.12, 0.18, 0.25],
            "sharpe_ratio": [0.30, 0.50, 0.56, 0.52],
        }
    )


# ============================================================================
# VisualizationConfig
# ============================================================================


@pytest.mark.unit
class TestVisualizationConfig:
    def test_defaults_are_valid(self):
        config = VisualizationConfig()
        assert config.backend == "matplotlib"
        assert config.style == "default"
        assert config.figure_size == (10.0, 6.0)
        assert config.color_palette == "tab10"
        assert config.max_solutions == 10

    def test_rejects_unknown_backend(self):
        with pytest.raises(VisualizationError):
            VisualizationConfig(backend="bokeh")

    @pytest.mark.parametrize(
        "figure_size",
        [(0.0, 6.0), (10.0, -1.0), (10.0,), (1.0, 2.0, 3.0)],
    )
    def test_rejects_invalid_figure_size(self, figure_size):
        with pytest.raises(VisualizationError):
            VisualizationConfig(figure_size=figure_size)

    @pytest.mark.parametrize("max_solutions", [0, -3])
    def test_rejects_non_positive_max_solutions(self, max_solutions):
        with pytest.raises(VisualizationError):
            VisualizationConfig(max_solutions=max_solutions)

    def test_rejects_empty_style_and_palette(self):
        with pytest.raises(VisualizationError):
            VisualizationConfig(style="")
        with pytest.raises(VisualizationError):
            VisualizationConfig(color_palette="")


# ============================================================================
# Result normalization
# ============================================================================


@pytest.mark.unit
class TestNormalizeQAOAResult:
    def test_accepts_dictionary_payload(self):
        payload = normalize_qaoa_result(make_result_dict())
        assert payload["best_bitstring"] == "101"
        assert payload["selected_assets"] == ["AAPL", "NVDA"]

    def test_accepts_qaoa_result_instance(self):
        payload = normalize_qaoa_result(make_result_instance())
        assert payload["best_bitstring"] == "101"
        assert payload["probabilities"]["101"] == pytest.approx(0.4)

    @pytest.mark.parametrize(
        "missing_field",
        [
            "best_bitstring",
            "selected_assets",
            "objective_value",
            "probabilities",
            "top_solutions",
            "convergence_history",
        ],
    )
    def test_missing_fields_raise(self, missing_field):
        payload = make_result_dict()
        del payload[missing_field]
        with pytest.raises(VisualizationError, match=missing_field):
            normalize_qaoa_result(payload)

    def test_empty_probabilities_raise(self):
        payload = make_result_dict()
        payload["probabilities"] = {}
        with pytest.raises(VisualizationError, match="probabilities"):
            normalize_qaoa_result(payload)

    def test_invalid_solution_records_raise(self):
        payload = make_result_dict()
        payload["top_solutions"] = [{"bitstring": "101"}]
        with pytest.raises(VisualizationError, match="top_solutions"):
            normalize_qaoa_result(payload)

    def test_rejects_non_result_inputs(self):
        with pytest.raises(VisualizationError):
            normalize_qaoa_result(42)


# ============================================================================
# Portfolio composition
# ============================================================================


@pytest.mark.unit
class TestPortfolioComposition:
    def test_defaults_to_equal_weights(self):
        labels, weights = prepare_composition_data(["AAPL", "MSFT", "NVDA"])
        assert labels == ["AAPL", "MSFT", "NVDA"]
        assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_normalizes_provided_weights(self):
        _labels, weights = prepare_composition_data(["A", "B"], weights=[3.0, 1.0])
        assert weights == pytest.approx([0.75, 0.25])

    def test_rejects_mismatched_weights(self):
        with pytest.raises(VisualizationError):
            prepare_composition_data(["A", "B"], weights=[1.0])

    def test_rejects_empty_assets(self):
        with pytest.raises(VisualizationError):
            prepare_composition_data([])

    def test_rejects_non_positive_weights(self):
        with pytest.raises(VisualizationError):
            prepare_composition_data(["A", "B"], weights=[1.0, -1.0])

    def test_returns_matplotlib_figure(self):
        fig = plot_portfolio_composition(["AAPL", "MSFT", "NVDA"])
        assert isinstance(fig, Figure)
        assert len(fig.axes[0].patches) == 3


# ============================================================================
# Risk-return scatter
# ============================================================================


@pytest.mark.unit
class TestRiskReturnScatter:
    def test_chart_data_matches_annualized_metrics(self, sample_returns):
        data = prepare_risk_return_data(sample_returns)
        for symbol in sample_returns.columns:
            series = sample_returns[symbol]
            assert data.loc[symbol, "annualized_return"] == pytest.approx(
                FinancialMetrics.annualized_return(series)
            )
            assert data.loc[symbol, "annualized_volatility"] == pytest.approx(
                FinancialMetrics.annualized_volatility(series)
            )

    def test_rejects_empty_returns(self):
        with pytest.raises(VisualizationError):
            prepare_risk_return_data(pd.DataFrame())

    def test_rejects_unknown_highlighted_assets(self, sample_returns):
        with pytest.raises(VisualizationError):
            plot_risk_return_scatter(sample_returns, highlighted_assets=["ZZZ"])

    def test_returns_matplotlib_figure(self, sample_returns):
        fig = plot_risk_return_scatter(sample_returns, highlighted_assets=["AAA"])
        assert isinstance(fig, Figure)


# ============================================================================
# Correlation heatmap
# ============================================================================


@pytest.mark.unit
class TestCorrelationHeatmap:
    def test_uses_square_correlation_matrix_with_labels(self, sample_returns):
        fig = plot_correlation_heatmap(sample_returns)
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == list(sample_returns.columns)
        labels_y = [tick.get_text() for tick in ax.get_yticklabels()]
        assert labels_y == list(sample_returns.columns)

    def test_rejects_empty_returns(self):
        with pytest.raises(VisualizationError):
            plot_correlation_heatmap(pd.DataFrame())


# ============================================================================
# Efficient frontier
# ============================================================================


@pytest.mark.unit
class TestEfficientFrontier:
    def test_returns_matplotlib_figure(self, frontier_points):
        fig = plot_efficient_frontier(frontier_points)
        assert isinstance(fig, Figure)

    def test_marks_selected_portfolio(self, frontier_points):
        fig = plot_efficient_frontier(
            frontier_points,
            selected_portfolio={"return": 0.10, "volatility": 0.15},
        )
        assert isinstance(fig, Figure)

    def test_rejects_missing_columns(self):
        with pytest.raises(VisualizationError):
            plot_efficient_frontier(pd.DataFrame({"return": [0.1]}))

    def test_rejects_invalid_selected_portfolio(self, frontier_points):
        with pytest.raises(VisualizationError):
            plot_efficient_frontier(frontier_points, selected_portfolio={"return": 0.1})


# ============================================================================
# QAOA visualizations
# ============================================================================


@pytest.mark.unit
class TestQAOAVisualizations:
    def test_convergence_plot_uses_full_history(self):
        payload = make_result_dict()
        fig = plot_qaoa_convergence(payload)
        assert isinstance(fig, Figure)
        line = fig.axes[0].lines[0]
        assert list(line.get_ydata()) == payload["convergence_history"]

    def test_convergence_accepts_result_instance(self):
        fig = plot_qaoa_convergence(make_result_instance())
        assert isinstance(fig, Figure)

    def test_convergence_rejects_empty_history(self):
        payload = make_result_dict()
        payload["convergence_history"] = []
        with pytest.raises(VisualizationError):
            plot_qaoa_convergence(payload)

    def test_probability_data_sorted_descending(self):
        data = prepare_probability_data(make_result_dict())
        probabilities = [probability for _bitstring, probability in data]
        assert probabilities == sorted(probabilities, reverse=True)
        assert data[0][0] == "101"

    def test_probability_plot_respects_max_solutions(self):
        config = VisualizationConfig(max_solutions=2)
        data = prepare_probability_data(make_result_dict(), max_solutions=2)
        assert len(data) == 2
        fig = plot_solution_probabilities(make_result_dict(), config=config)
        assert isinstance(fig, Figure)
        assert len(fig.axes[0].containers[0]) == 2

    def test_top_solutions_sorted_by_objective_ascending(self):
        data = prepare_top_solutions_data(make_result_dict())
        objectives = [entry["objective_value"] for entry in data]
        assert objectives == sorted(objectives)

    def test_top_solutions_plot_respects_max_solutions(self):
        config = VisualizationConfig(max_solutions=3)
        data = prepare_top_solutions_data(make_result_dict(), max_solutions=3)
        assert len(data) == 3
        fig = plot_top_solutions(make_result_dict(), config=config)
        assert isinstance(fig, Figure)
        assert len(fig.axes[0].containers[0]) == 3

    def test_circuit_summary_is_text(self):
        summary = render_qaoa_circuit_summary(make_result_dict())
        assert isinstance(summary, str)
        assert "Qubits" in summary
        assert "101" in summary
        assert "adam" in summary

    def test_circuit_summary_accepts_result_instance(self):
        summary = render_qaoa_circuit_summary(make_result_instance())
        assert "AAPL" in summary


# ============================================================================
# Solver comparison
# ============================================================================


@pytest.mark.unit
class TestSolverComparison:
    def make_records(self):
        return [
            {"solver_name": "qaoa", "objective_value": -1.5},
            {"solver_name": "brute_force", "objective_value": -1.6},
            {"solver_name": "simulated_annealing", "objective_value": -1.55},
        ]

    def test_chart_data_keeps_solver_labels(self):
        data = prepare_solver_comparison_data(self.make_records(), "objective_value")
        assert [solver for solver, _value in data] == [
            "qaoa",
            "brute_force",
            "simulated_annealing",
        ]

    def test_rejects_missing_metric(self):
        records = [{"solver_name": "qaoa"}]
        with pytest.raises(VisualizationError, match="objective_value"):
            prepare_solver_comparison_data(records, "objective_value")

    def test_rejects_missing_solver_name(self):
        records = [{"objective_value": -1.0}]
        with pytest.raises(VisualizationError, match="solver_name"):
            prepare_solver_comparison_data(records, "objective_value")

    def test_rejects_empty_records(self):
        with pytest.raises(VisualizationError):
            prepare_solver_comparison_data([], "objective_value")

    def test_returns_matplotlib_figure(self):
        fig = plot_solver_comparison(self.make_records())
        assert isinstance(fig, Figure)
        assert len(fig.axes[0].containers[0]) == 3


# ============================================================================
# Plotly backend
# ============================================================================


@pytest.mark.unit
class TestPlotlyBackend:
    @pytest.fixture(autouse=True)
    def plotly_config(self):
        pytest.importorskip("plotly")
        self.go = pytest.importorskip("plotly.graph_objects")
        self.config = VisualizationConfig(backend="plotly")

    def test_portfolio_composition(self):
        fig = plot_portfolio_composition(["A", "B"], config=self.config)
        assert isinstance(fig, self.go.Figure)

    def test_risk_return_scatter(self, sample_returns):
        fig = plot_risk_return_scatter(sample_returns, config=self.config)
        assert isinstance(fig, self.go.Figure)

    def test_correlation_heatmap(self, sample_returns):
        fig = plot_correlation_heatmap(sample_returns, config=self.config)
        assert isinstance(fig, self.go.Figure)

    def test_efficient_frontier(self, frontier_points):
        fig = plot_efficient_frontier(frontier_points, config=self.config)
        assert isinstance(fig, self.go.Figure)

    def test_qaoa_convergence(self):
        fig = plot_qaoa_convergence(make_result_dict(), config=self.config)
        assert isinstance(fig, self.go.Figure)

    def test_solution_probabilities(self):
        fig = plot_solution_probabilities(make_result_dict(), config=self.config)
        assert isinstance(fig, self.go.Figure)

    def test_top_solutions(self):
        fig = plot_top_solutions(make_result_dict(), config=self.config)
        assert isinstance(fig, self.go.Figure)

    def test_solver_comparison(self):
        records = [
            {"solver_name": "qaoa", "objective_value": -1.5},
            {"solver_name": "brute_force", "objective_value": -1.6},
        ]
        fig = plot_solver_comparison(records, config=self.config)
        assert isinstance(fig, self.go.Figure)


# ============================================================================
# Integration tests
# ============================================================================


@pytest.mark.integration
class TestVisualizationIntegration:
    def test_runs_headless_with_agg_backend(self):
        assert matplotlib.get_backend().lower() == "agg"

    def test_rust_qubo_qaoa_results_render(self):
        qaoa_portfolio_core = pytest.importorskip("qaoa_portfolio_core")
        base = np.arange(60, dtype=np.float64)
        prices = np.column_stack(
            [100.0 + base * 0.12, 120.0 + base * 0.09, 80.0 + base * 0.06]
        )
        symbols = ["A", "B", "C"]
        qubo = qaoa_portfolio_core.build_qubo(prices, symbols, 0.5, 2)
        result = solve_qubo_qaoa(
            qubo,
            labels=symbols,
            config=QAOAConfig(
                layers=1,
                optimizer="gradient_descent",
                max_iterations=5,
                convergence_threshold=1e-12,
                seed=7,
                num_restarts=1,
            ),
        )

        convergence_fig = plot_qaoa_convergence(result)
        probabilities_fig = plot_solution_probabilities(result)
        top_fig = plot_top_solutions(result)
        summary = render_qaoa_circuit_summary(result)

        assert isinstance(convergence_fig, Figure)
        assert isinstance(probabilities_fig, Figure)
        assert isinstance(top_fig, Figure)
        assert "Qubits" in summary

        # The same payload must render from the serialized dictionary form.
        dict_fig = plot_qaoa_convergence(result.to_dict())
        assert isinstance(dict_fig, Figure)

    def test_mock_market_data_renders_portfolio_figures(self):
        symbols = ["AAPL", "MSFT", "NVDA"]
        market_data = MockDataGenerator().create_realistic_price_data(
            symbols, days=80, seed=42
        )
        returns = FinancialMetrics.calculate_returns(market_data)

        risk_return_fig = plot_risk_return_scatter(returns, highlighted_assets=["AAPL"])
        heatmap_fig = plot_correlation_heatmap(returns)
        composition_fig = plot_portfolio_composition(symbols)

        assert isinstance(risk_return_fig, Figure)
        assert isinstance(heatmap_fig, Figure)
        assert isinstance(composition_fig, Figure)
