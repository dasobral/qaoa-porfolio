"""
Visualization layer for QAOA Portfolio Optimizer (QOPO)

Reusable plotting and chart-data helpers for portfolio composition,
risk-return structure, QAOA optimization behavior, and solver comparisons.
Plotting functions return figure objects and never call ``show()``;
chart-data helpers are independent from any rendering backend.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from .exceptions import VisualizationError
from .metrics import FinancialMetrics
from .utils import normalize_weights

if TYPE_CHECKING:  # heavy PennyLane-backed module: type-checking import only
    from .quantum_backend import QAOAResult

SUPPORTED_VISUALIZATION_BACKENDS = {"matplotlib", "plotly"}
REQUIRED_RESULT_FIELDS = (
    "best_bitstring",
    "selected_assets",
    "objective_value",
    "probabilities",
    "top_solutions",
    "convergence_history",
)
REQUIRED_SOLUTION_FIELDS = (
    "bitstring",
    "selected_assets",
    "objective_value",
    "probability",
)
TRADING_PERIODS_PER_YEAR = 252

# Payloads are duck-typed through normalize_qaoa_result so dictionaries work
# identically to QAOAResult instances; the union is real for type checkers.
ResultLike = Union["QAOAResult", Dict[str, Any]]


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for Phase 4 visualization helpers."""

    backend: str = "matplotlib"
    style: str = "default"
    figure_size: Tuple[float, float] = (10.0, 6.0)
    color_palette: str = "tab10"
    max_solutions: int = 10

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_VISUALIZATION_BACKENDS:
            raise VisualizationError(
                f"Unsupported backend '{self.backend}'. "
                f"Expected one of {sorted(SUPPORTED_VISUALIZATION_BACKENDS)}"
            )
        if not isinstance(self.style, str) or not self.style:
            raise VisualizationError("style must be a non-empty string")
        if not isinstance(self.color_palette, str) or not self.color_palette:
            raise VisualizationError("color_palette must be a non-empty string")
        if (
            not isinstance(self.figure_size, Sequence)
            or len(self.figure_size) != 2
            or not all(
                isinstance(value, Real) and float(value) > 0.0
                for value in self.figure_size
            )
        ):
            raise VisualizationError(
                "figure_size must contain exactly two positive numbers"
            )
        if not isinstance(self.max_solutions, int) or self.max_solutions <= 0:
            raise VisualizationError("max_solutions must be a positive integer")


# ============================================================================
# Result input normalization
# ============================================================================


def normalize_qaoa_result(result: ResultLike) -> Dict[str, Any]:
    """Normalize a ``QAOAResult`` or dictionary payload into a plain dict.

    Validates the Phase 3 result contract: required fields, a non-empty
    probability map, and well-formed top-solution records.
    """

    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    elif isinstance(result, Mapping):
        payload = dict(result)
    else:
        raise VisualizationError(
            "QAOA result must be a QAOAResult or a mapping, "
            f"got {type(result).__name__}"
        )

    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in payload]
    if missing:
        raise VisualizationError(
            f"QAOA result is missing required fields: {', '.join(missing)}"
        )

    probabilities = payload["probabilities"]
    if not isinstance(probabilities, Mapping) or not probabilities:
        raise VisualizationError(
            "QAOA result probabilities must be a non-empty mapping"
        )
    for bitstring, probability in probabilities.items():
        if not isinstance(probability, Real) or not np.isfinite(float(probability)):
            raise VisualizationError(
                f"Probability for bitstring '{bitstring}' must be a finite number"
            )

    top_solutions = payload["top_solutions"]
    if not isinstance(top_solutions, Sequence) or isinstance(top_solutions, str):
        raise VisualizationError("QAOA result top_solutions must be a sequence")
    for index, record in enumerate(top_solutions):
        if not isinstance(record, Mapping):
            raise VisualizationError(f"top_solutions record {index} must be a mapping")
        missing_keys = [key for key in REQUIRED_SOLUTION_FIELDS if key not in record]
        if missing_keys:
            raise VisualizationError(
                f"top_solutions record {index} is missing fields: "
                f"{', '.join(missing_keys)}"
            )

    history = payload["convergence_history"]
    if not isinstance(history, Sequence) or isinstance(history, str):
        raise VisualizationError(
            "QAOA result convergence_history must be a sequence of numbers"
        )

    return payload


# ============================================================================
# Chart-data helpers (rendering-free)
# ============================================================================


def prepare_composition_data(
    selected_assets: Sequence[str],
    weights: Optional[Sequence[float]] = None,
) -> Tuple[List[str], List[float]]:
    """Return asset labels and normalized weights for composition charts."""

    labels = [str(asset) for asset in selected_assets]
    if not labels:
        raise VisualizationError("selected_assets must be non-empty")

    if weights is None:
        return labels, normalize_weights([1.0] * len(labels))

    values = [float(weight) for weight in weights]
    if len(values) != len(labels):
        raise VisualizationError(
            f"weights length ({len(values)}) must match "
            f"selected_assets length ({len(labels)})"
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise VisualizationError("weights must be positive finite numbers")

    return labels, normalize_weights(values)


def prepare_risk_return_data(
    returns: pd.DataFrame,
    periods_per_year: int = TRADING_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Return annualized return/volatility per asset for risk-return charts."""

    _validate_returns_frame(returns)
    rows = {
        str(symbol): {
            "annualized_return": FinancialMetrics.annualized_return(
                returns[symbol].dropna(), periods_per_year
            ),
            "annualized_volatility": FinancialMetrics.annualized_volatility(
                returns[symbol].dropna(), periods_per_year
            ),
        }
        for symbol in returns.columns
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def prepare_probability_data(
    result: ResultLike,
    max_solutions: int = 10,
) -> List[Tuple[str, float]]:
    """Return (bitstring, probability) pairs sorted by probability descending."""

    payload = normalize_qaoa_result(result)
    if max_solutions <= 0:
        raise VisualizationError("max_solutions must be a positive integer")
    ranked = sorted(
        payload["probabilities"].items(),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return [(bitstring, float(probability)) for bitstring, probability in ranked][
        :max_solutions
    ]


def prepare_top_solutions_data(
    result: ResultLike,
    max_solutions: int = 10,
) -> List[Dict[str, Any]]:
    """Return top-solution records sorted by objective value ascending."""

    payload = normalize_qaoa_result(result)
    if max_solutions <= 0:
        raise VisualizationError("max_solutions must be a positive integer")
    ranked = sorted(
        payload["top_solutions"],
        key=lambda record: (float(record["objective_value"]), record["bitstring"]),
    )
    return [dict(record) for record in ranked[:max_solutions]]


def prepare_solver_comparison_data(
    results: Sequence[Mapping[str, Any]],
    metric: str = "objective_value",
) -> List[Tuple[str, float]]:
    """Return (solver_name, metric value) pairs for comparison charts."""

    if not isinstance(metric, str) or not metric:
        raise VisualizationError("metric must be a non-empty string")
    if not results:
        raise VisualizationError("results must contain at least one record")

    data: List[Tuple[str, float]] = []
    for index, record in enumerate(results):
        if "solver_name" not in record:
            raise VisualizationError(f"results record {index} is missing 'solver_name'")
        if metric not in record:
            raise VisualizationError(
                f"results record {index} ('{record['solver_name']}') "
                f"is missing metric '{metric}'"
            )
        value = record[metric]
        if not isinstance(value, Real) or not np.isfinite(float(value)):
            raise VisualizationError(
                f"Metric '{metric}' for solver '{record['solver_name']}' "
                "must be a finite number"
            )
        data.append((str(record["solver_name"]), float(value)))
    return data


# ============================================================================
# Portfolio visualizations
# ============================================================================


def plot_portfolio_composition(
    selected_assets: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    config: Optional[VisualizationConfig] = None,
) -> Any:
    """Plot portfolio composition as a pie chart (equal weights by default)."""

    cfg = config or VisualizationConfig()
    labels, normalized = prepare_composition_data(selected_assets, weights)

    if cfg.backend == "plotly":
        go = _import_plotly()
        fig = go.Figure(go.Pie(labels=labels, values=normalized, sort=False))
        fig.update_layout(title="Portfolio Composition")
        return fig

    fig, ax = _new_matplotlib_axes(cfg)
    ax.pie(
        normalized,
        labels=labels,
        autopct="%1.1f%%",
        colors=_palette_colors(cfg, len(labels)),
        startangle=90,
    )
    ax.set_title("Portfolio Composition")
    return fig


def plot_risk_return_scatter(
    returns: pd.DataFrame,
    highlighted_assets: Optional[Sequence[str]] = None,
    config: Optional[VisualizationConfig] = None,
) -> Any:
    """Plot annualized volatility vs annualized return per asset."""

    cfg = config or VisualizationConfig()
    data = prepare_risk_return_data(returns)
    highlighted = [str(asset) for asset in (highlighted_assets or [])]
    unknown = sorted(set(highlighted) - set(data.index))
    if unknown:
        raise VisualizationError(
            f"highlighted_assets not present in returns: {', '.join(unknown)}"
        )

    if cfg.backend == "plotly":
        go = _import_plotly()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data["annualized_volatility"],
                y=data["annualized_return"],
                mode="markers+text",
                text=list(data.index),
                textposition="top center",
                marker={
                    "size": 12,
                    "color": [
                        "crimson" if symbol in highlighted else "steelblue"
                        for symbol in data.index
                    ],
                },
            )
        )
        fig.update_layout(
            title="Risk-Return Profile",
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
        )
        return fig

    fig, ax = _new_matplotlib_axes(cfg)
    base = data.drop(index=highlighted)
    ax.scatter(
        base["annualized_volatility"],
        base["annualized_return"],
        color="steelblue",
        label="Assets",
        zorder=2,
    )
    if highlighted:
        marked = data.loc[highlighted]
        ax.scatter(
            marked["annualized_volatility"],
            marked["annualized_return"],
            color="crimson",
            marker="*",
            s=180,
            label="Selected",
            zorder=3,
        )
        ax.legend()
    for symbol, row in data.iterrows():
        ax.annotate(
            symbol,
            (row["annualized_volatility"], row["annualized_return"]),
            textcoords="offset points",
            xytext=(5, 5),
        )
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Risk-Return Profile")
    ax.grid(True, alpha=0.3)
    return fig


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    config: Optional[VisualizationConfig] = None,
) -> Any:
    """Plot the asset return correlation matrix as a heatmap."""

    cfg = config or VisualizationConfig()
    _validate_returns_frame(returns)
    correlation = returns.corr()
    labels = [str(symbol) for symbol in correlation.columns]

    if cfg.backend == "plotly":
        go = _import_plotly()
        fig = go.Figure(
            go.Heatmap(
                z=correlation.to_numpy(),
                x=labels,
                y=labels,
                zmin=-1.0,
                zmax=1.0,
                colorscale="RdBu",
            )
        )
        fig.update_layout(title="Asset Return Correlations")
        return fig

    import seaborn as sns  # type: ignore[import-untyped]

    fig, ax = _new_matplotlib_axes(cfg)
    sns.heatmap(
        correlation,
        ax=ax,
        annot=len(labels) <= 12,
        fmt=".2f",
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu",
        square=True,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title("Asset Return Correlations")
    return fig


def plot_efficient_frontier(
    frontier_points: pd.DataFrame,
    selected_portfolio: Optional[Mapping[str, float]] = None,
    config: Optional[VisualizationConfig] = None,
) -> Any:
    """Plot efficient frontier points, optionally marking a selected portfolio.

    ``frontier_points`` requires ``return`` and ``volatility`` columns;
    an optional ``sharpe_ratio`` column colors the points.
    """

    cfg = config or VisualizationConfig()
    if not isinstance(frontier_points, pd.DataFrame) or frontier_points.empty:
        raise VisualizationError("frontier_points must be a non-empty DataFrame")
    missing = [
        column
        for column in ("return", "volatility")
        if column not in frontier_points.columns
    ]
    if missing:
        raise VisualizationError(
            f"frontier_points is missing required columns: {', '.join(missing)}"
        )
    if selected_portfolio is not None:
        missing_keys = [
            key for key in ("return", "volatility") if key not in selected_portfolio
        ]
        if missing_keys:
            raise VisualizationError(
                "selected_portfolio is missing required keys: "
                f"{', '.join(missing_keys)}"
            )
    has_sharpe = "sharpe_ratio" in frontier_points.columns

    if cfg.backend == "plotly":
        go = _import_plotly()
        marker: Dict[str, Any] = {"size": 9}
        if has_sharpe:
            marker.update(
                color=frontier_points["sharpe_ratio"],
                colorscale="Viridis",
                colorbar={"title": "Sharpe Ratio"},
            )
        fig = go.Figure(
            go.Scatter(
                x=frontier_points["volatility"],
                y=frontier_points["return"],
                mode="markers",
                marker=marker,
                name="Frontier",
            )
        )
        if selected_portfolio is not None:
            fig.add_trace(
                go.Scatter(
                    x=[float(selected_portfolio["volatility"])],
                    y=[float(selected_portfolio["return"])],
                    mode="markers",
                    marker={"symbol": "star", "size": 16, "color": "crimson"},
                    name="Selected Portfolio",
                )
            )
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Volatility",
            yaxis_title="Return",
        )
        return fig

    fig, ax = _new_matplotlib_axes(cfg)
    if has_sharpe:
        points = ax.scatter(
            frontier_points["volatility"],
            frontier_points["return"],
            c=frontier_points["sharpe_ratio"],
            cmap="viridis",
            label="Frontier",
            zorder=2,
        )
        fig.colorbar(points, ax=ax, label="Sharpe Ratio")
    else:
        ax.scatter(
            frontier_points["volatility"],
            frontier_points["return"],
            color="steelblue",
            label="Frontier",
            zorder=2,
        )
    if selected_portfolio is not None:
        ax.scatter(
            [float(selected_portfolio["volatility"])],
            [float(selected_portfolio["return"])],
            color="crimson",
            marker="*",
            s=220,
            label="Selected Portfolio",
            zorder=3,
        )
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    ax.set_title("Efficient Frontier")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


# ============================================================================
# QAOA visualizations
# ============================================================================


def plot_qaoa_convergence(
    result: ResultLike,
    config: Optional[VisualizationConfig] = None,
) -> Any:
    """Plot QAOA objective value against optimization iteration index."""

    cfg = config or VisualizationConfig()
    payload = normalize_qaoa_result(result)
    history = [float(value) for value in payload["convergence_history"]]
    if not history:
        raise VisualizationError("convergence_history must be non-empty")
    iterations = list(range(1, len(history) + 1))

    if cfg.backend == "plotly":
        go = _import_plotly()
        fig = go.Figure(go.Scatter(x=iterations, y=history, mode="lines+markers"))
        fig.update_layout(
            title="QAOA Convergence",
            xaxis_title="Iteration",
            yaxis_title="Objective Value",
        )
        return fig

    fig, ax = _new_matplotlib_axes(cfg)
    ax.plot(iterations, history, marker="o", color="steelblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")
    ax.set_title("QAOA Convergence")
    ax.grid(True, alpha=0.3)
    return fig


def plot_solution_probabilities(
    result: ResultLike,
    config: Optional[VisualizationConfig] = None,
) -> Any:
    """Plot measured bitstring probabilities sorted descending."""

    cfg = config or VisualizationConfig()
    data = prepare_probability_data(result, cfg.max_solutions)
    bitstrings = [bitstring for bitstring, _probability in data]
    probabilities = [probability for _bitstring, probability in data]

    if cfg.backend == "plotly":
        go = _import_plotly()
        fig = go.Figure(go.Bar(x=bitstrings, y=probabilities))
        fig.update_layout(
            title="Solution Probabilities",
            xaxis_title="Bitstring",
            yaxis_title="Probability",
        )
        return fig

    fig, ax = _new_matplotlib_axes(cfg)
    ax.bar(bitstrings, probabilities, color=_palette_colors(cfg, len(bitstrings)))
    ax.set_xlabel("Bitstring")
    ax.set_ylabel("Probability")
    ax.set_title("Solution Probabilities")
    return fig


def plot_top_solutions(
    result: ResultLike,
    config: Optional[VisualizationConfig] = None,
) -> Any:
    """Plot top solutions by objective value, labeled with selected assets."""

    cfg = config or VisualizationConfig()
    data = prepare_top_solutions_data(result, cfg.max_solutions)
    bitstrings = [str(record["bitstring"]) for record in data]
    objectives = [float(record["objective_value"]) for record in data]
    asset_labels = [", ".join(record["selected_assets"]) or "(none)" for record in data]

    if cfg.backend == "plotly":
        go = _import_plotly()
        fig = go.Figure(go.Bar(x=bitstrings, y=objectives, text=asset_labels))
        fig.update_layout(
            title="Top QAOA Solutions",
            xaxis_title="Bitstring",
            yaxis_title="Objective Value",
        )
        return fig

    fig, ax = _new_matplotlib_axes(cfg)
    bars = ax.bar(bitstrings, objectives, color=_palette_colors(cfg, len(bitstrings)))
    ax.bar_label(bars, labels=asset_labels, fontsize=8)
    ax.set_xlabel("Bitstring")
    ax.set_ylabel("Objective Value")
    ax.set_title("Top QAOA Solutions")
    return fig


def render_qaoa_circuit_summary(result: ResultLike) -> str:
    """Render a text summary of the QAOA circuit behind a result.

    Phase 4 is text-first by design; rendered circuit diagrams are deferred
    to Phase 6 examples.
    """

    payload = normalize_qaoa_result(result)
    metadata = payload.get("metadata", {}) or {}
    parameters = payload.get("optimal_parameters", {}) or {}
    gammas = [float(value) for value in parameters.get("gammas", [])]
    betas = [float(value) for value in parameters.get("betas", [])]
    num_qubits = metadata.get("num_variables", len(payload["best_bitstring"]))
    layers = metadata.get("layers", len(gammas) or "unknown")

    lines = [
        "QAOA Circuit Summary",
        "====================",
        f"Qubits (assets): {num_qubits}",
        f"QAOA layers (p): {layers}",
        f"Optimizer: {metadata.get('optimizer', 'unknown')}",
        f"Device backend: {metadata.get('backend', 'unknown')}",
        "Structure: |+>^n -> [Cost(gamma_k) -> Mixer(beta_k)] x p -> measure",
        f"Optimal gammas: {gammas}",
        f"Optimal betas: {betas}",
        f"Best bitstring: {payload['best_bitstring']}",
        f"Selected assets: {', '.join(payload['selected_assets']) or '(none)'}",
        f"Objective value: {float(payload['objective_value']):.6f}",
    ]
    return "\n".join(lines)


# ============================================================================
# Solver comparison visualizations
# ============================================================================


def plot_solver_comparison(
    results: Sequence[Mapping[str, Any]],
    metric: str = "objective_value",
    config: Optional[VisualizationConfig] = None,
) -> Any:
    """Plot a metric across solver records (QAOA, brute force, annealing, ...).

    Each record must include ``solver_name`` and the selected ``metric``.
    """

    cfg = config or VisualizationConfig()
    data = prepare_solver_comparison_data(results, metric)
    solvers = [solver for solver, _value in data]
    values = [value for _solver, value in data]
    metric_label = metric.replace("_", " ").title()

    if cfg.backend == "plotly":
        go = _import_plotly()
        fig = go.Figure(go.Bar(x=solvers, y=values))
        fig.update_layout(
            title=f"Solver Comparison ({metric_label})",
            xaxis_title="Solver",
            yaxis_title=metric_label,
        )
        return fig

    fig, ax = _new_matplotlib_axes(cfg)
    ax.bar(solvers, values, color=_palette_colors(cfg, len(solvers)))
    ax.set_xlabel("Solver")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Solver Comparison ({metric_label})")
    ax.grid(True, axis="y", alpha=0.3)
    return fig


# ============================================================================
# Internal helpers
# ============================================================================


def _validate_returns_frame(returns: Any) -> None:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise VisualizationError(
            "returns must be a non-empty DataFrame with one column per asset"
        )


def _new_matplotlib_axes(config: VisualizationConfig) -> Tuple[Any, Any]:
    import matplotlib.style
    from matplotlib.figure import Figure

    figsize = (float(config.figure_size[0]), float(config.figure_size[1]))
    with matplotlib.style.context(config.style):
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
    return fig, ax


def _palette_colors(config: VisualizationConfig, count: int) -> List[Any]:
    import matplotlib

    try:
        cmap = matplotlib.colormaps[config.color_palette]
    except KeyError as exc:
        raise VisualizationError(
            f"Unknown color palette '{config.color_palette}'"
        ) from exc
    if hasattr(cmap, "colors"):
        colors = list(cmap.colors)
        return [colors[index % len(colors)] for index in range(count)]
    return [cmap(index / max(count - 1, 1)) for index in range(count)]


def _import_plotly() -> Any:
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - plotly is a declared dependency
        raise VisualizationError(
            "backend='plotly' requires the plotly package to be installed"
        ) from exc
    return go


__all__ = [
    "VisualizationConfig",
    "normalize_qaoa_result",
    "prepare_composition_data",
    "prepare_risk_return_data",
    "prepare_probability_data",
    "prepare_top_solutions_data",
    "prepare_solver_comparison_data",
    "plot_portfolio_composition",
    "plot_risk_return_scatter",
    "plot_correlation_heatmap",
    "plot_efficient_frontier",
    "plot_qaoa_convergence",
    "plot_solution_probabilities",
    "plot_top_solutions",
    "render_qaoa_circuit_summary",
    "plot_solver_comparison",
]
