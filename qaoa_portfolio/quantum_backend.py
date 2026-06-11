"""
PennyLane QAOA backend for QUBO portfolio optimization.

This module converts Phase 2 QUBO matrices into PennyLane Hamiltonians,
optimizes QAOA parameters, and decodes ranked portfolio selections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

from .exceptions import QuantumBackendError
from .params import QAOAParams

SUPPORTED_OPTIMIZERS = {"adam", "gradient_descent", "cobyla", "nelder_mead"}
SUPPORTED_BACKENDS = {"default.qubit", "lightning.qubit"}
SYMMETRY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class QAOAConfig:
    """Configuration for the PennyLane QAOA backend."""

    layers: int = QAOAParams.DEFAULT_LAYERS
    optimizer: str = "adam"
    max_iterations: int = QAOAParams.DEFAULT_MAX_ITERATIONS
    convergence_threshold: float = QAOAParams.DEFAULT_CONVERGENCE_THRESHOLD
    shots: Optional[int] = None
    seed: int = 42
    backend: str = "default.qubit"
    num_restarts: int = 3

    def __post_init__(self) -> None:
        if self.layers <= 0:
            raise QuantumBackendError("QAOA layers must be greater than zero")
        if self.max_iterations <= 0:
            raise QuantumBackendError("max_iterations must be greater than zero")
        if self.convergence_threshold <= 0:
            raise QuantumBackendError("convergence_threshold must be greater than zero")
        if self.shots is not None and self.shots <= 0:
            raise QuantumBackendError("shots must be None or greater than zero")
        if self.optimizer not in SUPPORTED_OPTIMIZERS:
            raise QuantumBackendError(
                f"Unsupported optimizer '{self.optimizer}'. "
                f"Expected one of {sorted(SUPPORTED_OPTIMIZERS)}"
            )
        if self.backend not in SUPPORTED_BACKENDS:
            raise QuantumBackendError(
                f"Unsupported backend '{self.backend}'. "
                f"Expected one of {sorted(SUPPORTED_BACKENDS)}"
            )
        if self.num_restarts <= 0:
            raise QuantumBackendError("num_restarts must be greater than zero")


@dataclass
class QAOAResult:
    """Result returned by a QAOA QUBO solve."""

    best_bitstring: str
    best_solution: List[bool]
    selected_indices: List[int]
    selected_assets: List[str]
    objective_value: float
    probabilities: Dict[str, float]
    top_solutions: List[Dict[str, Any]]
    optimal_parameters: Dict[str, List[float]]
    convergence_history: List[float]
    iterations: int
    elapsed_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the result."""

        return _json_safe(asdict(self))


def build_cost_hamiltonian(qubo: np.ndarray, offset: float = 0.0) -> qml.Hamiltonian:
    """Build a PennyLane cost Hamiltonian from an upper-triangle QUBO matrix."""

    matrix = _validate_qubo_matrix(qubo)
    num_wires = matrix.shape[0]

    coeffs: List[float] = [float(offset)]
    ops: List[Any] = [qml.Identity(0)]

    for i in range(num_wires):
        coeffs[0] += float(matrix[i, i]) / 2.0

    for i in range(num_wires):
        for j in range(i + 1, num_wires):
            coeffs[0] += float(matrix[i, j]) / 4.0

    for i in range(num_wires):
        coefficient = -float(matrix[i, i]) / 2.0
        for j in range(num_wires):
            if i == j:
                continue
            row, col = sorted((i, j))
            coefficient -= float(matrix[row, col]) / 4.0
        if abs(coefficient) > 0.0:
            coeffs.append(float(coefficient))
            ops.append(qml.PauliZ(i))

    for i in range(num_wires):
        for j in range(i + 1, num_wires):
            coefficient = float(matrix[i, j]) / 4.0
            if abs(coefficient) > 0.0:
                coeffs.append(coefficient)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    return qml.Hamiltonian(coeffs, ops)


def build_mixer_hamiltonian(num_wires: int) -> qml.Hamiltonian:
    """Build the standard X-mixer Hamiltonian."""

    if num_wires <= 0:
        raise QuantumBackendError("num_wires must be greater than zero")
    return qml.Hamiltonian(
        [1.0 for _ in range(num_wires)],
        [qml.PauliX(wire) for wire in range(num_wires)],
    )


def bitstring_to_solution(bitstring: str) -> List[bool]:
    """Convert a bitstring into a boolean selection mask."""

    _validate_bitstring(bitstring)
    return [bit == "1" for bit in bitstring]


def evaluate_qubo_bitstring(
    qubo: np.ndarray,
    bitstring: str,
    offset: float = 0.0,
) -> float:
    """Evaluate a bitstring with the Phase 2 upper-triangle QUBO convention."""

    matrix = _validate_qubo_matrix(qubo)
    _validate_bitstring(bitstring)
    if len(bitstring) != matrix.shape[0]:
        raise QuantumBackendError("Bitstring length must match QUBO dimension")

    bits = np.array([1.0 if bit == "1" else 0.0 for bit in bitstring], dtype=float)
    value = float(offset)
    for i in range(matrix.shape[0]):
        value += float(matrix[i, i]) * bits[i]
        for j in range(i + 1, matrix.shape[0]):
            value += float(matrix[i, j]) * bits[i] * bits[j]
    return float(value)


def decode_solution(
    bitstring: str, labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Decode a bitstring into indices, labels, and boolean solution values."""

    solution = bitstring_to_solution(bitstring)
    resolved_labels = _resolve_labels(len(solution), labels)
    selected_indices = [index for index, selected in enumerate(solution) if selected]
    selected_assets = [resolved_labels[index] for index in selected_indices]

    return {
        "bitstring": bitstring,
        "solution": solution,
        "selected_indices": selected_indices,
        "selected_assets": selected_assets,
    }


class QAOAQuantumBackend:
    """PennyLane QAOA solver for square symmetric QUBO matrices."""

    def __init__(self, config: Optional[QAOAConfig] = None):
        self.config = config or QAOAConfig()

    def solve(self, qubo: Any, labels: Optional[List[str]] = None) -> QAOAResult:
        """Solve a QUBO and return ranked decoded portfolio selections."""

        started = perf_counter()
        matrix, offset, resolved_labels, source = self._normalize_qubo_with_source(
            qubo,
            labels=labels,
        )
        num_wires = matrix.shape[0]
        cost_hamiltonian = build_cost_hamiltonian(matrix, offset)
        mixer_hamiltonian = build_mixer_hamiltonian(num_wires)

        optimization = self._optimize_parameters(
            cost_hamiltonian, mixer_hamiltonian, num_wires
        )
        probabilities = self._compute_probabilities(
            cost_hamiltonian,
            mixer_hamiltonian,
            num_wires,
            optimization["parameters"],
        )
        ranked_solutions = self._rank_solutions(
            matrix, offset, resolved_labels, probabilities
        )
        best = ranked_solutions[0]
        elapsed_ms = int((perf_counter() - started) * 1000)

        return QAOAResult(
            best_bitstring=best["bitstring"],
            best_solution=best["solution"],
            selected_indices=best["selected_indices"],
            selected_assets=best["selected_assets"],
            objective_value=best["objective_value"],
            probabilities=probabilities,
            top_solutions=ranked_solutions[:10],
            optimal_parameters={
                "gammas": _float_list(optimization["parameters"][: self.config.layers]),
                "betas": _float_list(optimization["parameters"][self.config.layers :]),
            },
            convergence_history=_float_list(optimization["history"]),
            iterations=int(optimization["iterations"]),
            elapsed_ms=elapsed_ms,
            metadata={
                "backend": self.config.backend,
                "optimizer": self.config.optimizer,
                "layers": self.config.layers,
                "shots": self.config.shots,
                "num_restarts": self.config.num_restarts,
                "num_variables": num_wires,
                "offset": float(offset),
                "source": source,
                "expected_cost": float(optimization["cost"]),
            },
        )

    def build_cost_hamiltonian(
        self,
        qubo: Any,
        offset: float = 0.0,
    ) -> qml.Hamiltonian:
        """Instance wrapper for cost Hamiltonian construction."""

        matrix, normalized_offset, _labels, _source = self._normalize_qubo_with_source(
            qubo
        )
        return build_cost_hamiltonian(matrix, normalized_offset + offset)

    def build_mixer_hamiltonian(self, num_wires: int) -> qml.Hamiltonian:
        """Instance wrapper for mixer Hamiltonian construction."""

        return build_mixer_hamiltonian(num_wires)

    def _normalize_qubo(
        self,
        qubo: Any,
        labels: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, float, List[str]]:
        matrix, offset, resolved_labels, _source = self._normalize_qubo_with_source(
            qubo,
            labels=labels,
        )
        return matrix, offset, resolved_labels

    def _normalize_qubo_with_source(
        self,
        qubo: Any,
        labels: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, float, List[str], str]:
        source = type(qubo).__name__
        offset = 0.0

        if hasattr(qubo, "to_numpy"):
            matrix = np.asarray(qubo.to_numpy(), dtype=float)
            offset = float(getattr(qubo, "offset", 0.0))
        else:
            matrix = np.asarray(qubo, dtype=float)
            source = "ndarray"

        matrix = _validate_qubo_matrix(matrix)
        resolved_labels = _resolve_labels(matrix.shape[0], labels)
        return matrix, offset, resolved_labels, source

    def _optimize_parameters(
        self,
        cost_hamiltonian: qml.Hamiltonian,
        mixer_hamiltonian: qml.Hamiltonian,
        num_wires: int,
    ) -> Dict[str, Any]:
        best: Optional[Dict[str, Any]] = None
        rng = np.random.default_rng(self.config.seed)

        for _restart in range(self.config.num_restarts):
            initial = self._initial_parameters(rng)
            cost_circuit = self._make_cost_qnode(
                cost_hamiltonian, mixer_hamiltonian, num_wires
            )
            if self.config.optimizer in {"adam", "gradient_descent"}:
                candidate = self._run_pennylane_optimizer(cost_circuit, initial)
            else:
                candidate = self._run_scipy_optimizer(cost_circuit, initial)

            if best is None or candidate["cost"] < best["cost"]:
                best = candidate

        if best is None:
            raise QuantumBackendError("QAOA optimization did not produce a candidate")
        return best

    def _initial_parameters(self, rng: np.random.Generator) -> np.ndarray:
        gammas = rng.uniform(0.0, np.pi, self.config.layers)
        betas = rng.uniform(0.0, np.pi / 2.0, self.config.layers)
        return np.concatenate([gammas, betas]).astype(float)

    def _run_pennylane_optimizer(
        self, cost_circuit: Any, initial: np.ndarray
    ) -> Dict[str, Any]:
        params = pnp.array(initial, requires_grad=True)
        optimizer: Any
        if self.config.optimizer == "adam":
            optimizer = qml.AdamOptimizer()
        else:
            optimizer = qml.GradientDescentOptimizer()

        history: List[float] = []
        best_cost = float("inf")
        best_params = np.asarray(initial, dtype=float)
        stale_iterations = 0
        previous_cost: Optional[float] = None

        for _iteration in range(1, self.config.max_iterations + 1):
            params, cost = optimizer.step_and_cost(cost_circuit, params)
            cost_value = float(cost)
            if cost_value < best_cost:
                best_cost = cost_value
                best_params = np.asarray(params, dtype=float)
            history.append(best_cost)

            if previous_cost is not None:
                improvement = abs(previous_cost - cost_value)
                if improvement < self.config.convergence_threshold:
                    stale_iterations += 1
                else:
                    stale_iterations = 0
                if stale_iterations >= 5:
                    break
            previous_cost = cost_value

        return {
            "parameters": best_params,
            "cost": best_cost,
            "history": history,
            "iterations": len(history),
        }

    def _run_scipy_optimizer(
        self, cost_circuit: Any, initial: np.ndarray
    ) -> Dict[str, Any]:
        history: List[float] = []
        best_cost = float("inf")
        best_params = np.asarray(initial, dtype=float)

        def objective(raw_params: np.ndarray) -> float:
            nonlocal best_cost, best_params
            params = pnp.array(raw_params, requires_grad=False)
            cost_value = float(cost_circuit(params))
            if cost_value < best_cost:
                best_cost = cost_value
                best_params = np.asarray(raw_params, dtype=float)
            return cost_value

        def callback(raw_params: np.ndarray) -> None:
            objective(raw_params)
            history.append(best_cost)

        method = "COBYLA" if self.config.optimizer == "cobyla" else "Nelder-Mead"
        result = minimize(
            objective,
            np.asarray(initial, dtype=float),
            method=method,
            callback=callback,
            options={"maxiter": self.config.max_iterations},
        )
        final_cost = objective(np.asarray(result.x, dtype=float))
        if not history:
            history.append(min(best_cost, final_cost))

        return {
            "parameters": best_params,
            "cost": min(best_cost, final_cost),
            "history": history,
            "iterations": max(1, len(history)),
        }

    def _compute_probabilities(
        self,
        cost_hamiltonian: qml.Hamiltonian,
        mixer_hamiltonian: qml.Hamiltonian,
        num_wires: int,
        parameters: np.ndarray,
    ) -> Dict[str, float]:
        probs_circuit = self._make_probabilities_qnode(
            cost_hamiltonian,
            mixer_hamiltonian,
            num_wires,
        )
        raw_probabilities = np.asarray(
            probs_circuit(pnp.array(parameters, requires_grad=False)),
            dtype=float,
        )
        raw_probabilities = np.maximum(raw_probabilities, 0.0)
        total = float(np.sum(raw_probabilities))
        if total > 0.0:
            raw_probabilities = raw_probabilities / total

        return {
            format(index, f"0{num_wires}b"): float(probability)
            for index, probability in enumerate(raw_probabilities)
        }

    def _rank_solutions(
        self,
        qubo: np.ndarray,
        offset: float,
        labels: List[str],
        probabilities: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for bitstring, probability in probabilities.items():
            decoded = decode_solution(bitstring, labels)
            decoded["objective_value"] = evaluate_qubo_bitstring(
                qubo, bitstring, offset
            )
            decoded["probability"] = float(probability)
            ranked.append(decoded)

        ranked.sort(key=lambda entry: (entry["objective_value"], entry["bitstring"]))
        return ranked

    def _make_cost_qnode(
        self,
        cost_hamiltonian: qml.Hamiltonian,
        mixer_hamiltonian: qml.Hamiltonian,
        num_wires: int,
    ) -> Any:
        device = self._make_device(num_wires)

        @qml.qnode(device, interface="autograd")
        def circuit(parameters: Any) -> Any:
            self._apply_qaoa_ansatz(
                parameters, cost_hamiltonian, mixer_hamiltonian, num_wires
            )
            return qml.expval(cost_hamiltonian)

        return circuit

    def _make_probabilities_qnode(
        self,
        cost_hamiltonian: qml.Hamiltonian,
        mixer_hamiltonian: qml.Hamiltonian,
        num_wires: int,
    ) -> Any:
        device = self._make_device(num_wires)

        @qml.qnode(device, interface="autograd")
        def circuit(parameters: Any) -> Any:
            self._apply_qaoa_ansatz(
                parameters, cost_hamiltonian, mixer_hamiltonian, num_wires
            )
            return qml.probs(wires=range(num_wires))

        return circuit

    def _apply_qaoa_ansatz(
        self,
        parameters: Any,
        cost_hamiltonian: qml.Hamiltonian,
        mixer_hamiltonian: qml.Hamiltonian,
        num_wires: int,
    ) -> None:
        gammas = parameters[: self.config.layers]
        betas = parameters[self.config.layers :]

        for wire in range(num_wires):
            qml.Hadamard(wires=wire)

        for layer in range(self.config.layers):
            qml.qaoa.cost_layer(gammas[layer], cost_hamiltonian)
            qml.qaoa.mixer_layer(betas[layer], mixer_hamiltonian)

    def _make_device(self, num_wires: int) -> Any:
        try:
            return qml.device(
                self.config.backend,
                wires=num_wires,
                shots=self.config.shots,
                seed=self.config.seed,
            )
        except TypeError:
            return qml.device(
                self.config.backend, wires=num_wires, shots=self.config.shots
            )
        except (
            Exception
        ) as exc:  # pragma: no cover - depends on optional backend installation
            raise QuantumBackendError(
                f"Unable to create PennyLane backend '{self.config.backend}': {exc}"
            ) from exc


def solve_qubo_qaoa(
    qubo: Any,
    labels: Optional[List[str]] = None,
    config: Optional[QAOAConfig] = None,
) -> QAOAResult:
    """Convenience function for solving a QUBO with the PennyLane QAOA backend."""

    return QAOAQuantumBackend(config).solve(qubo, labels=labels)


def _validate_qubo_matrix(qubo: Any) -> np.ndarray:
    matrix = np.asarray(qubo, dtype=float)
    if matrix.ndim != 2:
        raise QuantumBackendError("QUBO matrix must be two-dimensional")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise QuantumBackendError("QUBO matrix must be non-empty")
    if matrix.shape[0] != matrix.shape[1]:
        raise QuantumBackendError("QUBO matrix must be square")
    if not np.all(np.isfinite(matrix)):
        raise QuantumBackendError("QUBO matrix must contain only finite values")
    if not np.allclose(matrix, matrix.T, atol=SYMMETRY_TOLERANCE, rtol=0.0):
        raise QuantumBackendError("QUBO matrix must be symmetric")
    return matrix.astype(float, copy=True)


def _resolve_labels(num_variables: int, labels: Optional[List[str]]) -> List[str]:
    if labels is None:
        return [f"x{index}" for index in range(num_variables)]
    if len(labels) != num_variables:
        raise QuantumBackendError("Number of labels must match QUBO dimension")
    return list(labels)


def _validate_bitstring(bitstring: str) -> None:
    if not bitstring:
        raise QuantumBackendError("Bitstring must be non-empty")
    if any(bit not in {"0", "1"} for bit in bitstring):
        raise QuantumBackendError("Bitstring must contain only 0 and 1")


def _float_list(values: Any) -> List[float]:
    return [float(value) for value in values]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "QAOAConfig",
    "QAOAResult",
    "QAOAQuantumBackend",
    "bitstring_to_solution",
    "build_cost_hamiltonian",
    "build_mixer_hamiltonian",
    "decode_solution",
    "evaluate_qubo_bitstring",
    "solve_qubo_qaoa",
]
