use serde::{Deserialize, Serialize};

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverMetrics {
    solutions_evaluated: usize,
    best_found_at_iteration: usize,
    convergence_history: Vec<f64>,
}

impl SolverMetrics {
    pub fn new(
        solutions_evaluated: usize,
        best_found_at_iteration: usize,
        convergence_history: Vec<f64>,
    ) -> Self {
        Self {
            solutions_evaluated,
            best_found_at_iteration,
            convergence_history,
        }
    }

    pub fn solutions_evaluated(&self) -> usize {
        self.solutions_evaluated
    }

    pub fn best_found_at_iteration(&self) -> usize {
        self.best_found_at_iteration
    }

    pub fn convergence_history(&self) -> &[f64] {
        &self.convergence_history
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    solution: Vec<bool>,
    objective_value: f64,
    selected_assets: Vec<String>,
    solver_name: String,
    iterations: usize,
    elapsed_ms: u64,
    metadata: SolverMetrics,
}

impl OptimizationResult {
    pub fn new(
        solution: Vec<bool>,
        objective_value: f64,
        selected_assets: Vec<String>,
        solver_name: String,
        iterations: usize,
        elapsed_ms: u64,
        metadata: SolverMetrics,
    ) -> Self {
        Self {
            solution,
            objective_value,
            selected_assets,
            solver_name,
            iterations,
            elapsed_ms,
            metadata,
        }
    }

    pub fn solution(&self) -> &[bool] {
        &self.solution
    }

    pub fn objective_value(&self) -> f64 {
        self.objective_value
    }

    pub fn selected_symbols(&self) -> &[String] {
        &self.selected_assets
    }

    pub fn solver_name(&self) -> &str {
        &self.solver_name
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.elapsed_ms
    }

    pub fn metadata(&self) -> &SolverMetrics {
        &self.metadata
    }

    pub fn is_feasible(&self, target_assets: usize) -> bool {
        self.solution.iter().filter(|selected| **selected).count() == target_assets
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }
}
