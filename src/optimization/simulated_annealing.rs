use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::{QaoaError, Result};
use crate::qubo::QUBOMatrix;

use super::brute_force::BruteForceSolver;
use super::{OptimizationResult, SolverMetrics};

#[derive(Debug, Clone)]
pub struct SimulatedAnnealing {
    initial_temperature: f64,
    cooling_rate: f64,
    max_iterations: usize,
    seed: Option<u64>,
}

impl SimulatedAnnealing {
    pub fn new() -> Self {
        Self {
            initial_temperature: 100.0,
            cooling_rate: 0.995,
            max_iterations: 10_000,
            seed: None,
        }
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.initial_temperature = temperature;
        self
    }

    pub fn with_cooling_rate(mut self, cooling_rate: f64) -> Self {
        self.cooling_rate = cooling_rate;
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn solve(&self, qubo: &QUBOMatrix) -> Result<OptimizationResult> {
        if qubo.num_variables() == 0 {
            return Err(QaoaError::InvalidInput(
                "simulated annealing requires at least one variable".to_string(),
            ));
        }
        if !self.initial_temperature.is_finite() || self.initial_temperature <= 0.0 {
            return Err(QaoaError::InvalidInput(
                "initial temperature must be finite and positive".to_string(),
            ));
        }
        if !self.cooling_rate.is_finite() || !(0.0..1.0).contains(&self.cooling_rate) {
            return Err(QaoaError::InvalidInput(
                "cooling rate must be finite and between 0.0 and 1.0".to_string(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(QaoaError::InvalidInput(
                "max iterations must be greater than zero".to_string(),
            ));
        }

        let start = Instant::now();
        let mut rng = self
            .seed
            .map(StdRng::seed_from_u64)
            .unwrap_or_else(StdRng::from_entropy);

        let mut current: Vec<bool> = (0..qubo.num_variables())
            .map(|_| rng.gen_bool(0.5))
            .collect();
        let mut current_value = qubo.evaluate(&current)?;
        let mut best = current.clone();
        let mut best_value = current_value;
        let mut best_found_at_iteration = 0;
        let mut temperature = self.initial_temperature;
        let mut convergence_history = Vec::with_capacity(self.max_iterations + 1);
        convergence_history.push(best_value);

        for iteration in 1..=self.max_iterations {
            let flip_index = rng.gen_range(0..qubo.num_variables());
            let mut candidate = current.clone();
            candidate[flip_index] = !candidate[flip_index];
            let candidate_value = qubo.evaluate(&candidate)?;
            let delta = candidate_value - current_value;

            let accept =
                delta <= 0.0 || rng.gen_range(0.0..1.0) < (-delta / temperature.max(1e-12)).exp();
            if accept {
                current = candidate;
                current_value = candidate_value;
            }

            if current_value < best_value {
                best = current.clone();
                best_value = current_value;
                best_found_at_iteration = iteration;
            }

            convergence_history.push(best_value);
            temperature *= self.cooling_rate;
        }

        let selected_assets = BruteForceSolver::selected_assets(qubo, &best);
        Ok(OptimizationResult::new(
            best,
            best_value,
            selected_assets,
            "simulated-annealing".to_string(),
            self.max_iterations,
            start.elapsed().as_millis() as u64,
            SolverMetrics::new(
                self.max_iterations,
                best_found_at_iteration,
                convergence_history,
            ),
        ))
    }
}

impl Default for SimulatedAnnealing {
    fn default() -> Self {
        Self::new()
    }
}
