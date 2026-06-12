use std::time::Instant;

use rayon::prelude::*;

use crate::error::{QaoaError, Result};
use crate::qubo::QUBOMatrix;

use super::{OptimizationResult, SolverMetrics};

pub struct BruteForceSolver;

impl BruteForceSolver {
    pub fn solve(qubo: &QUBOMatrix) -> Result<OptimizationResult> {
        Self::solve_internal(qubo, None)
    }

    pub fn solve_constrained(qubo: &QUBOMatrix, target_k: usize) -> Result<OptimizationResult> {
        Self::solve_internal(qubo, Some(target_k))
    }

    fn solve_internal(qubo: &QUBOMatrix, target_k: Option<usize>) -> Result<OptimizationResult> {
        let n = qubo.num_variables();
        if n > 20 {
            return Err(QaoaError::OptimizationFailed(
                "brute force solver supports at most 20 variables".to_string(),
            ));
        }
        if let Some(k) = target_k
            && k > n
        {
            return Err(QaoaError::InvalidInput(format!(
                "target asset count {k} exceeds variable count {n}"
            )));
        }

        let start = Instant::now();
        let total_masks = 1_u64 << n;

        let candidates: Vec<(u64, f64)> = if n > 12 {
            (0..total_masks)
                .into_par_iter()
                .filter(|mask| Self::matches_constraint(*mask, target_k))
                .map(|mask| {
                    let solution = Self::mask_to_solution(mask, n);
                    let objective = qubo
                        .evaluate(&solution)
                        .expect("generated solution length matches QUBO dimensions");
                    (mask, objective)
                })
                .collect()
        } else {
            (0..total_masks)
                .filter(|mask| Self::matches_constraint(*mask, target_k))
                .map(|mask| {
                    let solution = Self::mask_to_solution(mask, n);
                    let objective = qubo
                        .evaluate(&solution)
                        .expect("generated solution length matches QUBO dimensions");
                    (mask, objective)
                })
                .collect()
        };

        if candidates.is_empty() {
            return Err(QaoaError::OptimizationFailed(
                "no feasible brute force candidates were generated".to_string(),
            ));
        }

        let mut best_mask = candidates[0].0;
        let mut best_value = candidates[0].1;
        let mut best_found_at_iteration = 0;
        // Record only improvements: a per-candidate history would carry up to
        // 2^20 entries across the Python bridge for zero analytical value.
        let mut convergence_history = vec![best_value];

        for (iteration, (mask, objective)) in candidates.iter().copied().enumerate().skip(1) {
            if objective < best_value {
                best_mask = mask;
                best_value = objective;
                best_found_at_iteration = iteration;
                convergence_history.push(best_value);
            }
        }

        let solution = Self::mask_to_solution(best_mask, n);
        let selected_assets = Self::selected_assets(qubo, &solution);
        let elapsed_ms = start.elapsed().as_millis() as u64;
        let evaluated = candidates.len();

        Ok(OptimizationResult::new(
            solution,
            best_value,
            selected_assets,
            if target_k.is_some() {
                "brute-force-constrained".to_string()
            } else {
                "brute-force".to_string()
            },
            evaluated,
            elapsed_ms,
            SolverMetrics::new(evaluated, best_found_at_iteration, convergence_history),
        ))
    }

    pub(crate) fn mask_to_solution(mask: u64, n: usize) -> Vec<bool> {
        (0..n).map(|index| (mask & (1_u64 << index)) != 0).collect()
    }

    pub(crate) fn selected_assets(qubo: &QUBOMatrix, solution: &[bool]) -> Vec<String> {
        solution
            .iter()
            .enumerate()
            .filter_map(|(index, selected)| {
                if *selected {
                    Some(qubo.labels()[index].clone())
                } else {
                    None
                }
            })
            .collect()
    }

    fn matches_constraint(mask: u64, target_k: Option<usize>) -> bool {
        target_k
            .map(|k| mask.count_ones() as usize == k)
            .unwrap_or(true)
    }
}
