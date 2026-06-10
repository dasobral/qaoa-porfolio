use nalgebra::{DMatrix, DVector};

use crate::error::{QaoaError, Result};
use crate::portfolio::ReturnSeries;

use super::{PenaltyBuilder, QUBOMatrix};

#[derive(Debug, Clone)]
pub struct QUBOFormulation {
    risk_aversion: f64,
    target_assets: usize,
    penalty_budget: Option<f64>,
    penalty_position: Option<f64>,
    penalty_diversity: Option<f64>,
}

impl QUBOFormulation {
    pub fn new(risk_aversion: f64, target_assets: usize) -> Result<Self> {
        if !(0.0..=1.0).contains(&risk_aversion) || !risk_aversion.is_finite() {
            return Err(QaoaError::InvalidInput(
                "risk aversion must be finite and between 0.0 and 1.0".to_string(),
            ));
        }
        if target_assets == 0 {
            return Err(QaoaError::InvalidInput(
                "target asset count must be greater than zero".to_string(),
            ));
        }

        Ok(Self {
            risk_aversion,
            target_assets,
            penalty_budget: None,
            penalty_position: None,
            penalty_diversity: None,
        })
    }

    pub fn with_budget_penalty(mut self, penalty: f64) -> Self {
        self.penalty_budget = Some(penalty);
        self
    }

    pub fn with_position_penalty(mut self, penalty: f64) -> Self {
        self.penalty_position = Some(penalty);
        self
    }

    pub fn with_diversity_penalty(mut self, penalty: f64) -> Self {
        self.penalty_diversity = Some(penalty);
        self
    }

    pub fn build(&self, returns: &ReturnSeries) -> Result<QUBOMatrix> {
        let covariance = returns.covariance_matrix()?;
        let expected_returns = returns.mean_returns();
        self.build_from_params(&covariance, &expected_returns, returns.symbols())
    }

    pub fn build_from_params(
        &self,
        covariance: &DMatrix<f64>,
        expected_returns: &DVector<f64>,
        symbols: &[String],
    ) -> Result<QUBOMatrix> {
        let n = symbols.len();
        if n == 0 {
            return Err(QaoaError::InvalidInput(
                "at least one symbol is required".to_string(),
            ));
        }
        if self.target_assets > n {
            return Err(QaoaError::InvalidInput(format!(
                "target asset count {} exceeds universe size {n}",
                self.target_assets
            )));
        }
        if covariance.nrows() != n || covariance.ncols() != n {
            return Err(QaoaError::InvalidInput(
                "covariance matrix dimensions must match symbols".to_string(),
            ));
        }
        if expected_returns.len() != n {
            return Err(QaoaError::DimensionMismatch {
                expected: n,
                got: expected_returns.len(),
            });
        }
        if covariance.iter().any(|value| !value.is_finite())
            || expected_returns.iter().any(|value| !value.is_finite())
        {
            return Err(QaoaError::InvalidInput(
                "covariance and return values must be finite".to_string(),
            ));
        }

        let mut qubo = QUBOMatrix::with_labels(symbols.to_vec());
        let mut max_objective = 0.0_f64;

        for i in 0..n {
            let diagonal = self.risk_aversion * covariance[(i, i)]
                - (1.0 - self.risk_aversion) * expected_returns[i];
            max_objective = max_objective.max(diagonal.abs());
            qubo.add(i, i, diagonal)?;
            for j in (i + 1)..n {
                let off_diagonal = self.risk_aversion * covariance[(i, j)];
                max_objective = max_objective.max(off_diagonal.abs());
                qubo.add(i, j, off_diagonal)?;
            }
        }

        let budget_penalty = match self.penalty_budget {
            Some(value) if value.is_finite() && value > 0.0 => value,
            Some(_) => {
                return Err(QaoaError::InvalidInput(
                    "budget penalty must be finite and positive".to_string(),
                ));
            }
            None if max_objective > 0.0 => max_objective * 2.0,
            None => 1.0,
        };

        PenaltyBuilder::budget(&mut qubo, self.target_assets, budget_penalty);

        if let Some(weight) = self.penalty_position
            && weight.is_finite()
            && weight > 0.0
        {
            let all_assets: Vec<usize> = (0..n).collect();
            PenaltyBuilder::position_limit(&mut qubo, &all_assets, self.target_assets, weight);
        }
        if let Some(weight) = self.penalty_diversity
            && weight.is_finite()
            && weight > 0.0
        {
            let singleton_classes: Vec<Vec<usize>> = (0..n).map(|index| vec![index]).collect();
            PenaltyBuilder::diversity(
                &mut qubo,
                &singleton_classes,
                self.target_assets.min(n),
                weight,
            );
        }

        qubo.validate()?;
        Ok(qubo)
    }
}
