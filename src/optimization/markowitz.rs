use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::error::{QaoaError, Result};
use crate::portfolio::ReturnSeries;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousResult {
    weights: Vec<f64>,
    expected_return: f64,
    volatility: f64,
    sharpe_ratio: f64,
    symbols: Vec<String>,
}

impl ContinuousResult {
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn expected_return(&self) -> f64 {
        self.expected_return
    }

    pub fn volatility(&self) -> f64 {
        self.volatility
    }

    pub fn sharpe_ratio(&self) -> f64 {
        self.sharpe_ratio
    }

    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }
}

#[derive(Debug, Clone)]
pub struct MarkowitzSolver {
    risk_free_rate: f64,
    target_return: Option<f64>,
}

impl MarkowitzSolver {
    pub fn new() -> Self {
        Self {
            risk_free_rate: 0.02,
            target_return: None,
        }
    }

    pub fn with_risk_free_rate(mut self, rate: f64) -> Self {
        self.risk_free_rate = rate;
        self
    }

    pub fn with_target_return(mut self, target: f64) -> Self {
        self.target_return = Some(target);
        self
    }

    pub fn solve(&self, returns: &ReturnSeries) -> Result<ContinuousResult> {
        if let Some(target_return) = self.target_return {
            self.target_return_portfolio(returns, target_return)
        } else {
            self.max_sharpe(returns)
        }
    }

    pub fn min_variance(&self, returns: &ReturnSeries) -> Result<ContinuousResult> {
        let covariance = returns.covariance_matrix()?;
        let inverse = invert_or_pseudo_inverse(&covariance)?;
        let ones = DVector::from_element(returns.num_assets(), 1.0);
        let numerator = &inverse * &ones;
        let denominator = ones.dot(&numerator);
        if denominator.abs() <= f64::EPSILON || !denominator.is_finite() {
            return Err(QaoaError::NumericalError(
                "minimum variance normalization denominator is zero".to_string(),
            ));
        }
        let weights = numerator / denominator;
        self.result_from_weights(returns, weights)
    }

    fn target_return_portfolio(
        &self,
        returns: &ReturnSeries,
        target_return: f64,
    ) -> Result<ContinuousResult> {
        if !target_return.is_finite() {
            return Err(QaoaError::InvalidInput(
                "target return must be finite".to_string(),
            ));
        }

        let covariance = returns.covariance_matrix()?;
        let inverse = invert_or_pseudo_inverse(&covariance)?;
        let ones = DVector::from_element(returns.num_assets(), 1.0);
        let expected_returns = returns.mean_returns();

        let inv_ones = &inverse * &ones;
        let inv_mu = &inverse * &expected_returns;
        let a = ones.dot(&inv_ones);
        let b = ones.dot(&inv_mu);
        let c = expected_returns.dot(&inv_mu);
        let determinant = a * c - b * b;

        if determinant.abs() <= 1e-12 || !determinant.is_finite() {
            return Err(QaoaError::NumericalError(
                "target-return Markowitz system is singular".to_string(),
            ));
        }

        let lambda = (c - b * target_return) / determinant;
        let gamma = (a * target_return - b) / determinant;
        let weights = inv_ones * lambda + inv_mu * gamma;
        self.result_from_weights(returns, weights)
    }

    pub fn max_sharpe(&self, returns: &ReturnSeries) -> Result<ContinuousResult> {
        let covariance = returns.covariance_matrix()?;
        let inverse = invert_or_pseudo_inverse(&covariance)?;
        let expected_returns = returns.mean_returns();
        let excess_returns = expected_returns.map(|value| value - self.risk_free_rate);
        let raw_weights = inverse * excess_returns;
        let total: f64 = raw_weights.iter().sum();
        if total.abs() <= f64::EPSILON || !total.is_finite() {
            return self.min_variance(returns);
        }
        self.result_from_weights(returns, raw_weights / total)
    }

    fn result_from_weights(
        &self,
        returns: &ReturnSeries,
        weights: DVector<f64>,
    ) -> Result<ContinuousResult> {
        let covariance = returns.covariance_matrix()?;
        let expected_returns = returns.mean_returns();
        let expected_return = weights.dot(&expected_returns);
        let variance = weights.dot(&(covariance * &weights));
        let volatility = variance.max(0.0).sqrt();
        let sharpe_ratio = if volatility <= f64::EPSILON {
            0.0
        } else {
            (expected_return - self.risk_free_rate) / volatility
        };

        Ok(ContinuousResult {
            weights: weights.iter().copied().collect(),
            expected_return,
            volatility,
            sharpe_ratio,
            symbols: returns.symbols().to_vec(),
        })
    }
}

impl Default for MarkowitzSolver {
    fn default() -> Self {
        Self::new()
    }
}

fn invert_or_pseudo_inverse(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    if let Some(inverse) = matrix.clone().try_inverse() {
        return Ok(inverse);
    }

    let svd = matrix.clone().svd(true, true);
    let u = svd.u.ok_or_else(|| {
        QaoaError::NumericalError("SVD did not return left singular vectors".to_string())
    })?;
    let v_t = svd.v_t.ok_or_else(|| {
        QaoaError::NumericalError("SVD did not return right singular vectors".to_string())
    })?;

    let tolerance = 1e-10;
    let mut sigma_inv = DMatrix::zeros(matrix.ncols(), matrix.nrows());
    for (index, singular_value) in svd.singular_values.iter().copied().enumerate() {
        if singular_value > tolerance {
            sigma_inv[(index, index)] = 1.0 / singular_value;
        }
    }

    Ok(v_t.transpose() * sigma_inv * u.transpose())
}
