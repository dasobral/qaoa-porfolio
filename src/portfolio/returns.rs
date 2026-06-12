use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::error::{QaoaError, Result};

const TRADING_DAYS_PER_YEAR: f64 = 252.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnSeries {
    symbols: Vec<String>,
    returns: DMatrix<f64>,
}

impl ReturnSeries {
    pub fn from_prices(symbols: Vec<String>, prices: DMatrix<f64>) -> Result<Self> {
        Self::validate_symbol_count(&symbols, prices.ncols())?;
        if prices.nrows() < 2 {
            return Err(QaoaError::InvalidInput(
                "price matrix must contain at least two periods".to_string(),
            ));
        }
        if prices
            .iter()
            .any(|price| !price.is_finite() || *price <= 0.0)
        {
            return Err(QaoaError::InvalidInput(
                "prices must be finite and strictly positive".to_string(),
            ));
        }

        let rows = prices.nrows() - 1;
        let cols = prices.ncols();
        let mut values = Vec::with_capacity(rows * cols);
        for row in 1..prices.nrows() {
            for col in 0..cols {
                values.push((prices[(row, col)] / prices[(row - 1, col)]).ln());
            }
        }

        Self::from_returns(symbols, DMatrix::from_row_slice(rows, cols, &values))
    }

    pub fn from_returns(symbols: Vec<String>, returns: DMatrix<f64>) -> Result<Self> {
        Self::validate_symbol_count(&symbols, returns.ncols())?;
        if returns.nrows() == 0 {
            return Err(QaoaError::InvalidInput(
                "returns matrix must contain at least one period".to_string(),
            ));
        }
        if returns.iter().any(|value| !value.is_finite()) {
            return Err(QaoaError::InvalidInput(
                "returns must contain only finite values".to_string(),
            ));
        }

        Ok(Self { symbols, returns })
    }

    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }

    pub fn returns(&self) -> &DMatrix<f64> {
        &self.returns
    }

    pub fn num_periods(&self) -> usize {
        self.returns.nrows()
    }

    pub fn num_assets(&self) -> usize {
        self.returns.ncols()
    }

    pub fn mean_returns(&self) -> DVector<f64> {
        self.daily_mean_returns() * TRADING_DAYS_PER_YEAR
    }

    pub fn covariance_matrix(&self) -> Result<DMatrix<f64>> {
        if self.num_periods() < 2 {
            return Err(QaoaError::NumericalError(
                "at least two return periods are required for covariance".to_string(),
            ));
        }

        let means = self.daily_mean_returns();
        let n = self.num_assets();
        let mut covariance = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in i..n {
                let mut total = 0.0;
                for row in 0..self.num_periods() {
                    total +=
                        (self.returns[(row, i)] - means[i]) * (self.returns[(row, j)] - means[j]);
                }
                let value = total / (self.num_periods() as f64 - 1.0) * TRADING_DAYS_PER_YEAR;
                covariance[(i, j)] = value;
                covariance[(j, i)] = value;
            }
        }

        Ok(covariance)
    }

    pub fn correlation_matrix(&self) -> Result<DMatrix<f64>> {
        let covariance = self.covariance_matrix()?;
        let n = covariance.nrows();
        let mut correlation = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in i..n {
                let denom = (covariance[(i, i)] * covariance[(j, j)]).sqrt();
                let value = if i == j {
                    1.0
                } else if denom.abs() <= f64::EPSILON {
                    0.0
                } else {
                    covariance[(i, j)] / denom
                };
                correlation[(i, j)] = value.clamp(-1.0, 1.0);
                correlation[(j, i)] = value.clamp(-1.0, 1.0);
            }
        }

        Ok(correlation)
    }

    fn daily_mean_returns(&self) -> DVector<f64> {
        let mut means = DVector::zeros(self.num_assets());
        for col in 0..self.num_assets() {
            let mut total = 0.0;
            for row in 0..self.num_periods() {
                total += self.returns[(row, col)];
            }
            means[col] = total / self.num_periods() as f64;
        }
        means
    }

    fn validate_symbol_count(symbols: &[String], matrix_cols: usize) -> Result<()> {
        if symbols.is_empty() {
            return Err(QaoaError::InvalidInput(
                "at least one symbol is required".to_string(),
            ));
        }
        if symbols.len() != matrix_cols {
            return Err(QaoaError::DimensionMismatch {
                expected: symbols.len(),
                got: matrix_cols,
            });
        }
        if symbols.iter().any(|symbol| symbol.trim().is_empty()) {
            return Err(QaoaError::InvalidInput(
                "symbols must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}
