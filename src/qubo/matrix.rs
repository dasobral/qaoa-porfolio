use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::error::{QaoaError, Result};

const SYMMETRY_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QUBOMatrix {
    pub(crate) matrix: DMatrix<f64>,
    pub(crate) variable_labels: Vec<String>,
    pub(crate) offset: f64,
}

impl QUBOMatrix {
    pub fn new(num_variables: usize) -> Self {
        let labels = (0..num_variables)
            .map(|index| format!("x{index}"))
            .collect();
        Self {
            matrix: DMatrix::zeros(num_variables, num_variables),
            variable_labels: labels,
            offset: 0.0,
        }
    }

    pub fn with_labels(labels: Vec<String>) -> Self {
        let num_variables = labels.len();
        Self {
            matrix: DMatrix::zeros(num_variables, num_variables),
            variable_labels: labels,
            offset: 0.0,
        }
    }

    pub fn from_matrix(matrix: DMatrix<f64>, labels: Vec<String>) -> Result<Self> {
        if matrix.nrows() != matrix.ncols() {
            return Err(QaoaError::InvalidInput(
                "QUBO matrix must be square".to_string(),
            ));
        }
        if labels.len() != matrix.ncols() {
            return Err(QaoaError::DimensionMismatch {
                expected: matrix.ncols(),
                got: labels.len(),
            });
        }

        let qubo = Self {
            matrix,
            variable_labels: labels,
            offset: 0.0,
        };
        qubo.validate()?;
        Ok(qubo)
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) -> Result<()> {
        self.check_index(i)?;
        self.check_index(j)?;
        if !value.is_finite() {
            return Err(QaoaError::InvalidInput(
                "QUBO values must be finite".to_string(),
            ));
        }
        self.matrix[(i, j)] = value;
        self.matrix[(j, i)] = value;
        Ok(())
    }

    pub fn add(&mut self, i: usize, j: usize, value: f64) -> Result<()> {
        self.check_index(i)?;
        self.check_index(j)?;
        if !value.is_finite() {
            return Err(QaoaError::InvalidInput(
                "QUBO values must be finite".to_string(),
            ));
        }
        self.matrix[(i, j)] += value;
        if i != j {
            self.matrix[(j, i)] += value;
        }
        Ok(())
    }

    pub fn add_offset(&mut self, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(QaoaError::InvalidInput(
                "QUBO offset must be finite".to_string(),
            ));
        }
        self.offset += value;
        Ok(())
    }

    pub fn evaluate(&self, solution: &[bool]) -> Result<f64> {
        if solution.len() != self.num_variables() {
            return Err(QaoaError::DimensionMismatch {
                expected: self.num_variables(),
                got: solution.len(),
            });
        }

        let mut value = self.offset;
        for i in 0..self.num_variables() {
            if !solution[i] {
                continue;
            }
            value += self.matrix[(i, i)];
            for (j, selected) in solution
                .iter()
                .enumerate()
                .take(self.num_variables())
                .skip(i + 1)
            {
                if *selected {
                    value += self.matrix[(i, j)];
                }
            }
        }
        Ok(value)
    }

    pub fn num_variables(&self) -> usize {
        self.matrix.ncols()
    }

    pub fn as_matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    pub fn labels(&self) -> &[String] {
        &self.variable_labels
    }

    pub fn offset(&self) -> f64 {
        self.offset
    }

    pub fn to_vec(&self) -> Vec<Vec<f64>> {
        (0..self.matrix.nrows())
            .map(|row| {
                (0..self.matrix.ncols())
                    .map(|col| self.matrix[(row, col)])
                    .collect()
            })
            .collect()
    }

    pub fn validate(&self) -> Result<()> {
        if self.matrix.nrows() != self.matrix.ncols() {
            return Err(QaoaError::InvalidInput(
                "QUBO matrix must be square".to_string(),
            ));
        }
        if self.variable_labels.len() != self.matrix.ncols() {
            return Err(QaoaError::DimensionMismatch {
                expected: self.matrix.ncols(),
                got: self.variable_labels.len(),
            });
        }
        if self.matrix.iter().any(|value| !value.is_finite()) || !self.offset.is_finite() {
            return Err(QaoaError::InvalidInput(
                "QUBO matrix and offset must contain only finite values".to_string(),
            ));
        }
        for row in 0..self.matrix.nrows() {
            for col in (row + 1)..self.matrix.ncols() {
                if (self.matrix[(row, col)] - self.matrix[(col, row)]).abs() > SYMMETRY_TOLERANCE {
                    return Err(QaoaError::InvalidInput(
                        "QUBO matrix must be symmetric".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn check_index(&self, index: usize) -> Result<()> {
        if index >= self.num_variables() {
            return Err(QaoaError::DimensionMismatch {
                expected: self.num_variables(),
                got: index + 1,
            });
        }
        Ok(())
    }
}
