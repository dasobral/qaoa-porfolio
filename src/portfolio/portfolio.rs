use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::error::{QaoaError, Result};

use super::Asset;

const WEIGHT_TOLERANCE: f64 = 1e-6;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    assets: Vec<Asset>,
    weights: Option<Vec<f64>>,
}

impl Portfolio {
    pub fn new(assets: Vec<Asset>) -> Result<Self> {
        let portfolio = Self {
            assets,
            weights: None,
        };
        portfolio.validate()?;
        Ok(portfolio)
    }

    pub fn with_weights(mut self, weights: Vec<f64>) -> Result<Self> {
        Self::validate_weights(self.assets.len(), &weights)?;
        self.weights = Some(weights);
        Ok(self)
    }

    pub fn num_assets(&self) -> usize {
        self.assets.len()
    }

    pub fn assets(&self) -> &[Asset] {
        &self.assets
    }

    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    pub fn symbols(&self) -> Vec<&str> {
        self.assets.iter().map(Asset::symbol).collect()
    }

    pub fn validate(&self) -> Result<()> {
        if self.assets.is_empty() {
            return Err(QaoaError::InvalidInput(
                "portfolio must contain at least one asset".to_string(),
            ));
        }

        let mut symbols = HashSet::with_capacity(self.assets.len());
        for asset in &self.assets {
            asset.validate()?;
            if !symbols.insert(asset.symbol().to_string()) {
                return Err(QaoaError::InvalidInput(format!(
                    "duplicate asset symbol '{}'",
                    asset.symbol()
                )));
            }
        }

        if let Some(weights) = &self.weights {
            Self::validate_weights(self.assets.len(), weights)?;
        }

        Ok(())
    }

    fn validate_weights(expected_len: usize, weights: &[f64]) -> Result<()> {
        if weights.len() != expected_len {
            return Err(QaoaError::DimensionMismatch {
                expected: expected_len,
                got: weights.len(),
            });
        }

        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(QaoaError::InvalidInput(
                "weights must be finite and non-negative".to_string(),
            ));
        }

        let total: f64 = weights.iter().sum();
        if (total - 1.0).abs() > WEIGHT_TOLERANCE {
            return Err(QaoaError::InvalidInput(format!(
                "weights must sum to 1.0, got {total}"
            )));
        }

        Ok(())
    }
}
