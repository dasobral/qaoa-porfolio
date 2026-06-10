use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{QaoaError, Result};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AssetClass {
    Stock,
    Crypto,
    Bond,
    Commodity,
    Etf,
}

impl FromStr for AssetClass {
    type Err = QaoaError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "stock" | "stocks" | "equity" => Ok(Self::Stock),
            "crypto" | "cryptocurrency" => Ok(Self::Crypto),
            "bond" | "bonds" => Ok(Self::Bond),
            "commodity" | "commodities" => Ok(Self::Commodity),
            "etf" | "fund" => Ok(Self::Etf),
            other => Err(QaoaError::InvalidInput(format!(
                "unknown asset class '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Asset {
    symbol: String,
    name: Option<String>,
    asset_class: AssetClass,
    expected_return: f64,
    volatility: f64,
}

impl Asset {
    pub fn new(symbol: impl Into<String>, asset_class: AssetClass) -> Self {
        Self {
            symbol: symbol.into().trim().to_ascii_uppercase(),
            name: None,
            asset_class,
            expected_return: 0.0,
            volatility: 0.0,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_return(mut self, expected_return: f64) -> Self {
        self.expected_return = expected_return;
        self
    }

    pub fn with_volatility(mut self, volatility: f64) -> Self {
        self.volatility = volatility;
        self
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn asset_class(&self) -> AssetClass {
        self.asset_class
    }

    pub fn expected_return(&self) -> f64 {
        self.expected_return
    }

    pub fn volatility(&self) -> f64 {
        self.volatility
    }

    pub fn validate(&self) -> Result<()> {
        if self.symbol.trim().is_empty() {
            return Err(QaoaError::InvalidInput(
                "asset symbol must not be empty".to_string(),
            ));
        }
        if !self.expected_return.is_finite() {
            return Err(QaoaError::InvalidInput(
                "expected return must be finite".to_string(),
            ));
        }
        if !self.volatility.is_finite() || self.volatility < 0.0 {
            return Err(QaoaError::InvalidInput(
                "volatility must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}
