use thiserror::Error;

#[derive(Error, Debug)]
pub enum QaoaError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, QaoaError>;
