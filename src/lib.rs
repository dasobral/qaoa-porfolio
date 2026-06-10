#![forbid(unsafe_code)]

pub mod error;
pub mod optimization;
pub mod portfolio;
pub mod qubo;

#[cfg(feature = "python-bindings")]
pub mod python;

pub use error::{QaoaError, Result};
pub use optimization::{
    BruteForceSolver, ContinuousResult, MarkowitzSolver, OptimizationResult, SimulatedAnnealing,
    SolverMetrics,
};
pub use portfolio::{Asset, AssetClass, Portfolio, ReturnSeries};
pub use qubo::{PenaltyBuilder, QUBOFormulation, QUBOMatrix};
