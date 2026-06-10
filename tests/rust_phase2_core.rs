use nalgebra::{DMatrix, DVector};
use qaoa_portfolio::optimization::{
    BruteForceSolver, MarkowitzSolver, OptimizationResult, SimulatedAnnealing, SolverMetrics,
};
use qaoa_portfolio::portfolio::{Asset, AssetClass, Portfolio, ReturnSeries};
use qaoa_portfolio::qubo::{PenaltyBuilder, QUBOFormulation, QUBOMatrix};

fn assert_close(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "expected {left} to be within {tolerance} of {right}"
    );
}

#[test]
fn asset_and_portfolio_validation_rejects_duplicates_and_bad_weights() {
    let aapl = Asset::new("AAPL", AssetClass::Stock)
        .with_return(0.12)
        .with_volatility(0.20);
    let msft = Asset::new("MSFT", AssetClass::Stock)
        .with_return(0.10)
        .with_volatility(0.18);

    assert!(aapl.validate().is_ok());
    assert_eq!(aapl.symbol(), "AAPL");

    let portfolio = Portfolio::new(vec![aapl.clone(), msft.clone()]).unwrap();
    assert_eq!(portfolio.num_assets(), 2);
    assert_eq!(portfolio.symbols(), vec!["AAPL", "MSFT"]);

    assert!(Portfolio::new(vec![aapl.clone(), aapl]).is_err());
    assert!(portfolio.clone().with_weights(vec![0.9, 0.2]).is_err());
    assert!(portfolio.with_weights(vec![0.6, 0.4]).is_ok());
}

#[test]
fn return_series_computes_log_returns_and_covariance() {
    let symbols = vec!["AAA".to_string(), "BBB".to_string()];
    let prices = DMatrix::from_row_slice(
        4,
        2,
        &[100.0, 200.0, 110.0, 210.0, 121.0, 220.5, 133.1, 231.525],
    );

    let returns = ReturnSeries::from_prices(symbols.clone(), prices).unwrap();
    assert_eq!(returns.num_periods(), 3);
    assert_eq!(returns.num_assets(), 2);
    assert_eq!(returns.symbols(), symbols.as_slice());

    let means = returns.mean_returns();
    assert_close(means[0], (1.10_f64).ln() * 252.0, 1e-12);
    assert_close(means[1], (1.05_f64).ln() * 252.0, 1e-12);

    let covariance = returns.covariance_matrix().unwrap();
    assert_eq!(covariance.nrows(), 2);
    assert_eq!(covariance.ncols(), 2);
    assert_close(covariance[(0, 1)], covariance[(1, 0)], 1e-12);

    let correlation = returns.correlation_matrix().unwrap();
    assert_close(correlation[(0, 0)], 1.0, 1e-12);
    assert_close(correlation[(1, 1)], 1.0, 1e-12);
}

#[test]
fn qubo_budget_penalty_matches_reference_expansion() {
    let mut qubo = QUBOMatrix::new(2);
    PenaltyBuilder::budget(&mut qubo, 1, 3.0);

    assert_close(qubo.as_matrix()[(0, 0)], -3.0, 1e-12);
    assert_close(qubo.as_matrix()[(1, 1)], -3.0, 1e-12);
    assert_close(qubo.as_matrix()[(0, 1)], 6.0, 1e-12);
    assert_close(qubo.as_matrix()[(1, 0)], 6.0, 1e-12);
    assert_close(qubo.offset(), 3.0, 1e-12);

    assert_close(qubo.evaluate(&[true, false]).unwrap(), 0.0, 1e-12);
    assert_close(qubo.evaluate(&[false, true]).unwrap(), 0.0, 1e-12);
    assert_close(qubo.evaluate(&[true, true]).unwrap(), 3.0, 1e-12);
    assert_close(qubo.evaluate(&[false, false]).unwrap(), 3.0, 1e-12);
}

#[test]
fn formulation_matches_two_asset_reference() {
    let covariance = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.02]);
    let expected_returns = DVector::from_row_slice(&[0.10, 0.05]);
    let symbols = vec!["A".to_string(), "B".to_string()];

    let qubo = QUBOFormulation::new(0.5, 1)
        .unwrap()
        .with_budget_penalty(10.0)
        .build_from_params(&covariance, &expected_returns, &symbols)
        .unwrap();

    assert_close(qubo.as_matrix()[(0, 0)], -10.03, 1e-12);
    assert_close(qubo.as_matrix()[(1, 1)], -10.015, 1e-12);
    assert_close(qubo.as_matrix()[(0, 1)], 20.005, 1e-12);
    assert_close(qubo.offset(), 10.0, 1e-12);
    assert!(qubo.evaluate(&[true, false]).unwrap() < qubo.evaluate(&[false, true]).unwrap());
}

#[test]
fn brute_force_constrained_finds_known_minimum() {
    let matrix = DMatrix::from_row_slice(
        4,
        4,
        &[
            -4.0, 1.0, 1.0, 1.0, 1.0, -3.0, 1.0, 1.0, 1.0, 1.0, -2.0, 1.0, 1.0, 1.0, 1.0, -1.0,
        ],
    );
    let labels = vec!["A", "B", "C", "D"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let qubo = QUBOMatrix::from_matrix(matrix, labels).unwrap();

    let result = BruteForceSolver::solve_constrained(&qubo, 2).unwrap();

    assert_eq!(result.solution(), &[true, true, false, false]);
    assert_eq!(
        result.selected_symbols(),
        &["A".to_string(), "B".to_string()]
    );
    assert!(result.is_feasible(2));
}

#[test]
fn simulated_annealing_seeded_run_is_reproducible() {
    let mut qubo = QUBOMatrix::new(3);
    qubo.add(0, 0, -2.0).unwrap();
    qubo.add(1, 1, -1.0).unwrap();
    qubo.add(2, 2, 0.5).unwrap();

    let solver = SimulatedAnnealing::new()
        .with_seed(7)
        .with_max_iterations(250)
        .with_temperature(5.0);
    let first = solver.solve(&qubo).unwrap();
    let second = solver.solve(&qubo).unwrap();

    assert_eq!(first.solution(), second.solution());
    assert_eq!(first.objective_value(), second.objective_value());
}

#[test]
fn markowitz_min_variance_weights_sum_to_one() {
    let symbols = vec!["A".to_string(), "B".to_string()];
    let returns = DMatrix::from_row_slice(
        5,
        2,
        &[
            0.01, 0.02, 0.02, 0.01, 0.015, 0.018, 0.01, 0.015, 0.02, 0.02,
        ],
    );
    let series = ReturnSeries::from_returns(symbols.clone(), returns).unwrap();

    let result = MarkowitzSolver::new().min_variance(&series).unwrap();
    assert_eq!(result.symbols(), symbols.as_slice());
    assert_close(result.weights().iter().sum::<f64>(), 1.0, 1e-9);
    assert!(result.volatility().is_finite());
}

#[test]
fn markowitz_target_return_solver_matches_requested_return() {
    let symbols = vec!["A".to_string(), "B".to_string()];
    let returns = DMatrix::from_row_slice(
        5,
        2,
        &[
            0.01, 0.03, 0.02, 0.025, 0.015, 0.02, 0.012, 0.022, 0.018, 0.028,
        ],
    );
    let series = ReturnSeries::from_returns(symbols, returns).unwrap();
    let target = 5.0;

    let result = MarkowitzSolver::new()
        .with_target_return(target)
        .solve(&series)
        .unwrap();

    assert_close(result.weights().iter().sum::<f64>(), 1.0, 1e-9);
    assert_close(result.expected_return(), target, 1e-6);
}

#[test]
fn optimization_result_serializes_to_json() {
    let result = OptimizationResult::new(
        vec![true, false, true],
        -1.25,
        vec!["A".to_string(), "C".to_string()],
        "test-solver".to_string(),
        8,
        12,
        SolverMetrics::new(8, 3, vec![0.0, -1.25]),
    );

    let json = result.to_json().unwrap();
    assert!(json.contains("test-solver"));
    assert!(json.contains("selected_assets"));
}
