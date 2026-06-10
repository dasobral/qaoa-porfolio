use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use qaoa_portfolio::optimization::{BruteForceSolver, MarkowitzSolver, SimulatedAnnealing};
use qaoa_portfolio::portfolio::ReturnSeries;
use qaoa_portfolio::qubo::QUBOFormulation;

fn sample_returns(periods: usize, assets: usize) -> ReturnSeries {
    let symbols = (0..assets)
        .map(|index| format!("A{index}"))
        .collect::<Vec<_>>();
    let values = (0..periods)
        .flat_map(|row| {
            (0..assets).map(move |col| {
                let drift = 0.0002 * (col + 1) as f64;
                let cycle = ((row + col) % 7) as f64 * 0.0001;
                drift + cycle
            })
        })
        .collect::<Vec<_>>();
    ReturnSeries::from_returns(symbols, DMatrix::from_row_slice(periods, assets, &values))
        .expect("benchmark fixture returns are valid")
}

fn benchmark_qubo_build(c: &mut Criterion) {
    let returns = sample_returns(252, 20);
    c.bench_function("qubo_build_20_assets", |bench| {
        bench.iter(|| {
            QUBOFormulation::new(0.5, 5)
                .expect("valid formulation")
                .build(&returns)
                .expect("valid QUBO")
        })
    });
}

fn benchmark_brute_force(c: &mut Criterion) {
    let covariance = DMatrix::identity(12, 12) * 0.02;
    let expected = DVector::from_element(12, 0.08);
    let symbols = (0..12).map(|index| format!("A{index}")).collect::<Vec<_>>();
    let qubo = QUBOFormulation::new(0.5, 4)
        .expect("valid formulation")
        .build_from_params(&covariance, &expected, &symbols)
        .expect("valid QUBO");

    c.bench_function("brute_force_12_assets", |bench| {
        bench.iter(|| BruteForceSolver::solve_constrained(&qubo, 4).expect("solver succeeds"))
    });
}

fn benchmark_simulated_annealing(c: &mut Criterion) {
    let returns = sample_returns(252, 50);
    let qubo = QUBOFormulation::new(0.5, 10)
        .expect("valid formulation")
        .build(&returns)
        .expect("valid QUBO");
    let solver = SimulatedAnnealing::new()
        .with_seed(42)
        .with_max_iterations(1_000);

    c.bench_function("simulated_annealing_50_assets", |bench| {
        bench.iter(|| solver.solve(&qubo).expect("solver succeeds"))
    });
}

fn benchmark_markowitz(c: &mut Criterion) {
    let returns = sample_returns(252, 50);
    c.bench_function("markowitz_min_variance_50_assets", |bench| {
        bench.iter(|| {
            MarkowitzSolver::new()
                .min_variance(&returns)
                .expect("solver succeeds")
        })
    });
}

criterion_group!(
    benches,
    benchmark_qubo_build,
    benchmark_brute_force,
    benchmark_simulated_annealing,
    benchmark_markowitz
);
criterion_main!(benches);
