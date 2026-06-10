use nalgebra::DMatrix;
use proptest::prelude::*;
use qaoa_portfolio::portfolio::ReturnSeries;
use qaoa_portfolio::qubo::{PenaltyBuilder, QUBOMatrix};

fn symbols(count: usize) -> Vec<String> {
    (0..count).map(|index| format!("A{index}")).collect()
}

proptest! {
    #[test]
    fn covariance_matrix_is_symmetric(values in prop::collection::vec(-0.05_f64..0.05, 120..180)) {
        let rows = values.len() / 3;
        let matrix = DMatrix::from_row_slice(rows, 3, &values[..rows * 3]);
        let series = ReturnSeries::from_returns(symbols(3), matrix).unwrap();

        let covariance = series.covariance_matrix().unwrap();

        for row in 0..covariance.nrows() {
            for col in 0..covariance.ncols() {
                prop_assert!((covariance[(row, col)] - covariance[(col, row)]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn qubo_evaluation_is_deterministic(solution in prop::collection::vec(any::<bool>(), 4)) {
        let mut qubo = QUBOMatrix::new(4);
        qubo.add(0, 0, -1.0).unwrap();
        qubo.add(1, 2, 0.25).unwrap();
        qubo.add(3, 3, 2.0).unwrap();

        let first = qubo.evaluate(&solution).unwrap();
        let second = qubo.evaluate(&solution).unwrap();

        prop_assert_eq!(first, second);
    }

    #[test]
    fn budget_penalty_prefers_target_cardinality(mask in 0_u8..16) {
        let mut qubo = QUBOMatrix::new(4);
        PenaltyBuilder::budget(&mut qubo, 2, 1.0);
        let solution = (0..4)
            .map(|index| (mask & (1 << index)) != 0)
            .collect::<Vec<_>>();
        let selected = solution.iter().filter(|value| **value).count();
        let objective = qubo.evaluate(&solution).unwrap();

        if selected == 2 {
            prop_assert!(objective.abs() < 1e-9);
        } else {
            prop_assert!(objective > 0.0);
        }
    }
}

#[test]
fn qubo_matrix_set_and_add_keep_matrix_symmetric() {
    let mut qubo = QUBOMatrix::new(3);
    qubo.set(0, 2, 1.5).unwrap();
    qubo.add(1, 2, -0.25).unwrap();

    for row in 0..3 {
        for col in 0..3 {
            assert!((qubo.as_matrix()[(row, col)] - qubo.as_matrix()[(col, row)]).abs() < 1e-12);
        }
    }
}
