mod bridge;

pub use bridge::*;

use pyo3::prelude::*;

#[pymodule]
pub fn qaoa_portfolio_core(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    bridge::register(py, module)
}
