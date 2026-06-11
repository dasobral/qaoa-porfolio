from pathlib import Path


def test_qaoa_portfolio_core_wrapper_package_exists_for_maturin_layout():
    """maturin needs a Python package matching module-name when python-source is set."""

    wrapper = Path("qaoa_portfolio_core") / "__init__.py"

    assert wrapper.is_file()
    assert "qaoa_portfolio_core" in wrapper.read_text()
