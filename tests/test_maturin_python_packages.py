from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


def test_maturin_includes_python_and_rust_wrapper_packages():
    """The wheel must contain the CLI package and Rust extension wrapper package."""

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    maturin_config = pyproject["tool"]["maturin"]

    assert maturin_config["python-packages"] == [
        "qaoa_portfolio",
        "qaoa_portfolio_core",
    ]
