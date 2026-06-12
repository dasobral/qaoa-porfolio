from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

pytestmark = pytest.mark.unit


def test_maturin_config_includes_python_package_sources():
    """uv sync must install both the Rust extension and qaoa_portfolio package."""

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    maturin_config = pyproject["tool"]["maturin"]

    assert maturin_config["module-name"] == "qaoa_portfolio_core"
    assert maturin_config["python-source"] == "."
