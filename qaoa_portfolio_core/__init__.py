"""Python wrapper package for the compiled qaoa_portfolio_core extension.

The compiled extension is installed as ``qaoa_portfolio_core.qaoa_portfolio_core``
when maturin builds this mixed Rust/Python project.
"""

from importlib import import_module

try:
    _native = import_module(".qaoa_portfolio_core", __name__)
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local build state
    if exc.name != f"{__name__}.qaoa_portfolio_core":
        raise
    raise ImportError(
        "qaoa_portfolio_core native extension is not installed. "
        "Run `uv sync` or `python3 -m maturin build --features python-bindings`."
    ) from exc

for _name in dir(_native):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_native, _name)

__all__ = [_name for _name in dir(_native) if not _name.startswith("_")]
