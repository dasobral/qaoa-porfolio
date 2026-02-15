"""
Command-line interface for QAOA Portfolio Optimizer.
"""

import argparse
import asyncio
import sys
from typing import List, Optional

from .exceptions import MarketDataError
from .portfolios import quick_portfolio_load, list_portfolio_presets


def _parse_symbols(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    symbols = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return symbols or None


async def _run(args: argparse.Namespace) -> int:
    symbols = _parse_symbols(args.symbols)

    try:
        price_data, returns_data = await quick_portfolio_load(
            symbols=symbols,
            portfolio_type=args.portfolio_type,
            days_back=args.days_back,
            preset=args.preset,
        )
    except (ValueError, MarketDataError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    summary = {
        "rows": len(price_data),
        "price_columns": price_data.shape[1] if hasattr(price_data, "shape") else 0,
        "return_rows": len(returns_data),
        "return_columns": returns_data.shape[1] if hasattr(returns_data, "shape") else 0,
    }

    print("QAOA Portfolio load summary")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QAOA Portfolio Optimizer CLI")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols, e.g. AAPL,MSFT,BTC-USD",
    )
    parser.add_argument(
        "--portfolio-type",
        choices=["stock", "crypto", "mixed"],
        default="stock",
        help="Sample portfolio type when symbols/preset are not provided.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=252,
        help="Number of calendar days to load.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help=f"Preset portfolio name. Available: {', '.join(list_portfolio_presets().keys())}",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = asyncio.run(_run(args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
