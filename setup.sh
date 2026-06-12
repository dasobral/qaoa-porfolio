#!/usr/bin/env bash
# Development environment setup for the QAOA Portfolio Optimizer.
#
# Requirements: uv (https://docs.astral.sh/uv/) and a Rust toolchain.
set -euo pipefail

cd "$(dirname "$0")"

# The repository-standard Python environment lives in qaoa-env/, not .venv/.
export UV_PROJECT_ENVIRONMENT=qaoa-env

echo "==> Syncing Python environment (qaoa-env/) and building the Rust extension..."
uv sync --extra dev

echo "==> Verifying installation..."
uv run qaoa-portfolio --help >/dev/null
uv run pytest -q
cargo test

cat <<'EOF'

Setup complete. Daily usage:

  export UV_PROJECT_ENVIRONMENT=qaoa-env   # once per shell
  source qaoa-env/bin/activate             # optional: activate the venv
  uv run pytest                            # Python test suite
  cargo test                               # Rust test suite

EOF
