#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/backend"

echo "==> Checking formatting..."
uv run ruff format --check .

echo "==> Running Ruff..."
uv run ruff check .

echo "==> Running Pyrefly..."
uv run pyrefly check

echo "==> Running tests..."
uv run pytest

echo "==> All checks passed."