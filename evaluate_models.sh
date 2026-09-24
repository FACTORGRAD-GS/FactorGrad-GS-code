#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Requires full model artifacts supplied separately; never substitutes saved metrics.
exec python3 "$ROOT/code/scripts/reproduce.py" --action evaluate --suite table1 "$@"
