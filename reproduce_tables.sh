#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$ROOT/code/scripts/reproduce.py" --action recorded --suite all \
  --artifacts "$ROOT/results/evidence" --output "$ROOT/results/aggregated" "$@"
