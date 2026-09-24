#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The manifest fixes all algorithm arguments. Data and Conda roots are user paths.
exec python3 "$ROOT/code/scripts/reproduce.py" --action train --suite main \
  --artifacts "$ROOT/results/evidence" "$@"
