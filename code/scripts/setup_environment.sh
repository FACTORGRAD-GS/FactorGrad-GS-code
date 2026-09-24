#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-fastgs}"
LOCK="$ROOT/environments/$NAME"
test -f "$LOCK/conda-explicit.txt"
if [[ "$NAME" == factorgrad-mini ]]; then
  if [[ -z "${FACTORGRAD_MINI_SOURCE:-}" || ! -d "$FACTORGRAD_MINI_SOURCE/ms" ]]; then
    printf 'Mini-Splatting source is not bundled; set FACTORGRAD_MINI_SOURCE to a separate checkout with an ms/ folder.\n' >&2
    exit 2
  fi
fi
# A new installation only: never overwrite a working research environment.
if conda run -n "$NAME" python -c 'import sys' >/dev/null 2>&1; then
  printf 'Environment %s already exists; use the existing installation or a separate Conda root.\n' "$NAME" >&2
  exit 2
fi
conda create --yes --name "$NAME" --file "$LOCK/conda-explicit.txt"
PIP=(conda run --no-capture-output -n "$NAME" python -m pip)
EXTRA=()
case "$NAME" in
  LeGS) EXTRA=(--extra-index-url https://download.pytorch.org/whl/cu128 --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html) ;;
  factorgrad-shorter) EXTRA=(--extra-index-url https://download.pytorch.org/whl/cu117) ;;
esac
"${PIP[@]}" install --no-deps "${EXTRA[@]}" --requirement "$LOCK/requirements.txt"
case "$NAME" in
  fastgs) SOURCE="$ROOT/sources/primary/fastgs-factorgrad" ;;
  factorgrad-vanilla) SOURCE="$ROOT/sources/baselines/3dgs" ;;
  factorgrad-mini) SOURCE="$FACTORGRAD_MINI_SOURCE" ;;
  factorgrad-speedy) SOURCE="$ROOT/sources/baselines/speedy" ;;
  factorgrad-taming) SOURCE="$ROOT/sources/transfer/taming" ;;
  factorgrad-dash) SOURCE="$ROOT/sources/transfer/dash" ;;
  LeGS) SOURCE="$ROOT/sources/transfer/legs" ;;
  factorgrad-shorter) SOURCE="$ROOT/sources/shorter_transfer" ;;
  *) printf 'Unknown environment: %s\n' "$NAME" >&2; exit 2 ;;
esac
if [[ "$NAME" == factorgrad-shorter ]]; then
  for MODULE in simple-knn fused_ssim lanczos-resampling gaussian_raster; do
    "${PIP[@]}" install --no-build-isolation --no-deps "$SOURCE/litegs/submodules/$MODULE"
  done
else
  for SETUP in "$SOURCE"/submodules/*/setup.py; do
    "${PIP[@]}" install --no-build-isolation --no-deps "$(dirname "$SETUP")"
  done
fi
conda run -n "$NAME" python -c 'import torch; print("Torch:",torch.__version__,"CUDA runtime:",torch.version.cuda,"GPU:",torch.cuda.get_device_name(0))'
