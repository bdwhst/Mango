#!/usr/bin/env bash
# Configure, build and test on macOS / Linux.
#   bash scripts/build.sh            # core + tests, no LibTorch
#   bash scripts/build.sh --torch    # with LibTorch (MPS on macOS, CUDA on Linux)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TORCH=0
NOTEST=0
for a in "$@"; do
  case "$a" in
    --torch) TORCH=1 ;;
    --no-test) NOTEST=1 ;;
  esac
done
if [[ "$(uname -s)" == "Darwin" ]]; then
  PRESET=$([[ $TORCH == 1 ]] && echo macos-mps || echo macos)
else
  PRESET=linux
  [[ $TORCH == 1 ]] && EXTRA="-DMANGO_WITH_TORCH=ON" || EXTRA=""
fi
cmake --preset "$PRESET" ${EXTRA:-}
cmake --build --preset "$PRESET"
if [[ $NOTEST == 0 ]]; then
  ctest --preset "$PRESET"
fi
