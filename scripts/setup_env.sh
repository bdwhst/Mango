#!/usr/bin/env bash
# macOS (Apple Silicon) / Linux environment setup: Python venv + PyTorch (MPS on macOS, CUDA on Linux if available).
# Run from the repository root:  bash scripts/setup_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  echo "Creating venv (.venv)..."
  "$PYTHON" -m venv .venv
fi
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install --upgrade pip
if [[ "$(uname -s)" == "Darwin" ]]; then
  "$PY" -m pip install -r requirements.txt
  "$PY" -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available()); print('cmake prefix', torch.utils.cmake_prefix_path)"
else
  "$PY" -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('cmake prefix', torch.utils.cmake_prefix_path)"
fi

echo
echo "Toolchain check:"
cmake --version | head -1 || echo "cmake missing: brew install cmake ninja"
command -v ninja >/dev/null && ninja --version || echo "ninja missing (optional): brew install ninja"
if [[ "$(uname -s)" == "Darwin" ]]; then
  xcode-select -p || echo "Xcode command line tools missing: xcode-select --install"
fi

echo
echo "Done. Configure the C++ engine with:"
echo "  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build"
