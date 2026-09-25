#!/bin/bash
# Mango GUI (docs/DESIGN.md 6.7): double-click in Finder to play a model or analyse a game.
# Needs the venv (scripts/setup_env.sh) and the LibTorch build (scripts/build.sh --torch).
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "The Python venv is missing: run scripts/setup_env.sh first."
  read -r -p "Press Enter to close."
  exit 1
fi
exec .venv/bin/python scripts/gui.py "$@"
