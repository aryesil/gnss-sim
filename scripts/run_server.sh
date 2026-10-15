#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=.venv
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
elif [ -f "$VENV/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/Scripts/activate"
else
  echo "Note: $VENV not found, using current Python environment..."
fi

# libiio/libad9361 were built into the venv's own lib/ (scripts/setup.sh),
# not a system prefix -- point the dynamic linker at them so pyadi-iio's
# ctypes dlopen finds them (macOS DYLD_LIBRARY_PATH, Linux LD_LIBRARY_PATH).
if [ -d "$VENV/lib" ]; then
  export DYLD_LIBRARY_PATH="$PWD/$VENV/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$PWD/$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec python -m uvicorn backend.app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "$@"
