#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${MM_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Set MM_PYTHON to the prepared Python 3.13 interpreter; see README.md." >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/code"
exec "$PYTHON" "$ROOT/tools/launch.py" "$@"
