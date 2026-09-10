#!/bin/bash
# Create the isolated Python env for the NDIF Assistant-Axis pipeline.
#
#   bash assistant_axis/pipeline/setup_venv_aa.sh
#
# Makes  <repo>/.venv-aa  (CPU-only, ~2 GB). Does NOT touch the repo's .venv / uv.lock.
# Requires `uv`  (curl -LsSf https://astral.sh/uv/install.sh | sh).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO/.venv-aa"

command -v uv >/dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

echo "=== creating $VENV (python 3.12) ==="
uv venv --python 3.12 "$VENV"

echo "=== installing pinned deps (CPU torch) ==="
VIRTUAL_ENV="$VENV" UV_LINK_MODE=copy uv pip install --python "$VENV/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  "nnsight==0.7.0" \
  "torch==2.14.0" \
  "transformers==5.17.0" \
  "python-socketio==5.16.4" \
  "python-engineio==4.13.0" \
  "websocket-client==1.8.0" \
  "huggingface_hub" \
  "python-dotenv" \
  "jsonlines" \
  "tqdm"
# NOTE: websocket-client 1.9.x + python-engineio 4.14 crash the NDIF result socket
# on long-running jobs ("invalid literal for int() ... b'P'"). The pins above are the
# combo verified to work for multi-minute remote .generate() / .trace() calls.

echo
echo "=== done. quick import check (first run is slow, ~3 min cold) ==="
"$VENV/bin/python" - <<'PY'
import time; t=time.time()
import torch, transformers, nnsight
print(f"torch {torch.__version__} | transformers {transformers.__version__} | nnsight {nnsight.__version__}  ({time.time()-t:.0f}s)")
PY
echo
echo "Next: cp assistant_axis/pipeline/env.aa.example assistant_axis/pipeline/.env  and fill it in."
