#!/bin/bash
# Llama-70B-3.3 over all five base liars-bench subsets, with resumable downloads.
#
# Safe to re-run: activations already fetched are loaded from disk, datasets already
# scored are skipped, and a trained detector.pt is reused. If the run dies -- NDIF
# eviction, a dropped connection, Ctrl-C -- just run this again and it picks up where it
# stopped. The results folder is deliberately NOT deleted on failure.
#
# Cache location defaults to activations/cache next to this repo (gitignored). Point it
# somewhere with more room if needed:
#   export ACTS_CACHE_DIR=/path/with/space
# Expect roughly 60 GB for a full sweep at bfloat16; the first batch logs the real figure
# and warns if it will not fit.

set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# nnsight's remote progress display prints non-ASCII (e.g. U+25C9). On Windows consoles
# defaulting to cp1252 that raises UnicodeEncodeError mid-trace and kills the run.
export PYTHONIOENCODING=utf-8

# nnsight authenticates against NDIF with the NDIF_API_KEY environment variable
# (nnsight/intervention/backends/remote.py falls back to it). Nothing in the probe
# pipeline loads .env, so pull the two keys across explicitly and fail early rather than
# hitting an auth error part way into a long download.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/../../../../.." && pwd))"
ENV_FILE="$REPO_ROOT/.env"

# `uv run` is unusable inside this vendored package: the anonymisation pass replaced the
# wheel URLs in its uv.lock with XXXX, so uv fails to parse the lockfile. Use the repo's
# venv interpreter directly. Override with PYTHON=... if yours lives elsewhere.
if [ -z "${PYTHON:-}" ]; then
    for candidate in "$REPO_ROOT/.venv/Scripts/python.exe" "$REPO_ROOT/.venv/bin/python"; do
        if [ -x "$candidate" ]; then PYTHON="$candidate"; break; fi
    done
fi
if [ -z "${PYTHON:-}" ] || [ ! -x "$PYTHON" ]; then
    echo "No Python interpreter found (looked under $REPO_ROOT/.venv)." >&2
    echo "Set PYTHON=/path/to/python and re-run." >&2
    exit 1
fi
if [ -f "$ENV_FILE" ]; then
    for key in NDIF_API_KEY HF_TOKEN; do
        value="$(grep -E "^${key}=" "$ENV_FILE" | head -1 | cut -d= -f2-)"
        if [ -n "$value" ]; then export "$key=$value"; fi
    done
fi

if [ -z "${NDIF_API_KEY:-}" ]; then
    echo "NDIF_API_KEY is not set (looked in the environment and $ENV_FILE)." >&2
    echo "Activation extraction runs remotely and cannot authenticate without it." >&2
    echo "Get a key at https://login.ndif.us" >&2
    exit 1
fi

cd "$SCRIPT_DIR/../.." || exit 1   # package root; config paths are relative to it
CONFIG=deception_detection/scripts/configs/XXXX-9-llama-all.yaml

echo "### Starting XXXX-9-llama-all (resumable; re-run this script to continue)"
"$PYTHON" -u deception_detection/scripts/experiment.py run --config-file "$CONFIG"
status=$?

if [ $status -ne 0 ]; then
    echo "### Run exited with status $status. Nothing is lost -- re-run this script to resume."
    exit $status
fi

echo "### Scoring complete; generating transcripts and dashboards"
"$PYTHON" deception_detection/scripts/export_html.py
"$PYTHON" deception_detection/scripts/generate_html_dashboard.py
echo "### Done"
