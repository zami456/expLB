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

CONFIG=deception_detection/scripts/configs/XXXX-9-llama-all.yaml

echo "### Starting XXXX-9-llama-all (resumable; re-run this script to continue)"
uv run python deception_detection/scripts/experiment.py run --config-file "$CONFIG"
status=$?

if [ $status -ne 0 ]; then
    echo "### Run exited with status $status. Nothing is lost -- re-run this script to resume."
    exit $status
fi

echo "### Scoring complete; generating transcripts and dashboards"
uv run python deception_detection/scripts/export_html.py
uv run python deception_detection/scripts/generate_html_dashboard.py
echo "### Done"
