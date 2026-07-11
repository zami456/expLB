# !/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1,7

# Run the experiment

echo "Start with XXXX-9"
uv run python deception_detection/scripts/experiment.py run --config-file deception_detection/scripts/configs/XXXX-9.yaml
echo "### Done with XXXX-9"

# echo "Start Sleeper..."
# uv run python deception_detection/scripts/experiment.py run --config-file deception_detection/scripts/configs/XXXX-18.yaml
# uv run python deception_detection/scripts/experiment.py run --config-file deception_detection/scripts/configs/XXXX-17.yaml
# uv run python deception_detection/scripts/experiment.py run --config-file deception_detection/scripts/configs/XXXX-19.yaml
# echo "### Done with sleeper"

# echo "Start Convince Me"
# uv run python deception_detection/scripts/experiment.py run --config-file deception_detection/scripts/configs/XXXX-15.yaml  
# echo "### Done with convince me"

# echo "start Apollo"
# uv run python deception_detection/scripts/experiment.py run --config-file deception_detection/scripts/configs/repe.yaml
# echo "### Done with repe"

# generate html transcripts (export_html.py)
uv run python deception_detection/scripts/export_html.py
# generate dashboards (will auto-detect latest experiment)
uv run python deception_detection/scripts/generate_html_dashboard.py