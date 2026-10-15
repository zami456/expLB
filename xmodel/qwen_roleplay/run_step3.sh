#!/usr/bin/env bash
# Owner 2026-10-14: "go ahead with both" -- (a) Qwen confirm (195 items on the 13 weakest roles) generate -> judge ->
# analyse, then (b) Llama-3.3 user_stay (175 screen items) generate -> judge -> analyse_screen. One job at a time.
# Every step is resumable; re-running this script continues where it stopped. Progress: runs_confirm/status.log.
#
#   bash xmodel/qwen_roleplay/run_step3.sh        (from the repo root)
set -u
cd "$(dirname "$0")/../.."
Q=xmodel/qwen_roleplay
ST=$Q/runs_confirm/status.log
PY=.venv-aa-win/Scripts/python.exe
export NDIF_MIN_DOWNLOAD_KBPS=40 NDIF_DOWNLOAD_GRACE=20 PYTHONIOENCODING=utf8
say() { echo "$(date '+%F %T') $*" >> "$ST"; }
count() { cat "$1"/*.jsonl 2>/dev/null | grep -c .; }
scored() { "$PY" -c "import json,glob,sys; print(sum(len(json.load(open(f))) for f in glob.glob(sys.argv[1]+'/*.json')))" "$1"; }

say "chain start pid $$"
# (a) Qwen confirm
timeout 3h "$PY" $Q/generate.py --bank $Q/bank/confirm_items.jsonl --out $Q/runs_confirm \
    --gen_batch 32 --session_batches 4 --retries 6 >> $Q/runs_confirm/generate.log 2>&1
say "confirm generate exit $? replies $(count $Q/runs_confirm/user_stay/responses)/195"
[ "$(count $Q/runs_confirm/user_stay/responses)" -eq 195 ] || { say "STOP: confirm incomplete"; exit 1; }
timeout 2h "$PY" xmodel/code/src/assistant_axis_ndif/3_judge_ndif.py --responses_dir $Q/runs_confirm/user_stay/responses \
    --roles_dir $Q/roles_kd --output_dir $Q/runs_confirm/user_stay/scores >> $Q/runs_confirm/judge_user_stay.log 2>&1
say "confirm judge exit $? scored $(scored $Q/runs_confirm/user_stay/scores)/195"
.venv-analysis/Scripts/python.exe -I $Q/analyse_confirm.py > $Q/runs_confirm/analysis_confirm.txt 2>&1
say "analyse_confirm exit $?"

# (b) Llama-3.3 user_stay
timeout 3h "$PY" $Q/generate.py --model meta-llama/Llama-3.3-70B-Instruct --bank $Q/bank/screen_items.jsonl \
    --out $Q/runs_llama33 --conds user_stay --gen_batch 32 --session_batches 4 --retries 6 >> $Q/runs_llama33/generate.log 2>&1
say "llama user_stay generate exit $? replies $(count $Q/runs_llama33/user_stay/responses)/175"
[ "$(count $Q/runs_llama33/user_stay/responses)" -eq 175 ] || { say "STOP: llama user_stay incomplete"; exit 1; }
timeout 2h "$PY" xmodel/code/src/assistant_axis_ndif/3_judge_ndif.py --responses_dir $Q/runs_llama33/user_stay/responses \
    --roles_dir kd/personas/kd_instructions --output_dir $Q/runs_llama33/user_stay/scores >> $Q/runs_llama33/judge_user_stay.log 2>&1
say "llama judge exit $? scored $(scored $Q/runs_llama33/user_stay/scores)/175"
.venv-analysis/Scripts/python.exe -I $Q/analyse_screen.py --runs $Q/runs_screen --llama_runs $Q/runs_llama33 \
    > $Q/runs_screen/analysis_screen.txt 2>&1
say "analyse_screen exit $?"
say "chain DONE"
