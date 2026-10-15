#!/usr/bin/env bash
# Smoke for the user_stay matched pair, one job at a time, with each run folder's DEPLOYED code: per model,
# default + pc_gatekeeper x 5 variants x 2 questions = 20 replies -> judge -> offline span check -> activations
# (one layer, min_count 1). Writes only under xmodel/userstay_pair/smoke/. Progress: smoke/status.log.
#
#   bash xmodel/userstay_pair/smoke.sh        (from anywhere)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv-aa-win/Scripts/python.exe"
BASE="$USERPROFILE/xmodel-runs-userstay"
SM="$HERE/smoke"
mkdir -p "$SM"
ST="$SM/status.log"
say() { echo "$(date '+%F %T') $*" >> "$ST"; }
export PYTHONIOENCODING=utf8 NDIF_MIN_DOWNLOAD_KBPS=40 NDIF_DOWNLOAD_GRACE=20 NDIF_SESSION_TIMEOUT=5400 XMODEL_CSV_MODEL=none

say "smoke start"
for spec in "Qwen/Qwen2.5-72B-Instruct|Qwen2.5-72B-Instruct|Qwen" "meta-llama/Llama-3.3-70B-Instruct|Llama-3.3-70B-Instruct|Llama"; do
    IFS='|' read -r MODEL FOLDER SHORT <<< "$spec"
    export XMODEL_SHORT_NAME=$SHORT
    C="$BASE/$FOLDER/code"; S="$C/src/assistant_axis_ndif"; O="$SM/$FOLDER"
    mkdir -p "$O"
    cd "$C" || exit 1
    "$PY" "$S/1_generate_ndif.py" --model "$MODEL" --roles_dir kd/personas/kd_instructions --roles default pc_gatekeeper \
        --questions_file kd/questions/kd_questions_shared.jsonl --question_count 2 --output_dir "$O/responses" \
        --temperature 0.7 --top_p 0.9 --top_k 0 --repetition_penalty 1.0 --max_new_tokens 512 \
        --gen_batch 32 --session_batches 4 --retries 6 --prompt_format user_stay > "$O/generate.log" 2>&1
    say "$SHORT generate exit $? rows $(cat "$O"/responses/*.jsonl 2>/dev/null | grep -c .)/20"
    "$PY" "$S/3_judge_ndif.py" --responses_dir "$O/responses" --roles_dir kd/personas/kd_instructions \
        --output_dir "$O/scores" --judge_batch 16 --session_batches 8 > "$O/judge.log" 2>&1
    say "$SHORT judge exit $? scores $(cat "$O"/scores/*.json 2>/dev/null)"
    "$PY" "$HERE/spans_check.py" --code "$C" --model "$MODEL" --responses "$O/responses" > "$O/spans.log" 2>&1
    say "$SHORT spans exit $? : $(tail -1 "$O/spans.log")"
    "$PY" "$S/2_activations_ndif.py" --model "$MODEL" --responses_dir "$O/responses" --scores_dir "$O/scores" \
        --output_dir "$O/vectors" --roles default pc_gatekeeper --keep_scores 0,1,2,3 --min_count 1 \
        --max_length 8192 --layers target --batch_size 8 --session_batches 20 > "$O/activations.log" 2>&1
    say "$SHORT activations exit $? vectors $(ls "$O"/vectors/*.pt 2>/dev/null | wc -l)/2"
done
say "smoke DONE"
