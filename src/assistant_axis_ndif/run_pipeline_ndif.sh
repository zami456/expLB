#!/bin/bash
#
# Assistant Axis for Llama-3.3-70B-Instruct -- every model call on NDIF.
#
#   1_generate_ndif.py   role responses          (Llama-3.3-70B on NDIF)
#   3_judge_ndif.py      0-3 role-adherence      (Llama-3.1-405B on NDIF)
#   2_activations_ndif.py per-role persona vectors (Llama-3.3-70B on NDIF)
#   5_axis.py            axis = mean(default) - mean(roles)   (local, CPU)
#
# Every step is independently restartable -- safe to Ctrl-C and re-run.
#
# Disk: ~1 GB of new space. responses/ is 0.4-0.8 GB at full scale and is a
# deletable byproduct; the deliverable (vectors/ + axis.pt) is ~9 MB.
#
# Config (env vars):
#   OUT_DIR          (required) output dir; needs ~1 GB free
#   MODEL            default meta-llama/Llama-3.3-70B-Instruct
#   JUDGE_MODEL      default meta-llama/Llama-3.1-405B-Instruct (HOT+pinned on NDIF)
#   LAYERS           default "all" = every layer (80 for Llama-70B; ~2.6 MB/role),
#                    which is what the per-layer axis and activation capping need.
#                    "target" = middle layer only (40), ~32 KB/role. Or an explicit
#                    list, e.g. "56 57 58".
#   ROLES           optional space-separated subset, e.g. "assistant pirate default"
#   ROLES_FILE      optional JSON role-set with group labels (default: role_groups.json,
#                    the curated 100-role deception/sycophancy set). ROLES overrides it.
#   GROUPS          optional subset of ROLES_FILE groups, e.g. "deception honesty"
#   PROMPT_INDICES  optional, e.g. "0 1"   (default: all 5)
#   QUESTION_COUNT  optional, default 240
#   GEN_BATCH / JUDGE_BATCH / ACT_BATCH   optional batch sizes
#   GEN_SESSION_BATCHES  batches per NDIF session in step 1 (default 8). Generation
#                    is slow, so keep this * seconds-per-batch under NDIF's ~1h
#                    per-request ceiling; step 0c's measurement sets it.
#   JUDGE_SESSION_BATCHES  batches per NDIF session in step 2 (default 8). A judge call
#                    is ~4 output tokens, so bundling mostly removes queue overhead.
#   ACT_SESSION_BATCHES  batches per NDIF session in step 3 (default 20). Forward-only
#                    passes are much faster, so this can be larger.
#
# Requires src/assistant_axis_ndif/.env  (see env.aa.example).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${AA_PY:-$REPO/.venv-aa/bin/python}"   # AA_PY: override for a venv outside the repo (e.g. WSL-native)
PIPE="$REPO/src/assistant_axis_ndif"          # ported NDIF stages
AA_PIPE="$REPO/assistant_axis/pipeline"       # upstream, unmodified (5_axis.py)
ROLES_DIR="$REPO/assistant_axis/data/roles/instructions"

MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/Llama-3.1-405B-Instruct}"
OUT_DIR="${OUT_DIR:?set OUT_DIR to an output directory with ~1 GB free}"
LAYERS="${LAYERS:-all}"
QUESTION_COUNT="${QUESTION_COUNT:-240}"
GEN_BATCH="${GEN_BATCH:-24}"
JUDGE_BATCH="${JUDGE_BATCH:-48}"
JUDGE_SESSION_BATCHES="${JUDGE_SESSION_BATCHES:-8}"
ACT_BATCH="${ACT_BATCH:-8}"
GEN_SESSION_BATCHES="${GEN_SESSION_BATCHES:-8}"
ACT_SESSION_BATCHES="${ACT_SESSION_BATCHES:-20}"
MIN_COUNT="${MIN_COUNT:-50}"   # min score-3 responses/role for a vector (lower for smoke tests)

[ -x "$PY" ] || { echo "missing $PY -- run: bash $PIPE/setup_venv_aa.sh"; exit 1; }
[ -f "$PIPE/.env" ] || { echo "missing $PIPE/.env -- cp $PIPE/env.aa.example $PIPE/.env and fill it"; exit 1; }

ROLES_FILE="${ROLES_FILE:-$PIPE/role_groups.json}"
ROLE_ARGS=()
if [ "${ROLES:-}" ]; then
    ROLE_ARGS=(--roles $ROLES)
elif [ -f "$ROLES_FILE" ]; then
    ROLE_ARGS=(--roles_file "$ROLES_FILE")
    [ "${GROUPS:-}" ] && ROLE_ARGS+=(--groups $GROUPS)
fi
PI_ARGS=();   [ "${PROMPT_INDICES:-}" ] && PI_ARGS=(--prompt_indices $PROMPT_INDICES)

mkdir -p "$OUT_DIR"
echo "=== Assistant Axis / NDIF ==="
echo "model=$MODEL  judge=$JUDGE_MODEL"
echo "out=$OUT_DIR  question_count=$QUESTION_COUNT  prompt_indices=${PROMPT_INDICES:-ALL}"
echo "roles=${ROLES:-${ROLES_FILE##*/}${GROUPS:+ [$GROUPS]}}"
echo "layers=$LAYERS  gen_batch=$GEN_BATCH  sessions: gen=$GEN_SESSION_BATCHES act=$ACT_SESSION_BATCHES"

echo; echo "=== step 1/5: generate (NDIF) ==="
"$PY" "$PIPE/1_generate_ndif.py" --model "$MODEL" \
    --output_dir "$OUT_DIR/responses" --question_count "$QUESTION_COUNT" \
    --gen_batch "$GEN_BATCH" --session_batches "$GEN_SESSION_BATCHES" \
    "${PI_ARGS[@]}" "${ROLE_ARGS[@]}"

echo; echo "=== step 2/5: judge (NDIF, $JUDGE_MODEL) ==="
"$PY" "$PIPE/3_judge_ndif.py" \
    --responses_dir "$OUT_DIR/responses" --roles_dir "$ROLES_DIR" \
    --output_dir "$OUT_DIR/scores" --judge_model "$JUDGE_MODEL" \
    --judge_batch "$JUDGE_BATCH" --session_batches "$JUDGE_SESSION_BATCHES"     "${ROLE_ARGS[@]}"

echo; echo "=== step 3/5: per-role persona vectors (NDIF) ==="
"$PY" "$PIPE/2_activations_ndif.py" --model "$MODEL" \
    --responses_dir "$OUT_DIR/responses" --scores_dir "$OUT_DIR/scores" \
    --output_dir "$OUT_DIR/vectors" --batch_size "$ACT_BATCH" --min_count "$MIN_COUNT" \
    --layers $LAYERS --session_batches "$ACT_SESSION_BATCHES" "${ROLE_ARGS[@]}"

echo; echo "=== step 4/5: assistant axis (local, upstream) ==="
"$PY" "$AA_PIPE/5_axis.py" --vectors_dir "$OUT_DIR/vectors" --output "$OUT_DIR/axis.pt"

echo; echo "=== step 5/5: axes + PCA + validation (local) ==="
"$PY" "$PIPE/6_axes.py" --vectors_dir "$OUT_DIR/vectors" --output "$OUT_DIR/axes.pt"     ${ROLES_FILE:+--roles_file "$ROLES_FILE"}

echo; echo "=== done ==="
echo "persona vectors : $OUT_DIR/vectors/*.pt"
echo "axes+PCA       : $OUT_DIR/axes.pt"
echo "assistant axis  : $OUT_DIR/axis.pt"
