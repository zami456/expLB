#!/usr/bin/env bash
# Watch the Llama-3.1-405B run (resumed 2026-10-11); every stdout line is one event for Claude's Monitor tool.
# The guard (xmodel/guard405.ps1, its own scheduled task) does the stopping; this only reports.
#
#   every 60 s   new lines in the persona chain, K chain, Alpaca chains, guard and Drive-copy status files
#   every 10 min one progress line; an ALERT if nothing has moved for 60 min, or if a 405B task is not Ready/Running
#
# Same pattern as random_persona_bank/scripts/rpb_monitor.sh (memory note monitor-orphans-on-windows): exits on its
# own after $LIFE seconds, a token file retires an older copy, read offsets persist in <run>/logs/monitor405.state.
#   bash xmodel/monitor405.sh [life_seconds]
R=/c/Users/SURAIYA/xmodel-runs/Llama-3.1-405B-Instruct
L=$R/logs
STATE=$L/monitor405.state
TOKEN=$L/monitor405.token
LIFE=${1:-1740}
me="$$-$(date +%s)"; echo "$me" > "$TOKEN"; start=$(date +%s)

declare -A off
fresh=1
if [ -f "$STATE" ]; then fresh=0; while IFS=$'\t' read -r k v; do [ -n "$k" ] && off["$k"]=$v; done < "$STATE"; fi
save_state() { : > "$STATE.tmp"; for k in "${!off[@]}"; do printf '%s\t%s\n' "$k" "${off[$k]}" >> "$STATE.tmp"; done; mv -f "$STATE.tmp" "$STATE"; }
clean() { tr -d '\r' | sed 's/^\xEF\xBB\xBF//'; }

watch_lines() {  # file label [grep-filter]
  local f=$1 label=$2 filt=${3:-.} n o
  [ -f "$f" ] || return
  n=$(clean < "$f" | wc -l); o=${off["L:$f"]:-}
  if [ -z "$o" ]; then if [ "$fresh" = 1 ]; then o=$n; else o=0; fi; fi
  [ "$n" -lt "$o" ] && o=0
  [ "$n" -gt "$o" ] && clean < "$f" | sed -n "$((o + 1)),${n}p" | grep -E -- "$filt" | sed "s/^/[$label] /"
  off["L:$f"]=$n
}
last() { tail -1 "$1" 2>/dev/null | clean; }

progress() {
  local rep sco v b g a
  rep=$(cat "$R"/persona/responses/*.jsonl 2>/dev/null | wc -l)
  sco=$(cat "$R"/persona/scores/*.json 2>/dev/null | grep -o '"pos_p' | wc -l)
  v=""; for b in vectors_s23 vectors_all vectors_s3; do v="$v ${b#vectors_}=$(ls "$R/persona/$b"/*.pt 2>/dev/null | wc -l)"; done
  g=$(cat "$R"/eval/alpaca_gen/replies.*.jsonl 2>/dev/null | wc -l)
  a=$(ls "$R"/eval/alpaca/chunk_*.pt 2>/dev/null | wc -l)
  SIG="$rep $sco $v | $(last "$R/K/k_chain.status" | cut -c1-80) | $g $a"
  echo "[progress] replies $rep/5400, scored $sco/5250, vectors$v | K: $(last "$R/K/k_chain.status" | cut -c21-100) | Alpaca replies $g/2000, read chunks $a"
}
done_all() {
  [[ "$(last "$L/chain.status")" == DONE* ]] && [[ "$(last "$R/K/k_chain.status")" =~ \ \ DONE$ ]] && [[ "$(last "$L/alpaca_gen.status")" =~ \ \ DONE$ ]]
}
check_tasks() {
  local s
  s=$(powershell.exe -NoProfile -Command "Get-ScheduledTask -TaskName 'XModel-llama405*' | ForEach-Object { \$_.TaskName + '=' + \$_.State }" 2>/dev/null | clean | tr '\n' ' ')
  local bad; bad=$(tr ' ' '\n' <<<"$s" | grep -v -E '=(Ready|Running)$' | grep . | tr '\n' ' ')
  if [ -n "$bad" ] && [ "$bad" != "${off[TASKS]:-}" ]; then echo "[ALERT] task not Ready/Running: $bad"; fi
  off[TASKS]=$bad
}

progress; echo "[monitor] armed (pid $$, exits after $((LIFE / 60)) min, $( [ "$fresh" = 1 ] && echo first run || echo resuming ))"
[ "$SIG" != "${off[SIG]:-}" ] && { off[SIG]=$SIG; off[SIGT]=$(date +%s); }
i=0
while :; do
  [ "$(cat "$TOKEN" 2>/dev/null)" = "$me" ] || exit 0
  watch_lines "$L/chain.status" persona
  watch_lines "$R/K/k_chain.status" K
  watch_lines "$L/alpaca_gen.status" alpaca
  watch_lines "$L/alpaca.status" "alpaca read"
  watch_lines "$L/guard.status" guard
  watch_lines "$L/sync.status" drive 'END exit [^0]|GAVE UP|NO RCLONE'
  if [ $((i % 10)) = 0 ] && [ "$i" -gt 0 ]; then
    progress; check_tasks
    now=$(date +%s)
    if [ "$SIG" != "${off[SIG]:-}" ]; then off[SIG]=$SIG; off[SIGT]=$now
    elif [ $((now - ${off[SIGT]:-$now})) -ge 3600 ] && [ "${off[SIGW]:-}" != "${off[SIGT]}" ]; then
      echo "[ALERT] no progress for $(((now - ${off[SIGT]}) / 60)) min"; off[SIGW]=${off[SIGT]}
    fi
  fi
  save_state; fresh=0
  if done_all; then echo "[monitor] persona, K and Alpaca all DONE -- monitor exits"; exit 0; fi
  [ $(($(date +%s) - start)) -ge "$LIFE" ] && { echo "[monitor] lifetime reached -- re-arm"; exit 0; }
  i=$((i + 1)); sleep 60
done
