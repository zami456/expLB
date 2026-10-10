#!/usr/bin/env bash
# Watch the random persona bank run; every stdout line is one event for Claude's Monitor tool.
#
#   every 60 s   new lines in queue.status, each model's chain.status, Llama's K and Alpaca status, and failed
#                Drive copies (sync.status)
#   every 5 min  new ERROR / Traceback lines in the running stages' .err.log files (one summary line per file);
#                what is alive (rpb_procs.ps1) -- an ALERT when nothing runs while work remains, when the task is not
#                Ready/Running, or when free RAM < 600 MB (only on a change, not every 5 min)
#   every 10 min one progress line for the model in progress; an ALERT if its counts have not moved for 45 min
#
# Built for this laptop (memory note monitor-orphans-on-windows): it exits on its own after $LIFE seconds, before
# the Monitor's 30-minute cap, and a token file makes an older copy exit when a newer one starts. Read offsets are
# kept in results/_queue/monitor.state, so a re-armed copy reports everything since the last one.
#
#   bash random_persona_bank/scripts/rpb_monitor.sh [life_seconds]
R=/c/Users/SURAIYA/Documents/STUDY/thesis/expLB/random_persona_bank/results
Q=$R/_queue
PROCS='C:\Users\SURAIYA\Documents\STUDY\thesis\expLB\random_persona_bank\scripts\rpb_procs.ps1'
STATE=$Q/monitor.state
TOKEN=$Q/monitor.token
LIFE=${1:-1740}
MODELS="Llama-3.3-70B-Instruct Qwen2.5-72B-Instruct Mistral-Small-24B-Instruct-2501 Mistral-Small-3.1-24B-Instruct-2503 gemma-3-27b-it"
L=$R/Llama-3.3-70B-Instruct
ERR='ERROR|Traceback|aborted|permanently failed|Exception|Killed|MemoryError'

me="$$-$(date +%s)"
echo "$me" > "$TOKEN"
start=$(date +%s)

declare -A off
fresh=1
if [ -f "$STATE" ]; then
  fresh=0
  while IFS=$'\t' read -r k v; do [ -n "$k" ] && off["$k"]=$v; done < "$STATE"
fi
save_state() {
  : > "$STATE.tmp"
  for k in "${!off[@]}"; do printf '%s\t%s\n' "$k" "${off[$k]}" >> "$STATE.tmp"; done
  mv -f "$STATE.tmp" "$STATE"
}
clean() { tr -d '\r' | sed 's/^\xEF\xBB\xBF//'; }

# every new line of a status file is an event (a file first seen after the first run is shown whole)
watch_lines() {  # file label [grep-filter]
  local f=$1 label=$2 filt=${3:-.}
  [ -f "$f" ] || return
  local n o
  n=$(clean < "$f" | wc -l)
  o=${off["L:$f"]:-}
  if [ -z "$o" ]; then if [ "$fresh" = 1 ]; then o=$n; else o=0; fi; fi
  [ "$n" -lt "$o" ] && o=0
  if [ "$n" -gt "$o" ]; then
    clean < "$f" | sed -n "$((o + 1)),${n}p" | grep -E -- "$filt" | sed "s/^/[$label] /"
  fi
  off["L:$f"]=$n
}

watch_errors() {  # dir label
  local f n o last
  for f in "$1"/*.err.log; do
    [ -f "$f" ] || continue
    n=$(grep -cE "$ERR" "$f" 2>/dev/null); n=${n:-0}
    o=${off["E:$f"]:-}
    if [ -z "$o" ]; then if [ "$fresh" = 1 ]; then o=$n; else o=0; fi; fi
    [ "$n" -lt "$o" ] && o=0          # the chain overwrites a stage's logs on each attempt
    if [ "$n" -gt "$o" ]; then
      last=$(grep -E "$ERR" "$f" | tail -1 | clean | cut -c1-240)
      echo "[$2 errors] $((n - o)) new in $(basename "$f"): $last"
    fi
    off["E:$f"]=$n
  done
}

done_line() { tail -1 "$1" 2>/dev/null | clean; }

# the piece of work in progress -> one line; sets SIG (counts) for the stall check, and ALLDONE
progress() {
  ALLDONE=0; SIG=""
  local m cs last rep sco v b
  for m in $MODELS; do
    cs=$R/$m/logs/chain.status
    last=$(done_line "$cs")
    if [[ "$last" == DONE* ]]; then
      if [ "$m" = "Llama-3.3-70B-Instruct" ]; then
        if ! [[ "$(done_line "$L/K/k_chain.status")" =~ \ \ DONE$ ]]; then
          SIG="K $(ls "$L"/K/llama-70b-3.3/*/parts_answer 2>/dev/null | wc -l) $(done_line "$L/K/k_chain.status")"
          echo "[progress] Llama K: $(done_line "$L/K/k_chain.status" | cut -c1-160)"; return
        fi
        if ! [[ "$(done_line "$L/logs/alpaca.status")" =~ \ \ DONE$ ]]; then
          SIG="A $(ls "$L"/eval/alpaca/chunk_*.pt 2>/dev/null | wc -l)"
          echo "[progress] Llama Alpaca read: $(ls "$L"/eval/alpaca/chunk_*.pt 2>/dev/null | wc -l) chunks | $(done_line "$L/logs/alpaca.status" | cut -c1-120)"; return
        fi
      fi
      continue
    fi
    if [ ! -f "$cs" ]; then echo "[progress] next up: $m (its chain has not started)"; SIG="wait $m"; return; fi
    rep=$(cat "$R/$m"/persona/responses/*.jsonl 2>/dev/null | wc -l)
    sco=$(cat "$R/$m"/persona/scores/*.json 2>/dev/null | grep -o '"pos_p' | wc -l)
    v=""
    for b in vectors_s23 vectors_all vectors_s3; do v="$v ${b#vectors_}=$(ls "$R/$m/persona/$b"/*.pt 2>/dev/null | wc -l)"; done
    SIG="$m $rep $sco $v"
    echo "[progress] $m: replies $rep/5400, scored $sco/5250, vectors$v | ${last:21:110}"
    return
  done
  ALLDONE=1
}

last_proc=${off[PROC]:-}      # kept in the state file, so a re-armed copy does not re-announce a standing alert
check_procs() {
  local s q c t mem
  s=$(powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$PROCS" 2>/dev/null | clean | tail -1)
  [ -z "$s" ] && return
  q=$(sed -n 's/.*queue=\([0-9]*\).*/\1/p' <<<"$s"); c=$(sed -n 's/.*chain=\([0-9]*\).*/\1/p' <<<"$s")
  t=$(sed -n 's/.*task=\([A-Za-z]*\).*/\1/p' <<<"$s"); mem=$(sed -n 's/.*freeMB=\([0-9]*\).*/\1/p' <<<"$s")
  local alert=""
  if [ "$ALLDONE" != 1 ] && [ "${q:-0}" = 0 ] && [ "${c:-0}" = 0 ]; then alert="nothing running while work remains"; fi
  if [ "$t" != "Ready" ] && [ "$t" != "Running" ]; then alert="${alert:+$alert; }task RPB-queue is $t"; fi
  if [ -n "$mem" ] && [ "$mem" -lt 600 ]; then alert="${alert:+$alert; }free RAM ${mem} MB"; fi
  # compare the alert's KIND, not its numbers: "free RAM 466 MB" and "free RAM 379 MB" are one condition
  # (2026-10-10: the changing MB value re-fired the alert every 5 minutes during the judge and vector stages)
  local kind; kind=$(sed 's/[0-9]//g' <<<"$alert")
  if [ -n "$alert" ] && [ "$kind" != "$last_proc" ]; then echo "[ALERT] $alert ($s)"; fi
  if [ -z "$alert" ] && [ -n "$last_proc" ]; then echo "[ok] cleared ($s)"; fi
  last_proc=$kind
  off[PROC]=$kind
}

stall_check() {
  local now; now=$(date +%s)
  if [ "$SIG" != "${off[SIG]:-}" ]; then off[SIG]=$SIG; off[SIGT]=$now; return; fi
  local since=${off[SIGT]:-$now} warned=${off[SIGW]:-0}
  if [ $((now - since)) -ge 2700 ] && [ "$warned" != "$since" ]; then
    echo "[ALERT] no progress for $(((now - since) / 60)) min: $SIG"
    off[SIGW]=$since
  fi
}

progress
echo "[monitor] armed (pid $$, exits after $((LIFE / 60)) min, $( [ "$fresh" = 1 ] && echo first run || echo resuming from saved offsets ))"
stall_check
i=0
while :; do
  [ "$(cat "$TOKEN" 2>/dev/null)" = "$me" ] || exit 0
  watch_lines "$Q/queue.status" queue
  for m in $MODELS; do
    watch_lines "$R/$m/logs/chain.status" "${m%%-*}"
    watch_lines "$R/$m/logs/sync.status" "${m%%-*} drive" 'END exit [^0]|GAVE UP|NO RCLONE'
  done
  watch_lines "$L/K/k_chain.status" "Llama K"
  watch_lines "$L/logs/alpaca.status" "Llama Alpaca"
  if [ $((i % 5)) = 0 ]; then
    for m in $MODELS; do watch_errors "$R/$m/logs" "${m%%-*}"; done
    watch_errors "$Q/llama-K-logs" "Llama K"
    watch_errors "$Q/llama-alpaca-logs" "Llama Alpaca"
    progress > /dev/null; check_procs
  fi
  if [ $((i % 10)) = 0 ] && [ "$i" -gt 0 ]; then progress; stall_check; fi
  save_state
  fresh=0
  if [ "$ALLDONE" = 1 ]; then echo "[monitor] all five models and the Llama steps are DONE -- monitor exits"; exit 0; fi
  [ $(($(date +%s) - start)) -ge "$LIFE" ] && { echo "[monitor] lifetime reached -- re-arm"; exit 0; }
  i=$((i + 1))
  sleep 60
done
