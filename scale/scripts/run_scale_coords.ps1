<#
  NEXT.md item 3 / OPEN_DECISIONS A4: evaluation-corpus activations for the 11-dataset grid at
  3B and 8B, so the scale ladder can produce AUROCs and not just persona geometry.

  WHY THIS IS THE JOB WORTH DOING. Until now the rungs hold persona vectors and axes only --
  every activation store (passA/B/C/P) is 70B. So we can say the persona GEOMETRY replicates at
  8B (cos(PC1,assistant) +0.732, sycophancy contrast 0.983) and cannot produce a single AUROC at
  any size but 70B, which is the number the paper actually reports.

  What it buys is NOT significance: each rung faces the same 11-dataset bootstrap ceiling
  independently (F-04). What it buys is THREE LAYER PROFILES instead of one. F-34 found the
  kd_s23 gain is smooth across depth -- negative L0-L20, peak L44, sustained to L76, 16/21
  layers positive -- and still only reaches q=0.130. If that same shape appears at 3B and 8B,
  the evidence becomes replication ACROSS MODELS, which sidesteps the dataset ceiling because
  the claim is no longer a p-value on one model.

  ORDER IS 8B FIRST, DELIBERATELY. The 8B arm is ~20 min and runs on a different NDIF
  deployment from the 3B, so it can start while Scale-3b is still finishing vectors_s23. By the
  time it is done Scale-3b has drained and the 3B arm gets an uncontended deployment.

  BATCHING IS RUNG-SPECIFIC AND THE 3B IS THE FRAGILE ONE. F-33: the 3B failed 24 of 24 roles at
  batch_size 8 because NDIF's per-PROCESS allowance leaves ~140 MiB once 3B weights are loaded.
  F-35: at batch 2 it runs at 1.32 rows/s against the 8B's 12.9 -- ten times slower while
  extracting FEWER layers. Smaller is slower here, and any estimate assuming otherwise is wrong.

  EVAL ROWS ARE LONGER THAN PERSONA ROLLOUTS, so F-35's projection (~20 min / ~3.2 h) is a
  FLOOR, not a forecast. attn_budget is set low for the 3B for the same reason: the per-call
  limit is rows x heads x T^2, not tokens, so one long row can cost more than sixteen short ones.
#>
param([int]$MaxWaitHours = 4, [int]$PollSeconds = 120)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$pyAA = Join-Path $repo ".venv-aa\Scripts\python.exe"
$SRC = Join-Path $repo "src\assistant_axis_ndif"
$logDir = Join-Path $repo "logs"
$status = Join-Path $logDir "scale_coords.status"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Append -Encoding utf8 $status

# SCOPED TO THE PAPER'S OWN DECEPTION BENCHMARK, 2026-09-20 18:40.
# These five ARE Mahadik & Skapars' Table 1 deception set -- AILiar, Roleplaying, ConvGame,
# InstrDec, HarmPressureChoice -- under our names. The rungs exist to compare against the
# paper, so extracting our other six datasets at 3B/8B buys nothing for that comparison and
# costs 83% of the runtime.
#
# --sample 800 matches the paper's own scale rather than shrinking below it. Table 1 splits:
#   AILiar 120/40  ConvGame 335/113  InstrDec 599/201  HarmPressureChoice 599/201
#   Roleplaying 444/150                                   -> at most 800 rows per dataset
# We were extracting all 14,718 rows only because the 70B run did. Stratified 800 gives ~2,508
# rows here, 17% of the corpus, because three of the five are already under 800.
#
# F-45: after the per-process-cap fixes throughput fell to ~57 rows/min (8 rows/chunk, from 64),
# so the full-corpus 8B arm alone projected to 4.3 h and the 3B did not fit the 20-hour limit.
# This scoping is what makes the job finish rather than a compromise on it.
$DS = @("instructed-deception", "harm-pressure-choice", "convincing-game", "ai_liar",
        "roleplaying")

# 8B first (see header). Each rung: model, output root, batch, attention budget, session size.
# attn budgets REVISED 2026-09-20 17:35 after the 8B arm OOMed on harm-pressure-knowledge-report:
#   "session attempt 4/4 (2 batches, 15072 padded tokens) failed:
#    OutOfMemoryError: CUDA out of memory. Tried to allocate 902.00 MiB"
# The first two datasets (instructed-deception 5,432 rows, harm-pressure-choice 2,134) completed
# fine at attn=2.0; HP-KR is WMDP-derived and its rows run to the 2,048 max_length, so the SAME
# budget buys far fewer rows per batch. The per-call cost is rows x heads x T^2, not tokens --
# at 32 heads and T=2048 a 4-row batch is ~2.1 GiB, which is exactly the budget it was given.
# Halving the budget is the only lever that responds to T^2. Extraction is resumable at chunk
# granularity ("leaving its rows for resume"), so the 119 completed chunks are not redone.
$rungs = @(
    # REVISED AGAIN 17:48. attn=0.75 was still too high: 57 OOMs and 14 permanently-failed
    # batches in 14 minutes, HP-KR writing 4 chunks then nothing. Sized from the actual cost
    # instead of halving again -- one row of attention is heads x T^2 x 4 bytes:
    #   8B, 32 heads, T=2048  ->  32 * 2048^2 * 4 = 537 MiB   -> budget 0.55 GiB admits ONE row
    #   3B, 24 heads, T=2048  ->  24 * 2048^2 * 4 = 402 MiB   -> budget 0.42 GiB admits ONE row
    # Batch 1 is the floor, so this is the smallest call the batcher can build at full length;
    # shorter rows still batch up because the bound is on T^2, not on row count. max_length
    # stays 2048 deliberately: the 70B passB ran at 8192 with the DEFAULT budget, and truncating
    # only the small rungs would make the layer profiles non-comparable across model size --
    # which is the entire reason this job exists.
    # REVISED 18:03. The attn budget was never the binding constraint -- it was being respected.
    # After 0.55 the failed allocations fell to 354-438 MiB (from 902), and it STILL OOMed,
    # because the log says "3 batches, 15904 padded tokens": nnsight bundles several batches
    # into ONE session and they are in flight together. Three ~400 MiB batches is ~1.2 GiB.
    # `session_batches` counts batches and the cost is batches x per-batch cost -- the standing
    # note in the nnsight skill, which I applied to batch size twice before applying it here.
    # So: one batch per session, and a session token cap well under the 16,000 default that
    # 15,904 was clearly being filled up to.
    # TRUNCATION, 18:05, as a STATED LIMITATION rather than a quiet parameter change.
    # The OOM message settles what three rounds of shrinking could not:
    #   "Tried to allocate 562.00 MiB. GPU 0 has a total capacity of 139.80 GiB
    #    of which 55.13 GiB is free."
    # 55 GiB free and a 562 MiB allocation fails -- this is F-21's per-PROCESS cap, not GPU
    # pressure, so no batch or session size reaches it. The only lever left is T, because the
    # single largest allocation is attention at heads x T^2 x 4:
    #   8B, 32 heads: T=2048 -> 537 MiB (fails) ; T=1024 -> 134 MiB
    #   3B, 24 heads: T=2048 -> 402 MiB          ; T=1024 -> 101 MiB
    #
    # WHAT IT COSTS, measured rather than assumed. HP-KR assistant spans are p50 5 tokens,
    # p99 13, max 70 -- NOT ONE exceeds 1024. The sequence is almost entirely context, and
    # build_items keeps the TAIL on over-long rows, so every response span we pool over
    # survives whole. What is lost is CONTEXT the response tokens attended to, which changes
    # the activations without removing the span.
    #
    # THE LIMITATION: the 70B passB ran at max_length 8192. These rungs run at 1024, so
    # absolute AUROCs are NOT comparable across model size. The layer-profile SHAPE still is,
    # and shape is what this job exists to replicate (see header). A clean absolute comparison
    # would need the 70B re-extracted at 1024.
    @{ tag = "8b"; model = "meta-llama/Llama-3.1-8B-Instruct"; batch = "8"; attn = "0.55"; sess = "1"; stok = "6000"; maxlen = "1024" },
    @{ tag = "3b"; model = "meta-llama/Llama-3.2-3B-Instruct"; batch = "2"; attn = "0.42"; sess = "1"; stok = "4000"; maxlen = "1024" }
)

foreach ($r in $rungs) {
    $out = Join-Path $repo "run-scale\$($r.tag)\coords"
    Note "$($r.tag): coords start  model=$($r.model) batch=$($r.batch) attn=$($r.attn) sess=$($r.sess)"

    # The 3B arm waits for Scale-3b so the two do not contend for the same NDIF deployment.
    if ($r.tag -eq "3b") {
        $deadline = (Get-Date).AddHours($MaxWaitHours)
        while ((Get-ScheduledTask -TaskName "Scale-3b" -EA SilentlyContinue).State -eq "Running") {
            if ((Get-Date) -gt $deadline) { Note "3b: gave up waiting for Scale-3b"; break }
            Start-Sleep -Seconds $PollSeconds
        }
        Note "3b: Scale-3b is $((Get-ScheduledTask -TaskName 'Scale-3b' -EA SilentlyContinue).State); proceeding"
    }

    $a = @("$SRC\8_coords_ndif.py", "--model", $r.model, "--datasets") + $DS + @(
         "--output_dir", $out, "--layers", "every4", "--pooling", "mean", "last",
         "--batch_size", $r.batch, "--attn_budget_gib", $r.attn,
         "--session_batches", $r.sess, "--session_tokens", $r.stok, "--max_length", $r.maxlen,
         "--sample", "800", "--seed", "42")
    & $pyAA @a 2>&1 | ForEach-Object { "$_" } |
        Out-File -Append -Encoding utf8 (Join-Path $logDir "scale_coords_$($r.tag).log")
    $n = @(Get-ChildItem $out -Directory -EA SilentlyContinue).Count
    Note "$($r.tag): coords exit $LASTEXITCODE  ($n dataset dirs written, expect $($DS.Count))"
    if ($n -lt $DS.Count) { Note "$($r.tag): INCOMPLETE -- $($DS.Count - $n) datasets missing" }
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $D "sync_scale.ps1") `
    -OnBusy Wait -Tag "coords/final"
Note "DONE"
