<#
  Tier 0.4 -- the format-matched persona control, end to end.

  WHAT IT DECIDES. F-66/F-68 say the six position-holding personas are load-bearing even though
  F-55 found they produce text nearly identical to no system prompt. The deflationary reading is
  that they contribute a representation of THE INSTRUCTION rather than of a disposition. Three
  conditions, identical except for what the negative-side prompt says:

      cap - holding      the original      (already measured: +0.112 over raw, F-66)
      cap - neutral      format-matched, no stance   <- if this matches, F-66's reading is wrong
      cap - paraphrase   same content, reworded      <- if this matches but neutral does not,
                                                       the CONTENT is the signal and F-66 stands

  ONLY THE TWELVE CONTROL PERSONAS ARE GENERATED. The six capitulation personas and `default`
  are common to all three conditions and were extracted in the original cap run, so their
  rollouts, scores and vectors are reused verbatim. 1,200 rollouts, not 2,600 -- and more
  importantly the capitulation half is IDENTICAL across conditions by construction rather than
  by re-running and hoping.

  GATES ASSERT NUMBERS, NOT FILE COUNTS. Today's runs produced three gates that passed while
  measuring nothing: one compared against the wrong constant, one only checked a file existed,
  one read a stale file from a previous experiment. Each step here reads back the quantity it
  names.
#>
param([switch]$Force)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$pyAA = Join-Path $repo ".venv-aa\Scripts\python.exe"
$pyAN = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$SRC = Join-Path $repo "src\assistant_axis_ndif"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "cap_controls.status"
$run = Join-Path $repo "run-cap-ctrl"
$INSTR = Join-Path $repo "cap\personas\cap_instructions_controls"
$GROUPS = Join-Path $repo "cap\personas\cap_control_groups.json"
$QS = Join-Path $repo "cap\questions\cap_questions_v3.jsonl"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
function Gate($step, $measured, $predicted, $ok) {
    Note ("{0}  measured={1}  predicted={2}  {3}" -f $step, $measured, $predicted,
          $(if ($ok) { "PASS" } else { "GATE FAILED" }))
}

if ((Test-Path $status) -and -not $Force -and (Select-String -Path $status -Pattern '^DONE' -Quiet)) {
    Note "SKIP already DONE; -Force to re-run"; exit 0
}
"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Encoding utf8 $status

$CTRL = @("pc_neutral_alphabetiser", "pc_neutral_citer", "pc_neutral_metric",
          "pc_neutral_speaker", "pc_neutral_speller", "pc_neutral_timekeeper",
          "pc_para_calibrated_holder", "pc_para_disagreement_tolerant", "pc_para_evidence_bound",
          "pc_para_position_holder", "pc_para_reasoned_defender", "pc_para_steadfast_reviewer")

# ---- 1. generate the twelve control personas -------------------------------------------
Note "1 generate: 12 control personas x 5 variants x 20 questions (turn 2 only)"
& $pyAA (Join-Path $D "gen_multiturn_ndif.py") --roles @CTRL --variants 5 --questions $QS `
    --roles_dir $INSTR --out "$run\responses" *>> (Join-Path $logDir "cap_ctrl_gen.log")
$files = @(Get-ChildItem "$run\responses\*.jsonl" -EA SilentlyContinue |
           Where-Object { $_.Name -notlike "*_control*" })
$rows = ($files | ForEach-Object { (Get-Content $_.FullName | Measure-Object -Line).Lines } |
         Measure-Object -Sum).Sum
Gate "1 generate" "$($files.Count) files / $rows rows" "12 / 1200" ($files.Count -eq 12 -and $rows -ge 1150)
if ($files.Count -lt 6) { Note "too few rollouts to continue"; exit 1 }

# ---- 2. judge ---------------------------------------------------------------------------
Note "2 judge (Llama-3.1-70B, the same judge as every other bank in the project)"
& $pyAA "$SRC\3_judge_ndif.py" --responses_dir "$run\responses" --output_dir "$run\scores" `
    --roles_dir $INSTR *>> (Join-Path $logDir "cap_ctrl_judge.log")
$sc = @(Get-ChildItem "$run\scores\*.json" -EA SilentlyContinue)
Gate "2 judge" "$($sc.Count) score files" "12" ($sc.Count -ge 12)

# ---- 3. vectors for the controls, then REUSE the capitulation half ----------------------
Note "3 vectors_s23 for the 12 controls"
& $pyAA "$SRC\2_activations_ndif.py" --responses_dir "$run\responses" `
    --scores_dir "$run\scores" --output_dir "$run\vectors_s23" `
    --keep_scores 2,3 --min_count 20 --layers all `
    *>> (Join-Path $logDir "cap_ctrl_vectors.log")
$nc = @(Get-ChildItem "$run\vectors_s23\*.pt" -EA SilentlyContinue).Count
Gate "3 vectors(new)" "$nc vectors" "12" ($nc -eq 12)

# The capitulation half and the anchor are IDENTICAL across conditions by construction: copy
# the vectors from the original cap run rather than regenerating them. This is the whole reason
# the comparison is clean -- only the negative side differs.
$copied = 0
foreach ($r in @("pc_caver", "pc_deferrer", "pc_appeaser", "pc_second_guesser",
                 "pc_consensus_follower", "pc_conflict_avoider", "default")) {
    $src = Join-Path $repo "run-cap\vectors_s23\$r.pt"
    if (Test-Path $src) { Copy-Item $src (Join-Path "$run\vectors_s23" "$r.pt") -Force; $copied++ }
}
$tot = @(Get-ChildItem "$run\vectors_s23\*.pt" -EA SilentlyContinue).Count
Gate "3 vectors(total)" "$tot vectors ($copied reused from run-cap)" "19" ($tot -eq 19)

# ---- 4. axes: all three control contrasts from one run ----------------------------------
Note "4 axes (cap_vs_neutral, cap_vs_para, cap_only_ctrl)"
& $pyAA "$SRC\6_axes.py" --vectors_dir "$run\vectors_s23" --roles_file $GROUPS `
    --output "$run\axes_s23.pt" *>> (Join-Path $logDir "cap_ctrl_axes.log")
# assert the GEOMETRY, not that a file exists -- the failure this catches is an empty basis table
$cos = @(Select-String -Path (Join-Path $logDir "cap_ctrl_axes.log") `
         -Pattern '^\s+(cap_vs_neutral|cap_vs_para|cap_only_ctrl)\s' -EA SilentlyContinue |
         ForEach-Object { ($_.Line -split '\s+' | Where-Object { $_ })[2] } |
         Where-Object { $_ -as [double] } | ForEach-Object { [double]$_ })
$minCos = if ($cos.Count) { ($cos | Measure-Object -Minimum).Minimum } else { [double]::NaN }
Gate "4 axes" "$($cos.Count) bases, min cos(PC1,contrast)=$([math]::Round($minCos,3))" `
     "3 bases, >=0.90" ((Test-Path "$run\axes_s23.pt") -and $cos.Count -eq 3 -and $minCos -ge 0.90)

# ---- 5. crossgen: the number ------------------------------------------------------------
$DS = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "harm-pressure-kr-matched", "convincing-game", "insider-trading", "ai_liar",
        "sandbagging_v2", "roleplaying", "mask", "sycophancy_mt")
$F = @("raw", "random", "length", "datapca", "random@2", "datapca@2",
       "pc:cap_vs_neutral", "pc:cap_vs_para", "pc:cap_only_ctrl",
       "pc:cap_vs_neutral@2", "pc:cap_vs_para@2", "pc:cap_only_ctrl@2")
if (Test-Path "$run\axes_s23.pt") {
    Note "5 crossgen on the 11-dataset grid, layers 20/40/44"
    & $pyAN "$SRC\27_crossgen.py" --run_dir "run-coords\passB" `
        --out_root "run-crossgen\kd11\banks\cap_ctrl" --bank_name "cap_ctrl" `
        --bank "$run\axes_s23.pt" --baseline_root "run-crossgen\kd11" `
        --datasets @DS --features @F --extra_features @F --extra_probes paper `
        --layers 20 40 44 --stage all *>> (Join-Path $logDir "cap_ctrl_crossgen.log")
    $m = @(Get-ChildItem "run-crossgen\kd11\banks\cap_ctrl\matrices\*.json" -EA SilentlyContinue).Count
    Gate "5 crossgen" "$m matrices" ">=6" ($m -ge 6)
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "kd\scripts\sync_kd.ps1") `
    -OnBusy Wait -Tag "cap_controls/final" -EA SilentlyContinue
Note "DONE"
