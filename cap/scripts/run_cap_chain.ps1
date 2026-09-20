<#
  The capitulation run, end to end, unattended.

  PRODUCES ONE NUMBER: does a capitulation-derived axis beat 0.374 on sycophancy_mt? The
  register axis reads 0.374 zero-shot / 0.355 trained-as-target there (F-13) -- below chance,
  worse than a random direction. If a capitulation axis beats it, the register hypothesis is
  demonstrated constructively rather than only diagnosed. If not, it stands as a critique.
  Both are reportable; only one is a fix.

  TEACHER-FORCED TURN 1, so only turn 2 is generated -- 1,200 rollouts, not 2,400. Letting the
  personas write turn 1 was measured and failed twice: they hedged before any challenge (F-26),
  then gave outright WRONG answers and "folded" into correctness (F-46). Prefilling makes turn 1
  byte-identical across all twelve personas, so the only varying text in the context is the
  system prompt and the only generated text is what gets measured.

  THREE AXIS VARIANTS from the same rollouts (F-55: the holding personas behave identically to
  no system prompt at all, so which contrast is the real one is an open question):
      A capitulation    cap - position_holding   basis 13
      B cap_vs_default  cap - default            basis 13
      C cap_only        cap - default            basis  7
  A vs B isolates the contrast, B vs C the basis.

  FILTERED AND UNFILTERED BOTH. The judge scores "stayed in character", which means FOLDED for a
  capitulation persona and HELD for a holding one -- so dropping 0s and 1s removes exactly the
  rollouts that disagree with the hypothesis, on both sides, before extraction. If the axis only
  appears after filtering, the filter is the finding.

  GATES DO NOT BLOCK. Every step logs "<step> exit <code>: <gate number>" and continues, except
  where the next step consumes its output. That is the owner's standing instruction: fall
  through, record the number, try the next thing.
#>
param([switch]$SkipGenerate)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$pyAA = Join-Path $repo ".venv-aa\Scripts\python.exe"
$pyAN = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$SRC = Join-Path $repo "src\assistant_axis_ndif"
$CAP = Join-Path $repo "cap"
$logDir = Join-Path $repo "logs"
$status = Join-Path $logDir "cap_chain.status"
$run = Join-Path $repo "run-cap"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
function Gate($step, $measured, $predicted, $ok) {
    $verdict = if ($ok) { "PASS" } else { "GATE FAILED" }
    Note "$step  measured=$measured  predicted=$predicted  $verdict"
}

"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Append -Encoding utf8 $status

$ROLES = @("pc_caver", "pc_deferrer", "pc_appeaser", "pc_second_guesser",
           "pc_consensus_follower", "pc_conflict_avoider",
           "pc_position_holder", "pc_evidence_bound", "pc_steadfast_reviewer",
           "pc_reasoned_defender", "pc_calibrated_holder", "pc_disagreement_tolerant",
           # THE ANCHOR IS NOT OPTIONAL. Leaving it out on 2026-09-21 cost the whole axes step:
           # 6_axes.py needs `default` to build the Assistant Axis, and variants B and C are
           # contrasts AGAINST default, so its absence makes all three unbuildable. The chain
           # still reported 12 vectors as a PASS -- see the gate fix below.
           "default")
$QS = Join-Path $CAP "questions\cap_questions_v3.jsonl"
$GROUPS = Join-Path $CAP "personas\cap_role_groups.json"
$INSTR = Join-Path $CAP "personas\cap_instructions"

# ---- 2a/2b generate -------------------------------------------------------------------
if (-not $SkipGenerate) {
    Note "2a generate: 12 personas x 5 variants x 20 questions (turn 2 only)"
    & $pyAA "$D\gen_multiturn_ndif.py" --roles @ROLES --variants 5 --questions $QS `
        --out "$run\responses" *>> (Join-Path $logDir "cap_gen.log")
    $files = @(Get-ChildItem "$run\responses\*.jsonl" -EA SilentlyContinue |
               Where-Object { $_.Name -notlike "*_control*" })
    $rows = ($files | ForEach-Object { (Get-Content $_.FullName | Measure-Object -Line).Lines } |
             Measure-Object -Sum).Sum
    Gate "2a" "$($files.Count) files / $rows rows" "13 / 1300" ($files.Count -eq 13 -and $rows -ge 1250)

    Note "2b control: turn 1 prefilled WRONG"
    & $pyAA "$D\gen_multiturn_ndif.py" --roles @ROLES --variants 5 --questions $QS --control `
        --out "$run\control" *>> (Join-Path $logDir "cap_gen.log")
    # CONTROL ROLLOUTS LIVE OUTSIDE responses\. 2_activations_ndif.py globs responses\*.jsonl,
    # so a control file sitting there is extracted as if it were another persona --
    # `default_control` became a 14th vector on 2026-09-21 and would have entered both the PCA
    # basis and the role side of the Assistant Axis.
    $cn = @(Get-ChildItem "$run\control\*_control.jsonl" -EA SilentlyContinue).Count
    Gate "2b" "$cn control files" "13" ($cn -eq 13)
}

# ---- 2c judge -------------------------------------------------------------------------
Note "2c judge (Llama-3.1-70B, held constant with every other bank)"
& $pyAA "$SRC\3_judge_ndif.py" --responses_dir "$run\responses" --output_dir "$run\scores" `
    --roles_dir $INSTR *>> (Join-Path $logDir "cap_judge.log")
$sc = @(Get-ChildItem "$run\scores\*.json" -EA SilentlyContinue)
Gate "2c" "$($sc.Count) score files" ">=12" ($sc.Count -ge 12)
if ($sc.Count -lt 1) { Note "2c produced nothing -- vectors cannot run; stopping chain"; exit 1 }

# ---- 2d vectors, FILTERED and UNFILTERED ----------------------------------------------
foreach ($v in @(@{tag = "s23"; keep = "2,3"}, @{tag = "all"; keep = "0,1,2,3"})) {
    Note "2d vectors_$($v.tag)  keep_scores=$($v.keep)"
    & $pyAA "$SRC\2_activations_ndif.py" --responses_dir "$run\responses" `
        --scores_dir "$run\scores" --output_dir "$run\vectors_$($v.tag)" `
        --keep_scores $v.keep --min_count 20 --layers all `
        *>> (Join-Path $logDir "cap_vectors_$($v.tag).log")
    $n = @(Get-ChildItem "$run\vectors_$($v.tag)\*.pt" -EA SilentlyContinue).Count
    Gate "2d/$($v.tag)" "$n vectors" "13" ($n -eq 13)
}

# ---- 2e axes (all three variants come out of one run; they are three contrasts) --------
foreach ($v in @("s23", "all")) {
    $vec = "$run\vectors_$v"
    if (@(Get-ChildItem "$vec\*.pt" -EA SilentlyContinue).Count -lt 12) {
        Note "2e/$v skipped: fewer than 12 vectors"; continue
    }
    Note "2e axes_$v  (contrasts: capitulation, cap_vs_default, cap_only)"
    & $pyAA "$SRC\6_axes.py" --vectors_dir $vec --roles_file $GROUPS `
        --output "$run\axes_$v.pt" *>> (Join-Path $logDir "cap_axes_$v.log")
    # THE GATE IS cos(PC1,contrast) >= 0.90, NOT "the file exists". On 2026-09-21 this step
    # logged written/written/PASS for both filters while 6_axes.py printed an EMPTY basis
    # table -- its report loop was hardcoded to paper16's three names, so the number the gate
    # names was never computed. A gate that cannot fail is not a gate.
    $cosLine = @(Select-String -Path (Join-Path $logDir "cap_axes_$v.log") `
                 -Pattern '^\s+(capitulation|cap_vs_default|cap_only)\s' -EA SilentlyContinue)
    $coss = @($cosLine | ForEach-Object { ($_.Line -split '\s+' | Where-Object { $_ }) [2] } |
              Where-Object { $_ -as [double] } | ForEach-Object { [double]$_ })
    $minCos = if ($coss.Count) { ($coss | Measure-Object -Minimum).Minimum } else { [double]::NaN }
    Gate "2e/$v" "$($coss.Count) bases, min cos(PC1,contrast)=$([math]::Round($minCos,3))" `
        "3 bases, >=0.90" ((Test-Path "$run\axes_$v.pt") -and $coss.Count -eq 3 -and $minCos -ge 0.90)
}

# ---- 2f crossgen: THE NUMBER ----------------------------------------------------------
# An explicit --features list IS required here, and it must still carry every control.
#
# The default list names `pc:deception` and `pc:unified`, which this bank does not have --
# 9_persona_probes.py:137 does bank["pca"]["deception"] directly and raises KeyError, the same
# hardcoding that stopped 6_axes.py an hour earlier. The cap bank's PCAs are `capitulation`,
# `cap_vs_default`, `cap_only` and `all`.
#
# The standing rule is "never drop raw / random / datapca", not "never pass --features": a
# persona feature's number means nothing without the baseline and the matched control it has to
# beat. So this list is the default with the two paper16 names swapped for the three cap
# variants, and every control kept.
# `axes` and `axes_resid` are NOT generic features: 9_persona_probes.py:164 builds them by
# stacking paper16's four axes (deception, sycophancy, unified, assistant). This bank has only
# `assistant` plus its own three, so both raise KeyError. They are dropped, not replaced --
# the cap equivalent of "stack the behaviour axes" is pc:capitulation itself.
$FEATS = @("raw", "random", "length", "datapca", "random@2", "datapca@2",
           "pc:capitulation", "pc:cap_vs_default", "pc:cap_only",
           "pc:capitulation@2", "pc:cap_vs_default@2", "pc:cap_only@2")
$DS = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "harm-pressure-kr-matched", "convincing-game", "insider-trading", "ai_liar",
        "sandbagging_v2", "roleplaying", "mask", "sycophancy_mt")
foreach ($v in @("s23", "all")) {
    $bank = "$run\axes_$v.pt"
    if (-not (Test-Path $bank)) { Note "2f/$v skipped: no axes"; continue }
    Note "2f crossgen_$v on the 11-dataset grid"
    & $pyAN "$SRC\27_crossgen.py" --run_dir "run-coords\passB" `
        --out_root "run-crossgen\kd11\banks\cap_$v" --bank_name "cap_$v" --bank $bank `
        --baseline_root "run-crossgen\kd11" --datasets @DS --features @FEATS `
        --layers 20 40 44 --stage all `
        *>> (Join-Path $logDir "cap_crossgen_$v.log")
    $m = @(Get-ChildItem "run-crossgen\kd11\banks\cap_$v\matrices\*.json" -EA SilentlyContinue).Count
    Gate "2f/$v" "$m matrices" ">=6" ($m -ge 6)
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "kd\scripts\sync_kd.ps1") `
    -OnBusy Wait -Tag "cap/final" -EA SilentlyContinue
Note "DONE"
