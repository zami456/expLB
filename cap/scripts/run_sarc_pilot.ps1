<#
  The deadpan-elicitation pilot, end to end. `cap/docs/SARCASM_DESIGN.md` sequence step 2.

  WHAT IT DECIDES, AND IT IS A GO/NO-GO. Can Llama-3.3-70B produce sarcasm that is NOT marked?
  The literature says probably not (Sarc7: models "slip into blunt reproach"; SarcasmBench: LLMs
  underperform supervised baselines). If it cannot, the on-policy factorial is dead and the
  design moves to teacher-forced identical responses -- accepting Kirch et al.'s point that
  text-ambiguous behaviours are exactly where off-policy data is least reliable.

  300 rollouts only: 3 arms x 5 variants x 20 imprudent-plan questions. Single turn, unlike the
  capitulation bank's teacher-forced two-turn format, because the question here is what the model
  writes unprompted.

  THE FIVE IRONIC VARIANTS ESCALATE the explicitness of the marker ban, from no mention (v1) to
  an exhaustive prohibition (v5). The pilot reports the marker rate PER VARIANT so the answer is
  "which instruction strength works", not just "does it work". If v1 already suppresses markers
  the weaker instruction is preferred, because it is closer to the other banks in this project.

  GATES ASSERT NUMBERS, NOT FILE COUNTS -- three gates on 2026-09-21 passed while measuring
  nothing (one compared a wrong constant, one checked only that a file existed, one read a stale
  file). Each step here reads back the quantity it names.
#>
param([switch]$Force)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

# Do NOT create $d, $src or any other name that differs from an existing variable only by case.
# PowerShell variable names are CASE-INSENSITIVE: $d overwrote $D at 16:00 on 2026-09-21 and
# $src overwrote $SRC at 20:00, each failing every later step in the same second.
$pyAA = Join-Path $repo ".venv-aa\Scripts\python.exe"
$pyAN = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$SRC = Join-Path $repo "src\assistant_axis_ndif"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "sarc_pilot.status"
$run = Join-Path $repo "cap\run-sarc-pilot"
$INSTR = Join-Path $repo "cap\personas\sarcasm_pilot"
$QS = Join-Path $repo "cap\questions\sarcasm_pilot_v1.jsonl"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
function Gate($step, $measured, $predicted, $ok) {
    Note ("{0}  measured={1}  predicted={2}  {3}" -f $step, $measured, $predicted,
          $(if ($ok) { "PASS" } else { "GATE FAILED" }))
}

if ((Test-Path $status) -and -not $Force -and (Select-String -Path $status -Pattern '^DONE' -Quiet)) {
    Note "SKIP already DONE; -Force to re-run"; exit 0
}
"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Encoding utf8 $status

# ---- 1. generate -------------------------------------------------------------------------
Note "1 generate: 3 arms x 5 variants x 20 questions = 300 single-turn rollouts"
& $pyAA "$SRC\1_generate_ndif.py" --roles_dir $INSTR --questions_file $QS `
    --output_dir "$run\responses" --question_count 20 --max_new_tokens 320 `
    --temperature 0.7 *>> (Join-Path $logDir "sarc_pilot_gen.log")
$files = @(Get-ChildItem "$run\responses\*.jsonl" -EA SilentlyContinue)
$rows = ($files | ForEach-Object { (Get-Content $_.FullName | Measure-Object -Line).Lines } |
         Measure-Object -Sum).Sum
Gate "1 generate" "$($files.Count) files / $rows rows" "3 / 300" ($files.Count -eq 3 -and $rows -ge 285)
if ($files.Count -lt 3) { Note "too few rollouts to continue"; exit 1 }

# ---- 2. judge ----------------------------------------------------------------------------
Note "2 judge (Llama-3.1-70B, the same judge as every other bank in the project)"
& $pyAA "$SRC\3_judge_ndif.py" --responses_dir "$run\responses" --output_dir "$run\scores" `
    --roles_dir $INSTR *>> (Join-Path $logDir "sarc_pilot_judge.log")
$sc = @(Get-ChildItem "$run\scores\*.json" -EA SilentlyContinue)
Gate "2 judge" "$($sc.Count) score files" "3" ($sc.Count -ge 3)

# ---- 3. the pilot's actual measurement ----------------------------------------------------
Note "3 marker rate, lexical separability, polarity"
& $pyAN (Join-Path $D "score_sarcasm_markers.py") --responses_dir "$run\responses" `
    --scores_dir "$run\scores" *>> (Join-Path $logDir "sarc_pilot_markers.log")
$rep = Join-Path $run "marker_report.json"
if (Test-Path $rep) {
    $j = Get-Content $rep -Raw | ConvertFrom-Json
    $ir = $j.markers.pc_ironic_critic
    $best = ($ir.per_variant | Where-Object { $_ -ne $null } | Measure-Object -Minimum).Minimum
    $verdict = if ($best -le 0.20) { "PRODUCIBLE -- build the factorial on that variant" }
               elseif ($best -le 0.50) { "PARTIAL -- prefer teacher-forced, filtering selects on the DV" }
               else { "NOT PRODUCIBLE -- factorial must be teacher-forced" }
    Gate "3 markers" ("lowest ironic marker rate={0:N2}" -f $best) "<=0.20 to proceed on-policy" ($best -le 0.20)
    Note "VERDICT $verdict"
    if ($j.lexical) {
        Note ("lexical separability ironic vs sycophant: raw={0:N3} stripped={1:N3} marker effect={2:N3}" -f `
              $j.lexical.auroc_raw, $j.lexical.auroc_stripped, $j.lexical.marker_effect)
        Note ("  near 0.5 stripped = the arms are textually alike, which is the premise the factorial needs")
    }
} else {
    Gate "3 markers" "<no report>" "marker_report.json" $false
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "kd\scripts\sync_kd.ps1") `
    -OnBusy Wait -Tag "sarc_pilot/final" -EA SilentlyContinue
Note "DONE"
