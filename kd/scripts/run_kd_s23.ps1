<#
  The s23 arm: persona vectors from rollouts scoring 2 OR 3, not 3 alone.

  WHY. Judge filtering at score 3 selects for persona drift, and it does so ASYMMETRICALLY.
  Measured on this bank (kd/scripts/verify_s23.py, 2026-09-20):

      group                 score-3/role   scores-2+3/role
      knowledge_denial          116.8           146.8
      sincere_ignorance          89.0           150.0
      plain_assertion            73.3           149.0

  So the contrast mean(feigned) - mean(sincere) rests on a 1.31x sample imbalance under s3,
  and 0.98x under s23. The imbalance is not random: the honest personas lose rollouts to
  score 2 ("identifies as itself but exhibits some attributes") because "answer plainly
  without softening" IS close to default assistant behaviour. Filtering at 3 therefore keeps
  the honest rollouts that drifted FURTHEST from the Assistant, which is exactly the
  confound DESIGN.md section 7 control 9 warns about -- the axis can re-learn drift.

  Only 16 of the 1,500 new-group rollouts scored 0 or 1, so s23 discards the 1.1% that
  genuinely broke character and keeps the ~30% that were merely partial. It is also CLOSER
  to the replicated paper, which applies no judge filtering at all.

  Runs after KD-Chain finishes so NDIF is uncontended, then extracts, builds axes, and
  scores the 11-dataset grid with the s23 bank. s3 and vectors_all remain as the robustness
  pair; the disagreement between the three banks is a measurement, not a worry.
#>
param([int]$MaxWaitHours = 10, [int]$PollSeconds = 120)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

# PowerShell variables are CASE-INSENSITIVE. The loop variable `$s` and the source-directory
# variable `$S` were THE SAME VARIABLE, so the first iteration overwrote the path with a
# Hashtable and python was handed a path containing 'System.Collections.Hashtable'.
# Both crossgen steps died with exit 2, while the axes step -- which runs BEFORE the loop --
# succeeded. That is why axes_vectors_s23.pt exists and its matrices do not.
# Renamed to $SRC / $step so the collision cannot recur.
$pyAA = Join-Path $repo ".venv-aa\Scripts\python.exe"
$pyAN = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$SRC = Join-Path $repo "src\assistant_axis_ndif"
$out = Join-Path $repo "run-kd"
$logDir = Join-Path $repo "logs"
$status = Join-Path $logDir "kd_s23.status"
$chainStatus = Join-Path $logDir "kd_chain.status"
$roles = Join-Path $repo "kd\personas\kd_role_groups.json"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

Note "WAITING for KD-Chain to finish (NDIF must be free before extracting)"
$deadline = (Get-Date).AddHours($MaxWaitHours)
while ($true) {
    # Match ANY line, not -Tail 1: run_kd_chain.ps1 writes "DONE <ts>" and then appends
    # "=== chain finished ===" after it, so the last line is never the DONE marker. Checking
    # the tail alone left this task waiting through a completed chain from 04:26 to 11:47.
    $done = (Test-Path $chainStatus) -and
            ((Get-Content $chainStatus -EA SilentlyContinue) -match '^DONE ').Count -gt 0
    if ($done) { break }
    if ((Get-Date) -gt $deadline) { Note "GAVE UP after $MaxWaitHours h"; exit 2 }
    Start-Sleep -Seconds $PollSeconds
}
Note "KD-Chain done; extracting vectors_s23"

$code = & {
    & $pyAA "$SRC\2_activations_ndif.py" `
        --model "meta-llama/Llama-3.3-70B-Instruct" `
        --responses_dir "$out\responses" --scores_dir "$out\scores" `
        --output_dir "$out\vectors_s23" --roles_file $roles `
        --keep_scores "2,3" --min_count 20 --max_length 8192 `
        --layers all --batch_size 8 --session_batches 20 2>&1 |
        ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 (Join-Path $logDir "kd_vectors_s23.log")
    $LASTEXITCODE
}
$n = (Get-ChildItem "$out\vectors_s23\*.pt" -EA SilentlyContinue).Count
Note "vectors_s23 exit ${code}: $n vectors (expect 36)"
if ($n -lt 30) { Note "too few vectors -- stopping"; exit 1 }

& $pyAN "$SRC\6_axes.py" --vectors_dir "$out\vectors_s23" --output "$out\axes_vectors_s23.pt" `
    --roles_file $roles --report_layer 40 2>&1 |
    ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 (Join-Path $logDir "kd_axes_s23.log")
Note "axes_s23 exit $LASTEXITCODE"

$DS = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "harm-pressure-kr-matched", "convincing-game", "insider-trading", "ai_liar",
        "sandbagging_v2", "roleplaying", "mask", "sycophancy_mt")
foreach ($step in @(@{ tag = "peak"; layers = @("40", "44") }, @{ tag = "l20"; layers = @("20") })) {
    Note "crossgen s23 $($step.tag) layers=$($step.layers -join ',')"
    $a = @("$SRC\27_crossgen.py", "--run_dir", "run-coords\passB",
           "--out_root", "run-crossgen\kd11\banks\kd_s23", "--bank_name", "kd_s23",
           "--bank", "$out\axes_vectors_s23.pt", "--baseline_root", "run-crossgen\kd11",
           "--datasets") + $DS + @("--layers") + $step.layers + @(
           "--features", "pc:deception", "axes", "axes_resid", "pc:unified",
                         "pc:knowledge_denial", "pc:kd_assertion",
           "--extra_features", "axes", "pc:knowledge_denial", "pc:kd_assertion",
           "--stage", "all")
    & $pyAN @a 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 (Join-Path $logDir "kd_s23_crossgen.log")
    Note "crossgen s23 $($step.tag) exit $LASTEXITCODE"
}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $D "sync_kd.ps1") -OnBusy Wait -Tag "s23/final"
Note "DONE"
