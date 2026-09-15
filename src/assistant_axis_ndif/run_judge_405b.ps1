<#
  Re-judge the paper16 rollouts with Llama-3.1-405B-Instruct (question Q7, judge sensitivity).

  This is a PARALLEL re-judge of the SAME rollouts, never a substitute for the 70B scores.
  The whole paper16-vs-own100 comparison is controlled on the judge, so replacing the judge
  would reintroduce the confound the bank exists to remove. Output goes to scores_405b/ and
  run-paper16/scores/ is left untouched.

  405B is pinned and HOT on NDIF and free, which makes it the right second judge: a genuinely
  different (and much larger) model, no paid key, no new confound beyond the judge itself.
  gpt-4.1-mini is NOT an option here -- NDIF serves open-weight models only; that name appears
  in the plan solely as the judge the Assistant-Axis authors used for their published vectors.

  Conservative batching on purpose. The first overnight judge run produced ZERO scores because
  judge_batch 48 sent 35,040 padded tokens and NDIF rejected every job 0.3s after dispatch.
  The 405B is far heavier than the 70B that limit was measured on, so this starts at half the
  70B's working budget and the scores are checked before trusting the run.

      Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
          '-File','src\assistant_axis_ndif\run_judge_405b.ps1' -WindowStyle Hidden
#>
param(
    [int]$JudgeBatch  = 8,
    [int]$TokenBudget = 6000,
    [int]$SessionBatches = 4
)
$ErrorActionPreference = "Continue"
$D    = $PSScriptRoot
$Repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $Repo

$py = $null
foreach ($v in @(".venv-aa-win", ".venv-aa")) {
    $cand = Join-Path $Repo "$v\Scripts\python.exe"
    if (Test-Path $cand) { $py = $cand; break }
}
if (-not $py) { Write-Host "no windows venv under $Repo"; exit 1 }

New-Item -ItemType Directory -Force -Path (Join-Path $Repo "logs") | Out-Null
$log = Join-Path $Repo "logs\judge_405b.log"
"$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  === 405B re-judge started (pid $PID) ===" |
    Add-Content $log

& $py "$D\3_judge_ndif.py" `
    --responses_dir "run-paper16\responses" `
    --roles_dir "$D\pc_instructions" --roles_file "$D\pc_role_groups.json" `
    --output_dir "run-paper16\scores_405b" `
    --judge_model meta-llama/Llama-3.1-405B-Instruct `
    --judge_batch $JudgeBatch --token_budget $TokenBudget --session_batches $SessionBatches `
    *>> (Join-Path $Repo "logs\judge_405b_raw.log")

$n = (Get-ChildItem "run-paper16\scores_405b\*.json" -ErrorAction SilentlyContinue).Count
"$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  405B re-judge finished, exit $LASTEXITCODE, $n score files" |
    Add-Content $log
