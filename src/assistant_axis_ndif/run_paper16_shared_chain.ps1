<#
  paper16 on a SHARED question set: generate -> judge -> vectors -> axes, then copy to Drive.

  Why this exists: in run-paper16 the deception/honesty personas answered the 10 deception
  questions and the sycophancy/anti-sycophancy personas answered the 12 sycophancy questions.
  The unified-axis PCA then mostly separates personas by QUESTION SET (PC1 cos -0.98 with the
  question-family direction at layer 40), not by harmfulness. Mahadik & Skapars (Appendix C)
  build the unified axis on one balanced set every persona answers: the 10 deception questions
  plus the first 10 sycophancy questions. That set is pc_questions_shared.jsonl.

  All 26 personas (25 + default) x 5 prompt variants x 20 questions = 2,600 rollouts.
  Output: run-paper16-shared/ (never run-paper16/). Every stage is resumable.

  Run as a Scheduled Task on the lab PC (register_paper16_shared_task.ps1), never over SSH.
#>
$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$py = Join-Path $repo ".venv-aa\Scripts\python.exe"
$out = "run-paper16-shared"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir, $out | Out-Null
$log = Join-Path $logDir "paper16_shared_chain.log"
$status = Join-Path $logDir "paper16_shared.status"

function Log($m) {
    $line = "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m"
    $line | Out-File -Append -Encoding utf8 $log
}
function Run($stageLog, [string[]]$pyArgs) {
    & $py @pyArgs 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 (Join-Path $logDir $stageLog)
    return $LASTEXITCODE
}

"START " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Encoding utf8 $status
Log "=== paper16 shared-question chain started (pid $PID) ==="

$roles = "$D\pc_role_groups.json"
$instr = "$D\pc_instructions"
$qfile = "$D\pc_questions_shared.jsonl"
foreach ($p in @($roles, $instr, $qfile)) {
    if (-not (Test-Path $p)) { Log "MISSING $p -- stopping"; "FAILED missing inputs" | Out-File -Append -Encoding utf8 $status; exit 1 }
}
$nInstr = (Get-ChildItem "$instr\*.json").Count
$nQ = (Get-Content $qfile | Where-Object { $_.Trim() }).Count
Log "inputs: $nInstr persona files, $nQ shared questions"
if ($nInstr -lt 26 -or $nQ -ne 20) { Log "expected 26 persona files and 20 questions -- stopping"; exit 1 }

# ---- preflight (warn only: a COLD subject model deploys on first request) ------------
$code = Run "paper16_shared_preflight.log" @("$D\0_preflight_ndif.py", "--warn-only")
Log "preflight exit $code"

# ---- stage 1: generation, all groups in one pass ---------------------------------------
Log "generating 26 personas x 5 variants x 20 shared questions"
$code = Run "paper16_shared_generate.log" @("$D\1_generate_ndif.py",
    "--model", "meta-llama/Llama-3.3-70B-Instruct",
    "--roles_dir", $instr, "--roles_file", $roles,
    "--questions_file", $qfile, "--question_count", "20",
    "--output_dir", "$out\responses", "--gen_batch", "96", "--session_batches", "8")
$n = (Get-ChildItem "$out\responses\*.jsonl" -EA SilentlyContinue).Count
$rows = 0
foreach ($f in (Get-ChildItem "$out\responses\*.jsonl" -EA SilentlyContinue)) { $rows += (Get-Content $f.FullName | Where-Object { $_.Trim() }).Count }
Log "generation exit ${code}: $n role files, $rows rollouts (expect 26 / 2600)"
"GENERATED $n files $rows rollouts " + (Get-Date -Format "HH:mm:ss") | Out-File -Append -Encoding utf8 $status
if ($n -lt 26 -or $rows -lt 2500) { Log "generation incomplete -- stopping before judging (re-run resumes)"; exit 1 }

# ---- stage 3: judge (same judge and batch as run-paper16) ------------------------------
Log "judging with meta-llama/Llama-3.1-70B-Instruct"
$code = Run "paper16_shared_judge.log" @("$D\3_judge_ndif.py",
    "--responses_dir", "$out\responses", "--roles_dir", $instr,
    "--roles_file", $roles, "--output_dir", "$out\scores",
    "--judge_model", "meta-llama/Llama-3.1-70B-Instruct",
    "--judge_batch", "16", "--session_batches", "8")
Log "judge exit ${code}: $((Get-ChildItem "$out\scores\*.json" -EA SilentlyContinue).Count) score files (default is not judged)"

# ---- stage 2: persona vectors, unfiltered and score-3 ----------------------------------
foreach ($v in @(@{ tag = "vectors_all"; keep = "0,1,2,3" }, @{ tag = "vectors_s3"; keep = "3" })) {
    Log "extracting $($v.tag)"
    $code = Run "paper16_shared_$($v.tag).log" @("$D\2_activations_ndif.py",
        "--model", "meta-llama/Llama-3.3-70B-Instruct",
        "--responses_dir", "$out\responses", "--scores_dir", "$out\scores",
        "--output_dir", "$out\$($v.tag)", "--roles_file", $roles,
        "--keep_scores", $v.keep, "--min_count", "20", "--max_length", "8192",
        "--layers", "all", "--batch_size", "8", "--session_batches", "20")
    Log "$($v.tag) exit ${code}: $((Get-ChildItem "$out\$($v.tag)\*.pt" -EA SilentlyContinue).Count) vectors"
}

# ---- stage 6: axes ---------------------------------------------------------------------
foreach ($tag in @("vectors_all", "vectors_s3")) {
    if ((Get-ChildItem "$out\$tag\*.pt" -EA SilentlyContinue).Count -lt 26) {
        Log "$tag has fewer than 26 vectors -- axes not built"
        continue
    }
    $code = Run "paper16_shared_axes.log" @("$D\6_axes.py", "--vectors_dir", "$out\$tag",
        "--output", "$out\axes_$tag.pt", "--roles_file", $roles, "--report_layer", "40")
    Log "axes_$tag exit $code"
}

# ---- Drive ---------------------------------------------------------------------------
$rc = Join-Path $repo "tools\rclone.exe"
if (-not (Test-Path $rc)) { $rc = "C:\Users\user4\tools\rclone.exe" }
if (Test-Path $rc) {
    & $rc copy $out "tamara:expLB/paper16-shared" --checksum --transfers 4 `
        --log-file (Join-Path $logDir "paper16_shared_rclone.log") --log-level INFO
    Log "rclone exit $LASTEXITCODE"
}
"DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
Log "=== chain finished ==="
