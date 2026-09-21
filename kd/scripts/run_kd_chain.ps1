<#
  Phase 3: the kd persona bank. generate -> judge -> vectors -> axes -> Drive.

  36 personas x 5 instruction variants x 30 shared questions = 5,400 rollouts.
  Modelled on run_paper16_shared_chain.ps1, which is the correct template: every stage is
  resumable, and the shared question set is the invariant that keeps the unified PCA from
  collapsing into a question-topic axis (it did once: cos -0.98, see the
  unified-pc1-is-question-family memory entry).

  Output root is run-kd\ -- never run-paper16-shared\, which holds the 26-persona bank this
  one extends.

  Scheduled Task only (register_kd_chain_task.ps1). Start-Process over SSH is killed with the
  session; that cost a crossgen step 24 seconds in on 2026-09-19.
#>
param([switch]$Force)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$py = Join-Path $repo ".venv-aa\Scripts\python.exe"
$S = Join-Path $repo "src\assistant_axis_ndif"
$out = Join-Path $repo "run-kd"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir, $out | Out-Null
$status = Join-Path $logDir "kd_chain.status"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
function Run($stageLog, [string[]]$a) {
    & $py @a 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 (Join-Path $logDir $stageLog)
    return $LASTEXITCODE
}

if (-not $Force -and (Test-Path $status) -and ((Get-Content $status -Tail 1) -match '^DONE ')) {
    Note "SKIP already DONE; -Force to re-run"; exit 0
}
"START " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Encoding utf8 $status

$roles = Join-Path $repo "kd\personas\kd_role_groups.json"
$instr = Join-Path $repo "kd\personas\kd_instructions"
$qfile = Join-Path $repo "kd\questions\kd_questions_shared.jsonl"
foreach ($p in @($roles, $instr, $qfile)) {
    if (-not (Test-Path $p)) { Note "MISSING $p -- stopping"; exit 1 }
}
$nInstr = (Get-ChildItem "$instr\*.json").Count
$nQ = (Get-Content $qfile | Where-Object { $_.Trim() }).Count
Note "inputs: $nInstr persona files, $nQ questions"
if ($nInstr -ne 36 -or $nQ -ne 30) { Note "expected 36 personas and 30 questions -- stopping"; exit 1 }

Note "preflight"
Run "kd_preflight.log" @("$S\0_preflight_ndif.py", "--warn-only") | Out-Null

# ---- stage 1: generation -------------------------------------------------------------
Note "generating 36 x 5 x 30 = 5400 rollouts"
$code = Run "kd_generate.log" @("$S\1_generate_ndif.py",
    "--model", "meta-llama/Llama-3.3-70B-Instruct",
    "--roles_dir", $instr, "--roles_file", $roles,
    "--questions_file", $qfile, "--question_count", "30",
    "--output_dir", "$out\responses", "--gen_batch", "96", "--session_batches", "8")
$n = (Get-ChildItem "$out\responses\*.jsonl" -EA SilentlyContinue).Count
$rows = 0
foreach ($f in (Get-ChildItem "$out\responses\*.jsonl" -EA SilentlyContinue)) {
    $rows += (Get-Content $f.FullName | Where-Object { $_.Trim() }).Count
}
Note "generation exit ${code}: $n role files, $rows rollouts (expect 36 / 5400)"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $D "sync_kd.ps1") -OnBusy Skip -Tag "chain/generate"
if ($n -lt 36 -or $rows -lt 5000) { Note "generation incomplete -- stopping before judging (re-run resumes)"; exit 1 }

# ---- stage 3: judge ------------------------------------------------------------------
Note "judging with meta-llama/Llama-3.1-70B-Instruct"
$code = Run "kd_judge.log" @("$S\3_judge_ndif.py",
    "--responses_dir", "$out\responses", "--roles_dir", $instr,
    "--roles_file", $roles, "--output_dir", "$out\scores",
    "--judge_model", "meta-llama/Llama-3.1-70B-Instruct",
    "--judge_batch", "16", "--session_batches", "8")
Note "judge exit ${code}: $((Get-ChildItem "$out\scores\*.json" -EA SilentlyContinue).Count) score files"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $D "sync_kd.ps1") -OnBusy Skip -Tag "chain/judge"

# ---- stage 2: persona vectors, unfiltered and score-3 --------------------------------
foreach ($v in @(@{ tag = "vectors_all"; keep = "0,1,2,3" }, @{ tag = "vectors_s3"; keep = "3" })) {
    Note "extracting $($v.tag)"
    $code = Run "kd_$($v.tag).log" @("$S\2_activations_ndif.py",
        "--model", "meta-llama/Llama-3.3-70B-Instruct",
        "--responses_dir", "$out\responses", "--scores_dir", "$out\scores",
        "--output_dir", "$out\$($v.tag)", "--roles_file", $roles,
        "--keep_scores", $v.keep, "--min_count", "20", "--max_length", "8192",
        "--layers", "all", "--batch_size", "8", "--session_batches", "20")
    Note "$($v.tag) exit ${code}: $((Get-ChildItem "$out\$($v.tag)\*.pt" -EA SilentlyContinue).Count) vectors"
}

# ---- stage 6: axes -------------------------------------------------------------------
foreach ($tag in @("vectors_all", "vectors_s3")) {
    if ((Get-ChildItem "$out\$tag\*.pt" -EA SilentlyContinue).Count -lt 30) {
        Note "$tag has fewer than 30 vectors -- axes not built"; continue
    }
    $code = Run "kd_axes.log" @("$S\6_axes.py", "--vectors_dir", "$out\$tag",
        "--output", "$out\axes_$tag.pt", "--roles_file", $roles, "--report_layer", "40")
    Note "axes_$tag exit $code"
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $D "sync_kd.ps1") -OnBusy Wait -Tag "chain/final"
"DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
Note "=== chain finished ==="
