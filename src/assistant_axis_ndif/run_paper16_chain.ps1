<#
  Wait for the running deception generation to finish, then start the sycophancy half.

  Exists so both halves of the paper16 bank complete unattended overnight. Claude only
  runs while a conversation turn is active, so nothing supervises this -- it is a plain
  detached process. Everything it calls is resumable per (role, prompt_index,
  question_index), so if the laptop sleeps or this is killed, re-running the same command
  continues rather than restarting.

  Launch detached:
      Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
          '-File','src\assistant_axis_ndif\run_paper16_chain.ps1' -WindowStyle Hidden
#>
$ErrorActionPreference = "Continue"
Set-Location "C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"

$py  = ".\.venv-aa-win\Scripts\python.exe"
$gen = "src\assistant_axis_ndif\1_generate_ndif.py"
$log = "logs\paper16_chain.log"

function Log($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Add-Content $log }

Log "chain started (pid $PID); waiting for the deception half to finish"

# Wait for any running generation to exit. Poll rather than Wait-Process because the venv
# python is a trampoline that re-execs, so the pid we would wait on is not the one that
# lasts (documented in the lab-PC notes).
while ($true) {
    $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
         Where-Object { $_.CommandLine -match '1_generate_ndif' }
    if (-not $p) { break }
    Start-Sleep -Seconds 30
}
Log "deception half done; starting sycophancy"

& $py $gen `
    --model meta-llama/Llama-3.3-70B-Instruct `
    --roles_dir src\assistant_axis_ndif\pc_instructions `
    --roles_file src\assistant_axis_ndif\pc_role_groups.json `
    --groups sycophancy anti_sycophancy `
    --questions_file src\assistant_axis_ndif\pc_questions_sycophancy.jsonl `
    --question_count 12 `
    --output_dir run-paper16\responses `
    --gen_batch 96 --session_batches 8 *>> "logs\paper16_sycophancy.log"

Log "sycophancy generation exited with code $LASTEXITCODE"

$n = (Get-ChildItem "run-paper16\responses\*.jsonl" -ErrorAction SilentlyContinue).Count
Log "responses on disk: $n role files (expect 26: 16 deception-side + 10 sycophancy-side)"
Log "chain finished"
