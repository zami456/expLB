<#
  Keep the paper16 chain alive overnight.

  The chain itself has no recovery: if a stage dies -- NDIF evicts the model, a session
  drops, the socket hiccups -- it stops where it stood and nothing restarts it. Nobody is
  watching, because Claude only runs while a conversation turn is active. This supervisor
  is the safety net, and it is deliberately the same shape as the one that carried the
  100-role run on the lab PC for two days.

  Every 5 minutes it asks three questions:
    1. Is the work finished?          -> both axes files exist. Log and exit.
    2. Is the chain still running?     -> leave it alone.
    3. Is NDIF ready?                  -> if yes, relaunch the chain. If no, wait.

  Relaunching is safe because every stage is resumable: generation per
  (role, prompt_index, question_index), judging per scored response, vectors per role
  file. A restart continues rather than redoing work.

  It gates on NDIF before relaunching for a specific reason: the subject model is
  UNPINNED. During the 2026-09-15 outage only 7 models came back HOT and every one of
  them was pinned, while ours came back COLD. Throwing a job at a cold model triggers a
  slow deploy or an error, so it is better to wait and re-check.

      Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
          '-File','src\assistant_axis_ndif\paper16_supervisor.ps1' -WindowStyle Hidden
#>
$ErrorActionPreference = "Continue"
Set-Location "C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"

$py    = ".\.venv-aa-win\Scripts\python.exe"
$D     = "src\assistant_axis_ndif"
$out   = "run-paper16"
$log   = "logs\paper16_supervisor.log"
$lock  = "logs\paper16_supervisor.lock"
$chain = "$D\run_paper16_chain.ps1"
New-Item -ItemType Directory -Force -Path logs | Out-Null

function Log($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Add-Content $log }

# one supervisor only -- two would double-launch the chain against each other
if (Test-Path $lock) {
    $old = (Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
        Log "supervisor already running (pid $old); exiting"
        exit 0
    }
    Log "stale lock (pid $old) ignored"
}
$PID | Set-Content $lock
Log "=== supervisor started (pid $PID) ==="

function Test-Done {
    (Test-Path "$out\axes_vectors_all.pt") -and (Test-Path "$out\axes_vectors_s3.pt")
}

function Test-ChainAlive {
    # Match the script as it is INVOKED (-File <path>), not any command line that merely
    # mentions it. A bare -match 'run_paper16_chain' also matches the query asking about it,
    # so a Get-CimInstance check counted ITSELF and reported a chain running when none was.
    # That happened, and sent a diagnosis off in the wrong direction for several minutes.
    $c = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
         Where-Object {
             $_.ProcessId -ne $PID -and
             $_.CommandLine -match '-File\s+\S*run_paper16_chain\.ps1' -and
             $_.CommandLine -notmatch 'Get-CimInstance'
         }
    # Stage scripts are invoked as `python <dir>/N_name_ndif.py`, so require the .py.
    $g = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
         Where-Object { $_.CommandLine -match 'assistant_axis_ndif\S*\.py' }
    return [bool]($c -or $g)
}

$script:lastNdifOk    = $false
$script:lastNdifCheck = [datetime]::MinValue

function Test-NdifReady {
    # Each call costs 30-60s importing torch/transformers/nnsight, so a READY answer is
    # cached briefly. A NOT-READY answer is never cached -- that is the state worth
    # noticing the moment it changes.
    if ($script:lastNdifOk -and ((Get-Date) - $script:lastNdifCheck).TotalMinutes -lt 15) {
        return $true
    }
    & $py "$D\0_preflight_ndif.py" *> "logs\paper16_preflight.log"
    $script:lastNdifOk    = ($LASTEXITCODE -eq 0)   # 0 = READY or WARMING, 2 = NOT READY
    $script:lastNdifCheck = Get-Date
    return $script:lastNdifOk
}

$restarts = 0
while ($true) {
    if (Test-Done) {
        Log "DONE -- both axes files present. Gate output is at the end of logs\paper16_chain.log"
        Remove-Item $lock -Force -ErrorAction SilentlyContinue
        exit 0
    }

    if (-not (Test-ChainAlive)) {
        if (Test-NdifReady) {
            $restarts++
            $n = (Get-ChildItem "$out\responses\*.jsonl" -ErrorAction SilentlyContinue).Count
            Log "chain not alive & not done -> relaunching (restart #$restarts; $n role files so far)"
            Start-Process powershell `
                -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',$chain `
                -WindowStyle Hidden
            Start-Sleep -Seconds 20
        } else {
            Log "chain not alive, but NDIF is NOT READY -- waiting rather than triggering a cold deploy"
        }
    }

    Start-Sleep -Seconds 300
}
