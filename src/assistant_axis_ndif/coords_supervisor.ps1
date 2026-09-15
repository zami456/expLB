<#
  Keep the coords extraction alive. Same shape as paper16_supervisor.ps1, which carried the
  100-role run for two days, with the paths derived rather than hardcoded.

  Every 5 minutes:
    1. Finished?      -> run-coords\passB\.done exists. Log and exit.
    2. Still running? -> leave it alone.
    3. NDIF ready?    -> relaunch. If not, wait.

  Relaunching is safe: 8_coords_ndif.py counts the chunk_*.pt already written and skips the
  row_ids they cover, so a restart continues rather than redoing work.

  It gates on NDIF before relaunching because the subject model is UNPINNED. During the
  2026-09-15 outage only 7 models came back HOT and every one was pinned; ours came back
  COLD. Throwing a job at a cold model triggers a slow deploy or an error.

      Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
          '-File','src\assistant_axis_ndif\coords_supervisor.ps1' -WindowStyle Hidden
#>
param([string]$Out = "run-coords")
$ErrorActionPreference = "Continue"

$D    = $PSScriptRoot
$Repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $Repo

$py = $null
foreach ($v in @(".venv-aa", ".venv-aa-win")) {
    $cand = Join-Path $Repo "$v\Scripts\python.exe"
    if (Test-Path $cand) { $py = $cand; break }
}
if (-not $py) { Write-Host "no windows venv found under $Repo"; exit 1 }

$log   = Join-Path $Repo "logs\coords_supervisor.log"
$lock  = Join-Path $Repo "logs\coords_supervisor.lock"
$chain = Join-Path $D "run_coords_chain.ps1"
New-Item -ItemType Directory -Force -Path (Join-Path $Repo "logs") | Out-Null
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
Log "=== coords supervisor started (pid $PID) repo=$Repo ==="

function Test-Done { Test-Path (Join-Path $Repo "$Out\passB\.done") }

function Test-ChainAlive {
    # Match the script as it is INVOKED (-File <path>), not any command line that merely
    # mentions it: a bare -match also matches the Get-CimInstance query asking about it,
    # so the check counts ITSELF and reports a chain running when none is.
    $c = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
         Where-Object {
             $_.ProcessId -ne $PID -and
             $_.CommandLine -match '-File\s+\S*run_coords_chain\.ps1' -and
             $_.CommandLine -notmatch 'Get-CimInstance'
         }
    $g = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
         Where-Object { $_.CommandLine -match '8_coords_ndif\.py' }
    return [bool]($c -or $g)
}

$script:lastNdifOk    = $false
$script:lastNdifCheck = [datetime]::MinValue

function Test-NdifReady {
    # Each call costs 30-60s importing torch/nnsight, so a READY answer is cached briefly.
    # A NOT-READY answer is never cached -- that is the state worth noticing when it changes.
    if ($script:lastNdifOk -and ((Get-Date) - $script:lastNdifCheck).TotalMinutes -lt 15) {
        return $true
    }
    & $py "$D\0_preflight_ndif.py" *> (Join-Path $Repo "logs\coords_preflight.log")
    $script:lastNdifOk    = ($LASTEXITCODE -eq 0)   # 0 = READY or WARMING, 2 = NOT READY
    $script:lastNdifCheck = Get-Date
    return $script:lastNdifOk
}

$restarts = 0
while ($true) {
    if (Test-Done) {
        Log "DONE -- run-coords\passB\.done present. See logs\coords_passB.log"
        Remove-Item $lock -Force -ErrorAction SilentlyContinue
        exit 0
    }

    if (-not (Test-ChainAlive)) {
        if (Test-NdifReady) {
            $restarts++
            $n = (Get-ChildItem "$Out\passB\*\chunk_*.pt" -Recurse -EA SilentlyContinue).Count
            Log "chain not alive & not done -> relaunching (restart #$restarts; $n chunks so far)"
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
