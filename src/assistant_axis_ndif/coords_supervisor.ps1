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

# ---- Drive sync -----------------------------------------------------------------------
# The lab PC is a single copy of a ~10 GB download that costs hours of NDIF time to rebuild,
# and it has rebooted unprompted before. Chunks go to Drive as they land rather than in one
# upload at the end, so an interruption costs minutes rather than the whole pass.
#
# Safe to run against a live output dir: 8_coords_ndif.py writes each chunk to .tmp and
# os.replace()s it into place, so a chunk_*.pt that exists is always complete. --include
# *.pt also skips the .tmp files by construction.
#
# Size+modtime matching, not --checksum: hashing a growing 10 GB tree every 5 minutes costs
# more than it catches. The end-of-run verification below is the checksum pass.
function Get-Rclone {
    # $Repo first, and deliberately. Under the S4U scheduled task neither PATH nor
    # $env:USERPROFILE resolved to a copy of rclone -- the supervisor logged
    # "rclone not found" while rclone.exe sat in the invoking user's profile -- so the only
    # dependable location is one derived from $PSScriptRoot, like everything else here.
    $cands = @((Join-Path $Repo "tools\rclone.exe"),
               (Join-Path $env:USERPROFILE "tools\rclone.exe"))
    foreach ($p in $cands) { if ($p -and (Test-Path $p)) { return $p } }
    $c = Get-Command rclone -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    return $null
}

$script:warnedNoRclone = $false

function Sync-Drive([switch]$Verify) {
    $rc = Get-Rclone
    if (-not $rc) {
        if (-not $script:warnedNoRclone) { Log "rclone not found -- Drive sync disabled"; $script:warnedNoRclone = $true }
        return
    }
    $src = Join-Path $Repo $Out
    if (-not (Test-Path $src)) { return }
    $rlog = Join-Path $Repo "logs\coords_rclone.log"
    $extra = @()
    if ($Verify) { $extra = @("--checksum") }
    & $rc copy $src "tamara:expLB/coords" --include "*.pt" `
        --transfers 4 --checkers 8 --log-file $rlog --log-level INFO @extra
    if ($LASTEXITCODE -ne 0) { Log "rclone copy exited $LASTEXITCODE -- see logs\coords_rclone.log" }
    elseif ($Verify) {
        & $rc check $src "tamara:expLB/coords" --include "*.pt" --checksum `
            --log-file $rlog --log-level INFO
        Log "rclone check exited $LASTEXITCODE (0 = every chunk verified on Drive)"
    }
}

# ---- stall detection ------------------------------------------------------------------
# Measured 2026-09-15: NDIF result downloads from prod-ndif-results.s3.amazonaws.com are
# erratic on this machine. The same 34.2 MB session came back in 3s once and took 11m45s
# the next time, at ~35 KB/s, while a 25 MB Cloudflare download on the SAME machine at the
# SAME moment still ran at 2.23 MB/s and the worker sat at 0% CPU. So it is not bandwidth,
# not the link and not compute (NDIF reports RUNNING=15s): it is one sick TCP flow.
#
# A hung flow is invisible to Test-ChainAlive -- python is alive and "working" -- so the
# supervisor would happily watch a 48-hour run. Killing python forces a fresh connection,
# and resume is chunk-granular, so the cost is bounded by whatever session was in flight.
#
# The counter matters: if every new connection is also slow, restarting is pure loss, so
# after MaxStallKills consecutive kills with no new chunk we stop killing and let it grind.
# 6, not 8: a healthy session on this corpus writes its chunk in 23-35s (measured
# 13:49:11 / 13:49:38 / 13:50:01), and the long-context datasets -- insider-trading at
# ctx p50 1267, ai_audit at 5285 -- should still land inside 4 min because the token
# budget gives them fewer rows per batch. 6 caps the waste of a bad flow while keeping
# roughly 10x headroom over a healthy session; MaxStallKills backs off if it is wrong.
$StallMinutes  = 6
$MaxStallKills = 5
$script:stallKills = 0
$script:lastChunkCount = -1

function Get-NewestChunkAge {
    $c = Get-ChildItem (Join-Path $Repo "$Out\passB\*\chunk_*.pt") -Recurse -ErrorAction SilentlyContinue
    if (-not $c) { return $null }
    $newest = ($c | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
    return @{ Age = ((Get-Date) - $newest.LastWriteTime).TotalMinutes; Count = $c.Count }
}

function Test-Stalled {
    $info = Get-NewestChunkAge
    if (-not $info) { return }
    if ($info.Count -ne $script:lastChunkCount) {
        # progress happened since the last look -- forgive the earlier kills
        $script:lastChunkCount = $info.Count
        $script:stallKills = 0
        return
    }
    if ($info.Age -lt $StallMinutes) { return }
    if ($script:stallKills -ge $MaxStallKills) {
        Log ("stalled {0:N1} min but {1} kills have not helped -- letting it grind" -f $info.Age, $script:stallKills)
        return
    }
    $py = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
          Where-Object { $_.CommandLine -match '8_coords_ndif\.py' }
    if (-not $py) { return }
    $script:stallKills++
    Log ("no new chunk for {0:N1} min at {1} chunks -- killing python to force a fresh connection (kill #{2})" -f `
         $info.Age, $info.Count, $script:stallKills)
    foreach ($p in $py) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    # The chain script outlives the python it was waiting on -- it still has to log the exit
    # code -- so Test-ChainAlive keeps returning true and the relaunch waits a whole tick.
    # Killing the chain too lets the relaunch check below fire in this same iteration.
    $ch = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
          Where-Object { $_.CommandLine -match '-File\s+\S*run_coords_chain\.ps1' -and
                         $_.CommandLine -notmatch 'Get-CimInstance' }
    foreach ($c in $ch) { Stop-Process -Id $c.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
}

$restarts = 0
while ($true) {
    if (Test-Done) {
        Log "DONE -- run-coords\passB\.done present. Final checksum sync to Drive."
        Sync-Drive -Verify
        Log "See logs\coords_passB.log and logs\coords_rclone.log"
        Remove-Item $lock -Force -ErrorAction SilentlyContinue
        exit 0
    }

    Test-Stalled

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

    Sync-Drive
    Start-Sleep -Seconds 300
}
