# Runs the reader jobs listed in <Chain>.jobs one after another (one NDIF process at a time).
# A job that exits 3 (watchdog), 4 (transient wait exhausted) or 6 (unexplained failures) is relaunched after
# two minutes, up to six attempts; it resumes by row id. Status lines go to data/emotion_coords/logs/chain_<Chain>.status.
# Launch detached from the PowerShell tool (WMI Win32_Process.Create), never from the Bash tool.
param([string]$Chain = "pilot", [string]$Python = "E:\aletheia\venv08\Scripts\python.exe")

$reader = Split-Path -Parent $MyInvocation.MyCommand.Path
$logs = Join-Path (Split-Path (Split-Path $reader)) "data\emotion_coords\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$status = Join-Path $logs "chain_$Chain.status"
$out = Join-Path $logs "chain_$Chain.out"
$jobs = Get-Content (Join-Path $reader "$Chain.jobs") | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith("#") }
"START $(Get-Date -Format s) pid $PID" | Add-Content $status
foreach ($job in $jobs) {
    $code = 1
    for ($attempt = 1; $attempt -le 6; $attempt++) {
        "RUN $job (attempt $attempt) $(Get-Date -Format s)" | Add-Content $status
        $jobArgs = $job.Trim() -split '\s+'
        Push-Location $reader
        & $Python -I ec_read.py @jobArgs *>> $out
        $code = $LASTEXITCODE
        Pop-Location
        "EXIT $code $job $(Get-Date -Format s)" | Add-Content $status
        if ($code -eq 0) { break }
        if ($code -in 3, 4, 6) { Start-Sleep -Seconds 120; continue }
        break
    }
    if ($code -ne 0) { "FAILED $job $(Get-Date -Format s)" | Add-Content $status; exit $code }
}
"DONE $(Get-Date -Format s)" | Add-Content $status
