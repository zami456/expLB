# Waits until chain <After> has finished (a DONE or FAILED line after its last START), then runs chain <Chain>.
# Keeps one NDIF reader at a time. Launch detached from the PowerShell tool (WMI Win32_Process.Create).
param([Parameter(Mandatory = $true)][string]$After, [Parameter(Mandatory = $true)][string]$Chain)

$reader = Split-Path -Parent $MyInvocation.MyCommand.Path
$logs = Join-Path (Split-Path (Split-Path $reader)) "data\emotion_coords\logs"
$status = Join-Path $logs "chain_$After.status"
while ($true) {
    if (Test-Path $status) {
        $lines = Get-Content $status
        $start = ($lines | Select-String -Pattern '^START' | Select-Object -Last 1).LineNumber
        $tail = if ($start) { $lines[($start - 1)..($lines.Count - 1)] } else { $lines }
        if ($tail | Where-Object { $_ -match '^(DONE|FAILED)' }) { break }
    }
    Start-Sleep -Seconds 30
}
& (Join-Path $reader "run_chain.ps1") -Chain $Chain
