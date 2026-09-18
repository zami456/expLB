<#
  Wait for the crossgen chain to finish, then write one report answering the questions that
  actually get asked: did every pass complete, how many layers each produced, and does Drive
  hold the same files as the lab PC.

  Runs on the lab PC so it survives SSH drops and does not depend on the laptop, which has
  ~0.4 GB of RAM free and reclaims long-lived watchers. Read logs\crossgen_final_report.txt
  afterwards instead of holding a session open.

  Read-only: Get-ChildItem, rclone size, rclone dedupe in LIST mode. It deletes nothing.
#>
param([int]$MaxWaitHours = 24, [int]$PollSeconds = 300)
$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$out = Join-Path $repo "logs\crossgen_final_report.txt"
function Say($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $out }

Say "=== waiting for Crossgen-Chain ==="
$deadline = (Get-Date).AddHours($MaxWaitHours)
while ((Get-Date) -lt $deadline) {
    $t = Get-ScheduledTask -TaskName 'Crossgen-Chain' -ErrorAction SilentlyContinue
    if (-not $t -or $t.State -ne 'Running') { break }
    Start-Sleep -Seconds $PollSeconds
}
$t = Get-ScheduledTask -TaskName 'Crossgen-Chain' -ErrorAction SilentlyContinue
if ($t -and $t.State -eq 'Running') { Say "GAVE UP: chain still running after $MaxWaitHours h"; exit 2 }
Say "chain finished"

# per-pass completion and layer coverage
Say "--- passes ---"
foreach ($p in @('B', 'A', 'C', 'B_fine', 'P')) {
    $s = Join-Path $repo "logs\crossgen_full_$p.status"
    $last = if (Test-Path $s) { (Get-Content $s -Tail 1) } else { "(no status file)" }
    Say ("  pass $p : $last")
}
Say "--- layer coverage (root matrices per pass) ---"
foreach ($sub in @('', 'passA', 'passC', 'passB_fine', 'passP')) {
    $m = if ($sub) { Join-Path $repo "run-crossgen\$sub\matrices" } else { Join-Path $repo "run-crossgen\matrices" }
    $name = if ($sub) { $sub } else { 'passB (root)' }
    if (Test-Path $m) {
        $f = @(Get-ChildItem $m -Filter *.json -ErrorAction SilentlyContinue)
        $L = $f | ForEach-Object { if ($_.Name -match '_L(\d+)_') { [int]$Matches[1] } } | Sort-Object -Unique
        Say ("  $name : $($f.Count) files, $($L.Count) layers")
    } else { Say ("  $name : absent") }
}

# local vs Drive
$rc = Join-Path $repo "tools\rclone.exe"
if (-not (Test-Path $rc)) { $rc = "C:\Users\user4\tools\rclone.exe" }
$lf = Get-ChildItem (Join-Path $repo "run-crossgen") -Recurse -File -ErrorAction SilentlyContinue
Say ("--- local: " + $lf.Count + " files, " + [math]::Round((($lf | Measure-Object Length -Sum).Sum)/1MB, 1) + " MB")
if (Test-Path $rc) {
    Say "--- Drive ---"
    & $rc size "tamara:expLB/crossgen" 2>&1 | ForEach-Object { Say ("  " + $_) }
    Say "--- duplicates (list only) ---"
    $dup = & $rc dedupe --dedupe-mode list "tamara:expLB/crossgen" 2>&1 | Where-Object { $_ -match 'duplicate' }
    if ($dup) { $dup | Select-Object -First 10 | ForEach-Object { Say ("  " + $_) } } else { Say "  none" }
}
Say "=== report complete ==="
