<#
  RECOVERY + BACKUP, to run the moment the lab PC is back on.

  Written 2026-09-24 after the machine powered off mid-run. Does three things, in
  this order, and reports each:

    1. REPORT where the KD-at-scale chain got to before the power-off.
    2. BACK UP to Drive, first -- so the new experiment output stops existing in
       one place. Copy only, into a NEW dated folder, never overwriting.
    3. RESTART the chain if it is not running. It was registered /SC ONCE, so it
       does NOT restart itself on boot. Every stage is resumable: generation skips
       finished rollout files, extraction skips vectors already built.

  Backup FIRST, restart SECOND -- deliberately. If the machine is unstable, the
  data that already exists matters more than the data still to come.

  Run it from the laptop:
    python audit/scripts/labpc_run.py <this file>
  or paste it into a PowerShell session on the lab PC.
#>
$ErrorActionPreference = "Continue"
$new  = "C:\Users\user4\expLB-resplit"
$rc   = "C:\Users\user4\expLB\tools\rclone.exe"
$stamp = Get-Date -Format "yyyy-MM-dd"
$dest = "tamara:expLB/_labpc_only_backup_$stamp"

function Line($m) { Write-Output ("  " + $m) }

Write-Output "=== 1. WHERE THE CHAIN GOT TO ==="
$status = Join-Path $new "logs\kd_scale.status"
if (Test-Path $status) { Get-Content $status | Select-Object -Last 14 }
else { Line "no status file -- chain never started?" }

Write-Output ""
Write-Output "  artifacts on disk:"
foreach ($t in @("3b", "8b")) {
    $p = Join-Path $new "run-scale-kd\$t"
    if (-not (Test-Path $p)) { Line "$t : not started"; continue }
    $r = 0
    Get-ChildItem "$p\responses\*.jsonl" -EA SilentlyContinue |
        ForEach-Object { $r += (Get-Content $_.FullName | Where-Object { $_.Trim() }).Count }
    Line ("$t : rollouts $r/5400 | scores " +
          (Get-ChildItem "$p\scores\*.json" -EA SilentlyContinue).Count + "/35" +
          " | vectors_all " + (Get-ChildItem "$p\vectors_all\*.pt" -EA SilentlyContinue).Count +
          " | vectors_s23 " + (Get-ChildItem "$p\vectors_s23\*.pt" -EA SilentlyContinue).Count +
          " | axes " + (Get-ChildItem "$p\axes_*.pt" -EA SilentlyContinue).Count)
}

# DF-48 check: the two axis variants must NOT be byte-identical.
Write-Output ""
Write-Output "  DF-48 check (the two axis variants must differ):"
foreach ($t in @("3b", "8b")) {
    $a = Get-Item (Join-Path $new "run-scale-kd\$t\axes_kd_${t}_vectors_all.pt") -EA SilentlyContinue
    $s = Get-Item (Join-Path $new "run-scale-kd\$t\axes_kd_${t}_vectors_s23.pt") -EA SilentlyContinue
    if ($a -and $s) {
        $ha = (Get-FileHash $a.FullName -Algorithm MD5).Hash
        $hs = (Get-FileHash $s.FullName -Algorithm MD5).Hash
        if ($ha -eq $hs) { Line "$t : *** ALERT - IDENTICAL, DF-48 REGRESSION ***" }
        else { Line "$t : OK, variants differ" }
    } else { Line "$t : axes not built yet" }
}

Write-Output ""
Write-Output "=== 2. BACK UP TO DRIVE (copy only, new folder, no overwrite) ==="
if (-not (Test-Path $rc)) { Line "rclone NOT FOUND at $rc -- skipping backup"; }
else {
    # The new experiment output first: it exists nowhere else.
    $jobs = @(
        @{ src = "$new\run-scale-kd";  dst = "$dest/run-scale-kd";  why = "NEW experiment output" },
        @{ src = "$new\logs";          dst = "$dest/resplit-logs";  why = "its logs" },
        @{ src = "$new\run-crossgen";  dst = "$dest/resplit-crossgen"; why = "leak-guard + km results" },
        @{ src = "C:\Users\user4\expLB\run-cap";      dst = "$dest/run-cap";      why = "RF-10, never backed up" },
        @{ src = "C:\Users\user4\expLB\run-cap-ctrl"; dst = "$dest/run-cap-ctrl"; why = "RF-10, never backed up" },
        @{ src = "C:\Users\user4\expLB\logs";         dst = "$dest/expLB-logs";   why = "RF-10, never backed up" }
    )
    foreach ($j in $jobs) {
        if (-not (Test-Path $j.src)) { Line ("skip " + $j.src + " (absent)"); continue }
        Line ("copying " + $j.src + "  ->  " + $j.dst + "   [" + $j.why + "]")
        & $rc copy $j.src $j.dst --create-empty-src-dirs --stats-one-line --stats 30s 2>&1 |
            Select-Object -Last 2
        $chk = & $rc check $j.src $j.dst 2>&1 | Select-String "0 differences|differences found" |
            Select-Object -Last 1
        Line ("  verify: " + $chk)
    }
}

Write-Output ""
Write-Output "=== 3. RESTART THE CHAIN IF IT IS NOT RUNNING ==="
$st = (schtasks /Query /TN "Audit-KD-Scale" /FO LIST | Select-String "Status")
Line ("task: " + $st)
$fin = 0
if (Test-Path $status) { $fin = (Get-Content $status | Select-String "chain finished").Count }
if ($fin -ge 1) {
    Line "chain already reported DONE -- not restarting."
} elseif ("$st" -match "Running") {
    Line "still Running -- leaving it alone."
} else {
    Line "not running and not finished -> restarting (stages are resumable)"
    schtasks /Run /TN "Audit-KD-Scale" | Out-Null
    Start-Sleep -Seconds 20
    Line ("after restart: " + (schtasks /Query /TN "Audit-KD-Scale" /FO LIST | Select-String "Status"))
}
Write-Output "=== done ==="
