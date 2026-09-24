# W6G forensics: scheduled tasks (S-16/S-17/S-18), rclone token + sync (S-19/S-20),
# foreign jobs / resume loop (S-21), HF cache size (S-28). READ-ONLY.
$ErrorActionPreference = 'Continue'

Write-Output "#### A. boot time and current time"
$os = Get-CimInstance Win32_OperatingSystem
Write-Output ("  NOW " + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + "  |  LastBootUpTime " + $os.LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss'))

Write-Output ""
Write-Output "#### B. every scheduled task with a boot trigger  (name | state | enabled | bootDelay | lastRun | lastResult)"
$n = 0; $boot = 0
foreach ($t in (Get-ScheduledTask -ErrorAction SilentlyContinue)) {
  if ($t.TaskPath -notmatch '^\\Microsoft') {
    $n++
    $bt = @($t.Triggers | Where-Object { $_.CimClass.CimClassName -eq 'MSFT_TaskBootTrigger' })
    if ($bt.Count -gt 0) {
      $boot++
      $i = $t | Get-ScheduledTaskInfo
      Write-Output ("  " + $t.TaskName.PadRight(26) + " state=" + $t.State + " enabled=" + $t.Settings.Enabled +
                    " delay=" + ($bt | ForEach-Object { $_.Delay }) -join ',' )
      Write-Output ("      lastRun=" + $i.LastRunTime + "  lastResult=" + $i.LastTaskResult + "  next=" + $i.NextRunTime)
    }
  }
}
Write-Output ("  non-Microsoft tasks: " + $n + " ; with a boot trigger: " + $boot)

Write-Output ""
Write-Output "#### C. tasks WITHOUT a boot trigger (name | state | enabled | triggers)"
foreach ($t in (Get-ScheduledTask -ErrorAction SilentlyContinue)) {
  if ($t.TaskPath -notmatch '^\\Microsoft') {
    $bt = @($t.Triggers | Where-Object { $_.CimClass.CimClassName -eq 'MSFT_TaskBootTrigger' })
    if ($bt.Count -eq 0) {
      Write-Output ("  " + $t.TaskName.PadRight(26) + " state=" + $t.State + " enabled=" + $t.Settings.Enabled +
                    " triggers=" + (($t.Triggers | ForEach-Object { $_.CimClass.CimClassName }) -join ','))
    }
  }
}

Write-Output ""
Write-Output "#### D. rclone config + backup (S-20)"
foreach ($p in @("$env:USERPROFILE\AppData\Roaming\rclone\rclone.conf", 'C:\Users\user4\AppData\Roaming\rclone\rclone.conf')) {
  Get-ChildItem (Split-Path $p) -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Output ("  " + $_.FullName + "  " + $_.Length + "  " + $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
  }
}

Write-Output ""
Write-Output "#### E. Drive-sync logs around 2026-09-22 12:18-17:28 (S-19)"
foreach ($lg in @('kd_rclone.log','crossgen_rclone.log','coords_rclone.log','sync_all.log','night_rclone.log','scale_rclone.log')) {
  $p = "C:\Users\user4\expLB\logs\$lg"
  if (-not (Test-Path $p)) { Write-Output ("  " + $lg + " MISSING"); continue }
  $fi = Get-Item $p
  Write-Output ("  --- " + $lg + "  " + $fi.Length + "  mtime " + $fi.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
  Get-Content $p | Where-Object { $_ -match '2026-09-22' } | Select-Object -Last 14 | ForEach-Object { Write-Output ("      " + $_) }
}

Write-Output ""
Write-Output "#### F. 70B layer sweep log, resume/failure lines (S-21)"
foreach ($lg in @('crossgen_scale_70b.log','scale_coords.status','crossgen_chain.status')) {
  $p = "C:\Users\user4\expLB\logs\$lg"
  if (-not (Test-Path $p)) { Write-Output ("  " + $lg + " MISSING"); continue }
  $fi = Get-Item $p
  Write-Output ("  --- " + $lg + "  " + $fi.Length + "  mtime " + $fi.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
  Get-Content $p | Where-Object { $_ -match 'attempt|resume|skip|MemoryError|memory|fail|exit [1-9]|Traceback|killed' } |
    Select-Object -First 30 | ForEach-Object { Write-Output ("      " + $_) }
}

Write-Output ""
Write-Output "#### G. HuggingFace cache for Llama-3.3-70B (S-28)"
$hf = 'C:\Users\user4\.cache\huggingface\hub\models--meta-llama--Llama-3.3-70B-Instruct'
if (Test-Path $hf) {
  foreach ($sub in @('blobs','snapshots','refs')) {
    $d = Join-Path $hf $sub
    if (Test-Path $d) {
      $fs = Get-ChildItem $d -Recurse -File -Force -ErrorAction SilentlyContinue
      $sum = ($fs | Measure-Object Length -Sum).Sum
      Write-Output ("  " + $sub + ": " + $fs.Count + " files, " + [math]::Round($sum/1GB,2) + " GB")
      $fs | Sort-Object Length -Descending | Select-Object -First 5 |
        ForEach-Object { Write-Output ("     " + $_.Name.Substring(0,[Math]::Min(24,$_.Name.Length)) + "  " + [math]::Round($_.Length/1GB,3) + " GB  " + $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm')) }
    }
  }
} else { Write-Output "  cache dir NOT PRESENT (deleted)" }
Write-Output "  free space on C:"
Get-PSDrive C | ForEach-Object { Write-Output ("     used " + [math]::Round($_.Used/1GB,2) + " GB  free " + [math]::Round($_.Free/1GB,2) + " GB") }
