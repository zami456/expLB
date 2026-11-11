# NARCBench stage runner: the reader jobs of one stage, in order, one process at a time.
# Exit codes of narc_read.py read: 0 done; 2 unfrozen input (stop); 3 session watchdog / 4 transient wait cap /
# 1 other (relaunch after 15 min, so zombie sessions finish first); 5 collision with another Llama reader (stop);
# 6 repeated unexplained failures (stop).
param([string]$Stage = "stage1")
$ErrorActionPreference = "Continue"
$py = "E:\aletheia\venv08\Scripts\python.exe"
$dir = "E:\narcbench\reader"
$log = "E:\narcbench\logs\$Stage.log"
$status = "E:\narcbench\logs\$Stage.status"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content -Path $log -Encoding utf8 }
$jobs = Get-Content "$dir\$Stage.jobs" | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith("#") }
Log "START $Stage ($($jobs.Count) jobs) pid $PID"
"RUNNING" | Set-Content $status
foreach ($job in $jobs) {
  $attempt = 0
  while ($true) {
    $attempt++
    Log "job [$job] attempt $attempt"
    $p = Start-Process -FilePath $py -ArgumentList ("-I narc_read.py " + $job) -WorkingDirectory $dir -NoNewWindow -Wait -PassThru `
      -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err"
    $code = $p.ExitCode
    Log "job [$job] exit $code"
    if ($code -eq 0) { break }
    if (($code -eq 2) -or ($code -eq 5) -or ($code -eq 6) -or ($attempt -ge 8)) {
      Log "STOP: job [$job] exit $code after $attempt attempt(s)"; "FAILED $job exit $code" | Set-Content $status; exit $code
    }
    Start-Sleep -Seconds 900
  }
}
Log "DONE $Stage"
"DONE" | Set-Content $status
