# qwen_d4_lb run, then report (narcbench, 2026-10-23). CPU only. Launch it detached with WMI from Claude Code's
# PowerShell tool (Win32_Process.Create), never the Bash tool. Status: E:\narcbench\logs\qwen_d4_lb_run.status.
$ErrorActionPreference = "Continue"
$dd = "D:\expLB\liars-bench\src\probes\deception-detection"
$py = "D:\expLB\liars-bench\.venv\Scripts\python.exe"
$log = "E:\narcbench\logs\qwen_d4_lb_run.log"
$status = "E:\narcbench\logs\qwen_d4_lb_run.status"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
"RUNNING" | Set-Content $status
$code = 0
foreach ($cmd in @("run", "report")) {
  $p = Start-Process -FilePath $py -ArgumentList "-m deception_detection.scripts.narcbench.qwen_d4_lb $cmd" `
    -WorkingDirectory $dd -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$log.$cmd.out" -RedirectStandardError "$log.$cmd.err"
  $code = $p.ExitCode
  "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  qwen_d4_lb $cmd exit $code" | Add-Content -Path $log -Encoding utf8
  if ($code -ne 0) { break }
}
$(if ($code -eq 0) { "DONE" } else { "FAILED exit $code" }) | Set-Content $status
