# qwen_d4_attn null_only, then report (narcbench, 2026-10-24): the null again after the 12:10 power loss. CPU only.
# Launch detached with WMI from Claude Code's PowerShell tool, never the Bash tool.
# Status: E:\narcbench\logs\qwen_d4_attn_run.status (WAITING, RUNNING, DONE, FAILED exit n, NOT RUN: ...).
$ErrorActionPreference = "Continue"
$dd = "D:\expLB\liars-bench\src\probes\deception-detection"
$py = "D:\expLB\liars-bench\.venv\Scripts\python.exe"
$log = "E:\narcbench\logs\qwen_d4_attn_null.log"
$status = "E:\narcbench\logs\qwen_d4_attn_null.status"
$reads = "E:\narcbench\verdict\logs\qwen_attn.status"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:MPLBACKEND = "Agg"
"WAITING for the span reads" | Set-Content $status
while ($true) {
  $s = Get-Content $reads -ErrorAction SilentlyContinue
  if ($s -eq "DONE") { break }
  if ($s -match "FAILED|STOPPED") { "NOT RUN: reads $s" | Set-Content $status; exit 1 }
  Start-Sleep -Seconds 60
}
"RUNNING" | Set-Content $status
$code = 0
foreach ($cmd in @("null_only", "report")) {
  $p = Start-Process -FilePath $py -ArgumentList "-m deception_detection.scripts.narcbench.qwen_d4_attn $cmd" `
    -WorkingDirectory $dd -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$log.$cmd.out" -RedirectStandardError "$log.$cmd.err"
  $code = $p.ExitCode
  "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  qwen_d4_attn $cmd exit $code" | Add-Content -Path $log -Encoding utf8
  if ($code -ne 0) { break }
}
$(if ($code -eq 0) { "DONE" } else { "FAILED exit $code" }) | Set-Content $status

