<#
  Qwen2.5-72B's Liars' Bench read for the detection comparison (owner 2026-10-16: "Run Qwen detection").
  eval_queue.ps1 restricted to Qwen and the three datasets that have both honest and lying Qwen rows (ID 5,160,
  HP-KR 2,571, HP-C 2,374; results/subsets): no gate, no Drive copy. Mean over the reply, every 4th layer + 40, into
  %USERPROFILE%\xmodel-runs\Qwen2.5-72B-Instruct\eval_lb\. Resumable; status in %USERPROFILE%\xmodel-runs\eval_queue.status.
  Keeps the laptop from idle sleep while it runs; closing the lid still sleeps it.
  Smoke 2026-10-16 01:24-01:30: 16 HP-KR rows, spans 16/16 exact, (16, 20, 8192) fp16, finite (_scratch\qwen_evallb_smoke).
#>
$repo = "C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$null = [Win32.Power]::SetThreadExecutionState([uint32]"0x80000001")   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
& (Join-Path $repo "xmodel\eval_queue.ps1") -WaitFor "" -Only @("qwen") -NoDrive `
    -Datasets @("instructed-deception", "harm-pressure-knowledge-report", "harm-pressure-choice")
