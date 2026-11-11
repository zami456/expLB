<#
  The user_stay matched pair (owner 2026-10-14, F-130): the kd persona step re-run for Qwen2.5-72B, then
  Llama-3.3-70B, with the role line + one "stay in character" sentence in the USER turn. Everything else is
  run_chain.ps1 unchanged: 36 roles x 5 variants x 30 questions, judge Llama-3.1-70B, vector banks s23 / all / s3
  over every layer, axes.

      %USERPROFILE%\xmodel-runs-userstay\<Model>\   the two run folders (deployed by register_pair.ps1)
      %USERPROFILE%\xmodel-runs-userstay\pair.status   one line per event

  One model at a time (laptop memory); each model's chain is resumable and exits at once when its own
  chain.status ends in DONE, so the task's 30-minute watchdog can re-enter this script safely.
  While it runs, the laptop is kept from idle sleep (SetThreadExecutionState); closing the lid still sleeps it.
  No Drive copies (-NoSync): the tamara: token expired 2026-10-14; copy by hand after `rclone config reconnect tamara:`.
#>
param([string]$SessionTimeout = "5400")
$ErrorActionPreference = "Continue"
$repo = "C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"
$Base = Join-Path $env:USERPROFILE "xmodel-runs-userstay"
$Py = Join-Path $repo ".venv-aa-win\Scripts\python.exe"
$status = Join-Path $Base "pair.status"
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$null = [Win32.Power]::SetThreadExecutionState([uint32]"0x80000001")   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

if ((Test-Path $status) -and ((Get-Content $status -Tail 1) -match 'PAIR DONE')) { exit 0 }
Note "pair start pid $PID sessionTimeout=$SessionTimeout"
foreach ($t in @(@{ Tag = "qwen"; Folder = "Qwen2.5-72B-Instruct" }, @{ Tag = "llama33"; Folder = "Llama-3.3-70B-Instruct" })) {
    $code = Join-Path $Base "$($t.Folder)\code"
    $cs = Join-Path $Base "$($t.Folder)\logs\chain.status"
    Note "$($t.Tag) chain start"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $code "run_chain.ps1") -Tag $t.Tag -RunBase $Base `
        -Py $Py -SkipEval -NoSync -PromptFormat user_stay -SessionTimeout $SessionTimeout -MinDownloadKbps 40 -DownloadGrace 20
    $last = if (Test-Path $cs) { Get-Content $cs -Tail 1 } else { "(no chain.status)" }
    Note "$($t.Tag) chain exit $LASTEXITCODE, last: $last"
    if ($last -notmatch '^DONE ') { Note "STOP: $($t.Tag) not DONE; the watchdog re-runs this script (resumable)"; exit 1 }
}
Note "PAIR DONE"
