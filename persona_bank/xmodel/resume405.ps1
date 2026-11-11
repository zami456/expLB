<#
  Morning resume of the Llama-3.1-405B Alpaca read (owner 2026-10-12 22:0x: "pause it, resume in the morning").

  The read was paused because NDIF's result downloads (Amazon S3, us-east) came at 4-19 KB/s -- the ISP's route to AWS,
  not this laptop's link (Cloudflare 4 MB/s at the same time). Each run of this script measures the speed of a 2 MB
  range from S3 us-east. At >= MinKBps it re-enables and starts task XModel-llama405-alpaca (the chain skips the finished
  generation and resumes the read at "have 0 / 2000"), then disables itself. Below it, it logs and waits for the next
  trigger. Log: <run>\logs\resume.status.

  Task XModel-llama405-resume: from 07:00 every 30 min for 6 h; WakeToRun.
      powershell -ExecutionPolicy Bypass -File xmodel\resume405.ps1 -Register
      powershell -ExecutionPolicy Bypass -File xmodel\resume405.ps1 -DryRun     # measure and say what it would do
#>
param([int]$MinKBps = 300, [switch]$Register, [switch]$DryRun, [string]$At = "07:00")
$ErrorActionPreference = "Continue"
$name = "XModel-llama405-resume"
$Logs = Join-Path $env:USERPROFILE "xmodel-runs\Llama-3.1-405B-Instruct\logs"
$status = Join-Path $Logs "resume.status"
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

if ($Register) {
    $start = [datetime]::Today.AddDays(1).Add([timespan]::Parse($At))
    if ((Get-Date).TimeOfDay -lt [timespan]::Parse($At)) { $start = [datetime]::Today.Add([timespan]::Parse($At)) }
    $arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $PSCommandPath + '"'
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $PSScriptRoot
    $trigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Minutes 30) `
                   -RepetitionDuration (New-TimeSpan -Hours 6)
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun `
        -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings -Trigger $trigger `
        -Description "resume the paused Llama-405B Alpaca read once S3 downloads are healthy" | Out-Null
    $i = Get-ScheduledTaskInfo -TaskName $name
    Write-Output ("{0}: state={1} nextRun={2}" -f $name, (Get-ScheduledTask -TaskName $name).State, $i.NextRunTime)
    return
}

$url = "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip"
$bps = 0
$out = & curl.exe -s -o NUL -r 0-2000000 -w "%{speed_download}" --max-time 40 $url 2>$null
if ($out) { $bps = [double]($out -replace ',', '.') }
$kbps = [int]($bps / 1024)
$alp = Get-ScheduledTask -TaskName XModel-llama405-alpaca -ErrorAction SilentlyContinue
if ($DryRun) { "S3 us-east: $kbps KB/s (need $MinKBps); alpaca task: $($alp.State); would resume: $($kbps -ge $MinKBps)"; return }
if (-not $alp) { Note "STOP task XModel-llama405-alpaca not found"; exit 1 }
if ($alp.State -ne "Disabled") { Note "alpaca task is $($alp.State), not Disabled -- nothing to resume; disabling $name"; Disable-ScheduledTask -TaskName $name | Out-Null; exit 0 }
if ($kbps -lt $MinKBps) { Note "S3 us-east $kbps KB/s < ${MinKBps}: still slow, not resumed"; exit 0 }
Enable-ScheduledTask -TaskName XModel-llama405-alpaca | Out-Null
Start-ScheduledTask -TaskName XModel-llama405-alpaca
Start-Sleep -Seconds 15
$i = Get-ScheduledTaskInfo -TaskName XModel-llama405-alpaca
Note "S3 us-east $kbps KB/s >= ${MinKBps}: RESUMED XModel-llama405-alpaca (state $((Get-ScheduledTask -TaskName XModel-llama405-alpaca).State), lastRun $($i.LastRunTime))"
Disable-ScheduledTask -TaskName $name | Out-Null
