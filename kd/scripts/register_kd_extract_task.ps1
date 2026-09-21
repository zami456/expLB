# Extract pooled activations for the two factor-probe corpora (kd Phase 2).
#
# Layers are pinned to Pass B's exact 21, read off a stored chunk rather than assumed:
# d_knowledge and d_utterance are only comparable with the corpus activations if they live
# at the same layers, and "--layers every4" would skip 22, which is Apollo's layer.
#
# Scheduled Task, never Start-Process over SSH. 8_coords_ndif.py is resumable: it writes
# chunk_<seq>.pt per batch and skips what exists, so re-entry after a reboot is free.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "KD-Extract"
$py = Join-Path $repo ".venv-aa\Scripts\python.exe"
$script = Join-Path $repo "src\assistant_axis_ndif\8_coords_ndif.py"
$log = Join-Path $repo "logs\kd_extract.log"
$out = Join-Path $repo "kd\run\factor_probes"
# --layers is nargs="+", so SPACE separated, not comma. --pooling takes both so the stored
# tensor matches Pass B's (n, 21, 8192, 2): mean and last, in that order.
$LAYERS = "0 4 8 12 16 20 22 24 28 32 36 40 44 48 52 56 60 64 68 72 76"

$inner = "`$env:PYTHONIOENCODING='utf8'; & '$py' '$script' " +
         "--datasets kd_knowledge kd_utterance " +
         "--layers $LAYERS --pooling mean last --output_dir '$out' " +
         "2>&1 | Out-File -Append -Encoding utf8 '$log'"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -Command "' + $inner.Replace('"', '\"') + '"') `
    -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 8) -StartWhenAvailable
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT12M"

Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger -Description "kd factor probes: pooled activations at Pass B's 21 layers" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 25
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
Get-Content $log -Tail 6 -ErrorAction SilentlyContinue
