<#
  Stall guard for the Llama-3.1-405B run (owner 2026-10-11: "continue llama 405B"). The persona chain, the K chain and
  the Alpaca chain each retry a python process that EXITS; none of them stops one that HANGS. This does.

  The random-bank queue's guard (random_persona_bank/scripts/queue.ps1, Guard; ran 2026-10-10/11 on five models),
  extended to the K and Alpaca stages, with the 405B limits of xmodel/LOG.md (2026-10-09 02:44): QUEUED never counts --
  405B serves one request at a time, so a job waits in line. Every minute, for each python process working on 405B:
    - its .err.log ENDS in the engineio b'P' error (socketio's receive thread is dead; a shard that recovered would
      have logged further lines), or
    - its current NDIF job has been RUNNING / DISPATCHED for >= RunMinutes (judge: JudgeMinutes) since this guard
      first saw it, or
    - its logs have been silent for >= SilentMinutes and its last NDIF state is not QUEUED, or
    - (added 2026-10-11 09:2x, after a lost job) its job is QUEUED and a LATER job of the same stage has left the
      queue while it waited >= OvertakenMinutes, or it has been QUEUED >= QueuedMinutes in any case
  -> the process is stopped. The chain that owns it records the exit and its next attempt resumes.
  K's python waits inside k_run.py for the persona chain's DONE (--gate_status), silently: K is not guarded until then.

  Exits when all three chains end in DONE. Log: <run>\logs\guard.status.
      powershell -ExecutionPolicy Bypass -File xmodel\guard405.ps1 -Register    # task XModel-llama405-guard (starts it)
      powershell -ExecutionPolicy Bypass -File xmodel\guard405.ps1 -Probe       # what it sees now; stops nothing
#>
param([int]$RunMinutes = 45, [int]$JudgeMinutes = 20, [int]$SilentMinutes = 60,
      [int]$OvertakenMinutes = 20, [int]$QueuedMinutes = 120, [switch]$Register,
      # print what the guard sees for each live 405B python process, stop nothing, exit
      [switch]$Probe)
$ErrorActionPreference = "Continue"
$Folder  = "Llama-3.1-405B-Instruct"
$RunBase = Join-Path $env:USERPROFILE "xmodel-runs"
$Root    = Join-Path $RunBase $Folder
$Code    = Join-Path $Root "code"
$Logs    = Join-Path $Root "logs"
$KLogs   = Join-Path $Root "K-logs"
$ALogs   = Join-Path $RunBase "alpaca-logs-llama405"
$Persona = Join-Path $Logs "chain.status"
$KStatus = Join-Path $Root "K\k_chain.status"
$AStatus = Join-Path $Logs "alpaca_gen.status"

if ($Register) {
    $name = "XModel-llama405-guard"
    $arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $PSCommandPath + '"'
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $PSScriptRoot
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Days 3) -StartWhenAvailable
    # a 30-minute repeat re-enters the guard if it is killed; a second copy exits at once (lock below)
    $t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
            -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 3)
    $t2 = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $t2.Delay = "PT4M"
    Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    # S4U needs admin on this laptop (2026-10-07): Interactive, as every other xmodel task
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
        -Trigger @($t1, $t2) -Description "stall guard for the Llama-3.1-405B persona, K and Alpaca chains" | Out-Null
    Start-ScheduledTask -TaskName $name
    Start-Sleep -Seconds 15
    $i = Get-ScheduledTaskInfo -TaskName $name
    Write-Output ("{0}: state={1} lastRun={2} lastResult={3}" -f $name, (Get-ScheduledTask -TaskName $name).State, $i.LastRunTime, $i.LastTaskResult)
    return
}

New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$status = Join-Path $Logs "guard.status"
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
# one guard at a time (the probe takes no lock: it only reads)
if (-not $Probe) {
    # outside the run folder (2026-10-11 15:02): a lock held open inside <run>\logs failed every Drive copy of the run
    try { $lock = [System.IO.File]::Open((Join-Path $RunBase "_guard405.lock"), 'OpenOrCreate', 'ReadWrite', 'None') } catch { exit 0 }
}

function IsDone([string]$f, [string]$pattern) { (Test-Path $f) -and ((Get-Content $f -Tail 1) -match $pattern) }
function AllDone { (IsDone $Persona '^DONE ') -and (IsDone $KStatus '  DONE') -and (IsDone $AStatus '  DONE$') }

function LastStatus([string]$text) {
    # last "[<job id>] STATUS" in a chunk of nnsight status output
    $m = [regex]::Matches($text, '\[([0-9a-f\-]{36})\]\s+([A-Z_]+)\s')
    if ($m.Count -eq 0) { return $null }
    $x = $m[$m.Count - 1]
    return @{ Job = $x.Groups[1].Value; State = $x.Groups[2].Value }
}

function Tail([string]$path, [int]$bytes = 4096) {
    try {
        $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
        try {
            $n = [Math]::Min($bytes, $fs.Length)
            $null = $fs.Seek(-$n, 'End')
            $buf = New-Object byte[] $n
            $null = $fs.Read($buf, 0, $n)
            return [System.Text.Encoding]::UTF8.GetString($buf)
        } finally { $fs.Close() }
    } catch { return "" }
}

function Newest([string]$dir, [string]$pattern) {
    Get-ChildItem $dir -Filter $pattern -EA SilentlyContinue | Where-Object { $_.Name -notlike "*.err.log" } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

# python process -> @{ Key; Log; Err; Limit }, or $null when it is not a guarded NDIF stage
function Classify([string]$c) {
    # persona chain shards (as queue.ps1). Only its three NDIF scripts: the Alpaca generator and reader also run with
    # this code folder on their command lines (caught by the test, 2026-10-11)
    if ($c.Contains($Code) -and $c -match '(1_generate|3_judge|2_activations)_ndif\.py') {
        $stage = $null
        if ($c -match '1_generate_ndif\.py') { $stage = "generate" }
        elseif ($c -match '3_judge_ndif\.py') { $stage = "judge_eval" }   # xmodel/run_chain.ps1's log name (the random-bank chain says "judge")
        elseif ($c -match '2_activations_ndif\.py' -and $c -match 'persona\\(vectors_\w+)') { $stage = $Matches[1] }
        if (-not $stage -or $c -notmatch '--shard\s+(\d)/3') { return $null }
        $key = "$stage.$($Matches[1])"
        return @{ Key = $key; Log = Join-Path $Logs "$key.log"; Err = Join-Path $Logs "$key.err.log"
                  Limit = $(if ($stage -eq "judge_eval") { $JudgeMinutes } else { $RunMinutes }) }
    }
    if ($c -match 'llama-405b-3\.1' -and $c -match 'k_run\.py"?\s+(\w+)') {
        $step = $Matches[1]                                         # the LAST -match sets $Matches
        if ($step -eq "directions" -or -not (IsDone $Persona '^DONE ')) { return $null }   # CPU / waiting at the gate
        if ($c -notmatch '--set\s+(\w+)') { return $null }
        $f = Newest $KLogs "k_$($Matches[1])_$step.*.log"
        if (-not $f) { return $null }
        return @{ Key = "K $($Matches[1]) $step"; Log = $f.FullName; Err = $f.FullName -replace '\.log$', '.err.log'; Limit = $RunMinutes }
    }
    if ($c -match 'alpaca_gen\.py' -and $c.Contains($Folder) -and $c -match '--shard\s+(\d)/3') {
        $i = $Matches[1]
        return @{ Key = "alpaca gen.$i"; Log = Join-Path $ALogs "gen.$i.log"; Err = Join-Path $ALogs "gen.$i.err.log"; Limit = $RunMinutes }
    }
    if ($c -match '8_coords_ndif\.py' -and $c.Contains($Folder) -and $c -notmatch '--prepare_only') {
        $f = Newest $ALogs "alpaca.*.log"
        if (-not $f) { return $null }
        return @{ Key = "alpaca read"; Log = $f.FullName; Err = $f.FullName -replace '\.log$', '.err.log'; Limit = $RunMinutes }
    }
    return $null
}

$firstRunning = @{}     # job id -> when this guard first saw it RUNNING / DISPATCHED
$firstSeen = @{}        # job id -> when this guard first saw it at all
$started = @{}          # job id -> when this guard first saw it out of the queue (DISPATCHED / RUNNING / COMPLETED)
$jobStage = @{}         # job id -> its stage (generate, judge, vectors_s23, alpaca gen, ...)
function Guard {
    $groups = @{}
    foreach ($p in @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue)) {
        if (-not $p.CommandLine) { continue }
        $g = Classify $p.CommandLine
        if (-not $g) { continue }
        if (-not $groups.ContainsKey($g.Key)) { $groups[$g.Key] = @{ G = $g; Pids = @() } }
        $groups[$g.Key].Pids += $p.ProcessId                        # a venv launcher and its interpreter share one line
    }
    # pass 1: every live process's current job and state; when each job was first seen, and first seen past the queue
    $now = Get-Date
    $obs = @{}
    foreach ($key in $groups.Keys) {
        $g = $groups[$key].G
        if (-not (Test-Path $g.Log)) { continue }
        $times = @((Get-Item $g.Log).LastWriteTime)
        if (Test-Path $g.Err) { $times += (Get-Item $g.Err).LastWriteTime }
        $silent = ($now - ($times | Sort-Object -Descending | Select-Object -First 1)).TotalMinutes
        # K waited silently at its gate until the persona chain wrote DONE: count its silence from that moment
        # (2026-10-11 17:31 the guard stopped K 28 s after the gate opened, "silent for 543 min")
        if ($key -like "K *") { $silent = [Math]::Min($silent, ($now - (Get-Item $Persona).LastWriteTime).TotalMinutes) }
        $st = LastStatus (Tail $g.Log)
        if (-not $st) { $st = LastStatus (Tail $g.Err) }
        if ($st) {
            if (-not $firstSeen.ContainsKey($st.Job)) { $firstSeen[$st.Job] = $now; $jobStage[$st.Job] = $key -replace '\.\d+$', '' }
            if ($st.State -in @("DISPATCHED", "RUNNING", "COMPLETED") -and -not $started.ContainsKey($st.Job)) { $started[$st.Job] = $now }
        }
        $obs[$key] = @{ St = $st; Silent = $silent }
    }
    # pass 2: the rules
    foreach ($key in $obs.Keys) {
        $g = $groups[$key].G
        $st = $obs[$key].St
        $silent = $obs[$key].Silent
        $why = $null
        # 2026-10-11 09:10: a 405B job sat QUEUED 32 min at "position 2" while two LATER jobs of the other shards ran --
        # lost in NDIF's queue; "QUEUED never counts" would have left that shard waiting forever. A queued job is now
        # stopped once a job of the same stage that was first seen >= 2 min after it has left the queue (and it has
        # waited >= OvertakenMinutes), or after QueuedMinutes in any case.
        if ($st -and $st.State -eq "QUEUED") {
            $qt = $firstSeen[$st.Job]
            $queued = ($now - $qt).TotalMinutes
            $stage = $jobStage[$st.Job]
            $later = @($started.Keys | Where-Object { $_ -ne $st.Job -and $jobStage[$_] -eq $stage -and $firstSeen[$_] -gt $qt.AddMinutes(2) })
            if ($later.Count -and $queued -ge $OvertakenMinutes) {
                $why = "job $($st.Job) QUEUED $([int]$queued) min, overtaken by later job(s) $(($later | ForEach-Object { $_.Substring(0, 8) }) -join ',')"
            } elseif ($queued -ge $QueuedMinutes) {
                $why = "job $($st.Job) QUEUED for $([int]$queued) min"
            }
        }
        if (-not $why -and $st -and ($st.State -eq "RUNNING" -or $st.State -eq "DISPATCHED")) {
            if (-not $firstRunning.ContainsKey($st.Job)) { $firstRunning[$st.Job] = $now }
            $run = ($now - $firstRunning[$st.Job]).TotalMinutes
            if ($run -ge $g.Limit) { $why = "job $($st.Job) $($st.State) for $([int]$run) min" }
        }
        if (-not $why -and $silent -ge $SilentMinutes -and -not ($st -and $st.State -eq "QUEUED")) {
            $why = "logs silent for $([int]$silent) min"
        }
        if (-not $why) {
            $errLines = @((Tail $g.Err 2048) -split "[\r\n]+" | Where-Object { $_.Trim() })
            if ($errLines.Count -and $errLines[-1] -match "invalid literal for int\(\) with base 10: b'P'") {
                $why = "socketio receive thread died (engineio b'P')"
            }
        }
        if ($Probe) {
            "{0}: pids {1} | last state {2} | silent {3:N0} min | limit {4} | would stop: {5}" -f $key,
                ($groups[$key].Pids -join ','), $(if ($st) { $st.State } else { "none" }), $silent, $g.Limit, $(if ($why) { $why } else { "no" })
            continue
        }
        if ($why) {
            Note "STALL $key ($why): stopping pids $($groups[$key].Pids -join ',')"
            foreach ($id in $groups[$key].Pids) { Stop-Process -Id $id -Force -EA SilentlyContinue }
        }
    }
}

if ($Probe) { Guard; "probe done (persona DONE: $(IsDone $Persona '^DONE '))"; exit 0 }
if (AllDone) { $lock.Close(); exit 0 }
Note "GUARD START run=$RunMinutes judge=$JudgeMinutes silent=$SilentMinutes overtaken=$OvertakenMinutes queued=$QueuedMinutes (min)"
while (-not (AllDone)) {
    try { Guard } catch { Note "guard error: $($_.Exception.Message)" }
    Start-Sleep -Seconds 60
}
Note "GUARD DONE (persona, K and Alpaca chains all DONE)"
$lock.Close()
