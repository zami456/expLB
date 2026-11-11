<#
  Runs the random-bank chain for each model IN ORDER, one model at a time (at most three NDIF requests at once).
  Started by the RPB-queue Scheduled Task (register_queue.ps1) every 30 minutes; a second start while one runs is
  ignored (MultipleInstances IgnoreNew, plus the lock below). Each chain is resumable and exits at once when DONE,
  so re-entry is always safe.

  Order: llama33, qwen, mistral2501, mistral31, gemma3. A model whose chain ends without DONE stops the queue
  until the next trigger (the chain resumes where it stopped); the next model never starts out of turn.

  Stall guard (the earlier runs' by-hand rule, now automatic): NDIF jobs sometimes sit RUNNING / DISPATCHED for
  30-46 minutes and never finish, and an engineio socket fault can leave a shard silent. Every minute, for each
  live shard of the running chain: if its .err.log ends in the engineio b'P' error (receive thread dead), or its
  current NDIF job has been RUNNING / DISPATCHED for >= $StallMinutes since
  this guard first saw it, or its log has been silent for >= $StallMinutes, the shard's python processes are stopped.
  The chain records the exit and its next attempt regenerates only what is missing. QUEUED never counts.

  Llama-3.3 extra steps (owner 2026-10-10: "full depth K and alpaca for llama"), run after its chain is DONE and
  before Qwen starts, one at a time, so they never share the laptop's link with a persona chain:
      K       xmodel\k\k_chain.ps1 -Model llama-70b-3.3 (both item sets, all 80 layers) -> <Llama run>\K
      alpaca  xmodel\alpaca_chain.ps1 (Llama's 2,000 Liars' Bench Alpaca rows, all 80 layers, fp16) -> <Llama run>\eval\alpaca
  The other four models already have both (%USERPROFILE%\xmodel-runs\<Model>\K, eval*\alpaca).

  Attach (2026-10-10): a chain or step already running from an earlier queue instance is waited on, never started a
  second time (two chains on one run folder would append the same rows twice).

  Log: <RunBase>\_queue\queue.status
#>
param(
    [string[]]$Tags = @("llama33", "qwen", "mistral2501", "mistral31", "gemma3"),
    [string]$RunBase = "",
    [string]$Py = "",
    [string]$DriveBase = "tamara:expLB/random_persona_bank_2026-10-10",
    [string]$Rclone = "",
    [int]$StallMinutes = 25
)
$ErrorActionPreference = "Continue"
# powershell -File passes "-Tags a,b,c" as ONE string, not an array
$Tags = @($Tags | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$folders = @{ "llama33" = "Llama-3.3-70B-Instruct"; "qwen" = "Qwen2.5-72B-Instruct"
              "mistral2501" = "Mistral-Small-24B-Instruct-2501"; "mistral31" = "Mistral-Small-3.1-24B-Instruct-2503"
              "gemma3" = "gemma-3-27b-it" }
if (-not $RunBase) { $RunBase = Split-Path $PSScriptRoot -Parent }       # deployed at <RunBase>\_queue\queue.ps1
$Q = Join-Path $RunBase "_queue"
New-Item -ItemType Directory -Force -Path $Q | Out-Null
$status = Join-Path $Q "queue.status"
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

$lockFile = Join-Path $Q "queue.lock"
try { $lock = [System.IO.File]::Open($lockFile, 'OpenOrCreate', 'ReadWrite', 'None') } catch { exit 0 }

function LastStatus([string]$text) {
    # last "[<job id>] STATUS (elapsed)" in a chunk of nnsight spinner output
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

$firstRunning = @{}     # job id -> when this guard first saw it RUNNING / DISPATCHED

function Guard([string]$folder) {
    $codeDir = Join-Path (Join-Path $RunBase $folder) "code"
    $logs = Join-Path (Join-Path $RunBase $folder) "logs"
    $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
               Where-Object { $_.CommandLine -and $_.CommandLine.Contains($codeDir) })
    $shards = @{}
    foreach ($p in $procs) {
        $c = $p.CommandLine
        $stage = $null
        if ($c -match '1_generate_ndif\.py') { $stage = "generate" }
        elseif ($c -match '3_judge_ndif\.py') { $stage = "judge" }
        elseif ($c -match '2_activations_ndif\.py' -and $c -match 'persona\\(vectors_\w+)') { $stage = $Matches[1] }
        if (-not $stage -or $c -notmatch '--shard\s+(\d)/3') { continue }
        $key = "$stage.$($Matches[1])"
        if (-not $shards.ContainsKey($key)) { $shards[$key] = @() }
        $shards[$key] += $p.ProcessId
    }
    foreach ($key in $shards.Keys) {
        $log = Join-Path $logs "$key.log"
        if (-not (Test-Path $log)) { continue }
        $silent = ((Get-Date) - (Get-Item $log).LastWriteTime).TotalMinutes
        $st = LastStatus (Tail $log)
        $why = $null
        if ($st -and ($st.State -eq "RUNNING" -or $st.State -eq "DISPATCHED")) {
            if (-not $firstRunning.ContainsKey($st.Job)) { $firstRunning[$st.Job] = Get-Date }
            $run = ((Get-Date) - $firstRunning[$st.Job]).TotalMinutes
            if ($run -ge $StallMinutes) { $why = "job $($st.Job) $($st.State) for $([int]$run) min" }
        }
        if (-not $why -and $silent -ge $StallMinutes -and -not ($st -and $st.State -eq "QUEUED")) {
            $why = "log silent for $([int]$silent) min"
        }
        # 2026-10-10: the engineio b'P' fault kills socketio's receive thread; the shard then waits forever for a
        # result that can no longer arrive (seen 4 times in one day). A shard whose .err.log ENDS in that error is
        # dead now -- a shard that recovered would have logged further lines after it.
        if (-not $why) {
            $errLines = @((Tail (Join-Path $logs "$key.err.log") 2048) -split "[\r\n]+" | Where-Object { $_.Trim() })
            if ($errLines.Count -and $errLines[-1] -match "invalid literal for int\(\) with base 10: b'P'") {
                $why = "socketio receive thread died (engineio b'P')"
            }
        }
        if ($why) {
            Note "STALL $folder $key ($why): stopping pids $($shards[$key] -join ',')"
            foreach ($id in $shards[$key]) { Stop-Process -Id $id -Force -EA SilentlyContinue }
        }
    }
}

$Repo = Split-Path (Split-Path $RunBase -Parent) -Parent      # <repo>\random_persona_bank\results -> <repo>
$LlamaRoot = Join-Path $RunBase $folders["llama33"]
$LlamaSteps = @(
    @{ Name = "llama K"; Script = Join-Path $Repo "xmodel\k\k_chain.ps1"; Done = Join-Path $LlamaRoot "K\k_chain.status"
       Args = @("-Model", "llama-70b-3.3", "-OutRoot", "`"$(Join-Path $LlamaRoot 'K')`"",
                "-LogDir", "`"$(Join-Path $Q 'llama-K-logs')`"", "-Gate", "`"$(Join-Path $LlamaRoot 'logs\chain.status')`"") },
    @{ Name = "llama alpaca"; Script = Join-Path $Repo "xmodel\alpaca_chain.ps1"; Done = Join-Path $LlamaRoot "logs\alpaca.status"
       Args = @("-Run", "`"$LlamaRoot`"", "-LogDir", "`"$(Join-Path $Q 'llama-alpaca-logs')`"",
                "-Model", "meta-llama/Llama-3.3-70B-Instruct", "-CsvModel", "llama-v3.3-70b-instruct",
                "-DriveBase", $DriveBase) + $(if ($Rclone) { @("-Rclone", "`"$Rclone`"") } else { @() }) }
)

function StepDone($s) { (Test-Path $s.Done) -and ((Get-Content $s.Done -Tail 1) -match '  DONE$') }

function FindRunning([string]$needle) {
    # a powershell already running this script (from an earlier queue instance)
    Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -EA SilentlyContinue |
        Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -and $_.CommandLine.Contains($needle) } |
        Select-Object -First 1
}

function WaitPid([int]$id, [string]$folder) {
    while (Get-Process -Id $id -EA SilentlyContinue) {
        Start-Sleep -Seconds 60
        if ($folder) { try { Guard $folder } catch { Note "guard error: $($_.Exception.Message)" } }
    }
}

function LlamaSteps([string]$root) {
    foreach ($s in $LlamaSteps) {
        if (StepDone $s) { continue }
        $running = FindRunning $s.Script
        if ($running) {
            Note "$($s.Name) already running (pid $($running.ProcessId)): attaching"
            WaitPid $running.ProcessId ""
        } else {
            Note "$($s.Name) start"
            $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$($s.Script)`"") + $s.Args
            $p = Start-Process -FilePath "powershell.exe" -ArgumentList $a -WindowStyle Hidden -PassThru
            $null = $p.Handle
            WaitPid $p.Id ""
        }
        $ok = StepDone $s
        Note "$($s.Name) done=$ok"
        if (-not $ok) { Note "QUEUE PAUSED at $($s.Name) until the next trigger"; $lock.Close(); exit 1 }
    }
}

function AllDone() {
    foreach ($tag in $Tags) {
        $cs = Join-Path (Join-Path $RunBase $folders[$tag]) "logs\chain.status"
        if (-not ((Test-Path $cs) -and ((Get-Content $cs -Tail 1) -match '^DONE '))) { return $false }
        if ($tag -eq "llama33") { foreach ($s in $LlamaSteps) { if (-not (StepDone $s)) { return $false } } }
    }
    return $true
}
if (AllDone) { $lock.Close(); exit 0 }      # silent: the watchdog keeps firing after the last model
Note "QUEUE START tags=$($Tags -join ',') stall=${StallMinutes}min"
foreach ($tag in $Tags) {
    $folder = $folders[$tag]
    $root = Join-Path $RunBase $folder
    $chainStatus = Join-Path $root "logs\chain.status"
    if ((Test-Path $chainStatus) -and ((Get-Content $chainStatus -Tail 1) -match '^DONE ')) {
        if ($tag -eq "llama33") { LlamaSteps $root }     # a chain DONE before its extra steps ran
        continue
    }
    $chain = Join-Path $root "code\run_chain.ps1"
    if (-not (Test-Path $chain)) { Note "STOP $tag not deployed ($chain missing)"; break }
    $running = FindRunning "$folder\code\run_chain.ps1"
    if ($running) {
        Note "chain $tag already running (pid $($running.ProcessId)): attaching"
        WaitPid $running.ProcessId $folder
        $code = "n/a (attached)"
    } else {
        Note "chain $tag start"
        $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$chain`"", "-Tag", $tag,
               "-RunBase", "`"$RunBase`"", "-DriveBase", $DriveBase)
        if ($Py) { $a += @("-Py", "`"$Py`"") }
        if ($Rclone) { $a += @("-Rclone", "`"$Rclone`"") }
        $p = Start-Process -FilePath "powershell.exe" -ArgumentList $a -WindowStyle Hidden -PassThru
        $null = $p.Handle
        WaitPid $p.Id $folder
        $code = $p.ExitCode
    }
    $done = (Test-Path $chainStatus) -and ((Get-Content $chainStatus -Tail 1) -match '^DONE ')
    Note "chain $tag exit $code done=$done"
    if (-not $done) { Note "QUEUE PAUSED at $tag until the next trigger"; $lock.Close(); exit 1 }
    if ($tag -eq "llama33") { LlamaSteps $root }
}
if (AllDone) { Note "QUEUE DONE (all $($Tags.Count) models)" }
$lock.Close()
