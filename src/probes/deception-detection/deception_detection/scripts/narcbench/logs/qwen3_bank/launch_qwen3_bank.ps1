# Launch the Qwen3-32B persona-bank chain detached (narcbench, 2026-10-23), as E:\qbank\launch_task.ps1 does:
# Win32_Process.Create spawns it under the WMI provider host, outside the caller's job object. Run it from Claude
# Code's PowerShell tool or a plain PowerShell, never the Bash tool (its sandbox kills escaped processes in ~15 s).
#
#   .\launch_qwen3_bank.ps1                # the s3 store at layers 26-30 (the pre-registered bank)
#   .\launch_qwen3_bank.ps1 -PqBanks s23   # only if s3 leaves a role under 20 kept replies (the pre-registered rule)
#
# Every stage resumes per key, so relaunching the same line continues a stopped run. The chain refuses a question
# file whose SHA-256 differs from the thesis bank's 30 shared questions.
param(
    [ValidateSet("s3", "s23")][string]$PqBanks = "s3",
    [int]$Shards = 2,
    [string]$Py = "E:\aletheia\venv08\Scripts\python.exe"
)
$chain = "E:\narcbench\persona\Qwen3-32B\code\run_chain.ps1"
$qfile = "E:\narcbench\persona\Qwen3-32B\code\kd\questions\kd_questions_shared.jsonl"
$qsha  = "1A5F7709E778669BE724475E146679B9D45E0144ED0485AC436A6E931868FFB7"
$a = @("-Tag", "qwen3", "-RunBase", "E:\narcbench\persona", "-Py", $Py, "-SkipEval", "-Force",
       "-NQuestions", "30", "-QFile", $qfile, "-QSha256", $qsha, "-MinCount", "20", "-Banks", "none",
       "-PerQuestion", "-PqLayers", "26,27,28,29,30", "-PqBanks", $PqBanks, "-VecSuffix", "_q30",
       "-MinDownloadKbps", "40", "-Shards", "$Shards", "-GenBatch", "16", "-JudgeBatch", "8",
       "-StoreShards", "1", "-PqSessionBatches", "8", "-PqTokenBudget", "4000", "-Attempts", "8")
$quoted = ($a | ForEach-Object { if ($_ -match '[\s,]') { "`"$_`"" } else { $_ } }) -join " "
$cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$chain`" $quoted"
$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = (Split-Path $chain) }
"$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  WMI launch qwen3_bank $PqBanks rc=$($res.ReturnValue) pid=$($res.ProcessId): $cmd" |
    Tee-Object -Append -FilePath "E:\narcbench\persona\launches.txt"
