# Launch a chain run detached from the launching session.
#
# Start-Process children died with the Claude Code session on 2026-10-16 (the session's processes sit in a
# job object), a scheduled task with an interactive logon was killed within a second (0xC000013A) and an
# S4U task needs elevation. Win32_Process.Create spawns the chain under the WMI provider host instead, outside
# any job of the caller, in the current user's session, with a hidden window.
#
#   .\launch_task.ps1 -Which llama_q150
#   .\launch_task.ps1 -Which qwen35_q150
#   .\launch_task.ps1 -Which llama_q30_store [-PqBanks s23]
param(
    [Parameter(Mandatory = $true)][ValidateSet("llama_q150", "qwen35_q150", "llama_q30_store")][string]$Which,
    [string]$PqBanks = "s3,s23",
    # 2026-10-17 16:50: NDIF 0.8 executes concurrent requests on one replica; two gen_batch-32 generation
    # streams exceed the Llama-3.3-70B replica's headroom (per-process cap 37.90 GiB) and OOM in pairs.
    # Generation therefore runs with -Shards 1; judge / store stages are relaunched with -Shards 3.
    [int]$Shards = 3,
    [int]$GenBatch = 32,
    [int]$JudgeBatch = 16,
    # 2026-10-18: module-output stages (store, bank): shards (0 = -Shards; use 1 under NDIF 0.8) and traces/session
    [int]$StoreShards = 0,
    [int]$PqSessionBatches = 20,
    [int]$BankSessionBatches = 20,
    [int]$BankBatchSize = 8,
    [int]$PqTokenBudget = 12000,
    [int]$BankTokenBudget = 12000,
    # 2026-10-17: NDIF was upgraded to 0.8 mid-run; the nnsight 0.7 venv (E:\aletheia\venv) can no longer connect
    # (socket.io path 404, server subscribes at /subscribe). venv08 = nnsight 0.8.0 with the same torch 2.6.0+cpu /
    # transformers 5.8.0 pins, validated 2026-10-17 02:20 (trace, generation session, accumulator session, judge).
    [string]$Py = "E:\aletheia\venv08\Scripts\python.exe"
)
$py = $Py
$v2sha = "D5BCDD705D558CE24BED391E4E4ADE1D94AEC21173FF616F96B934797B4E1694"
$origsha = "1A5F7709E778669BE724475E146679B9D45E0144ED0485AC436A6E931868FFB7"
$runs = @{
    "llama_q150" = @{
        chain = "E:\qbank\banks\Llama-3.3-70B-Instruct\code\run_chain.ps1"
        args = @("-Tag", "llama33", "-RunBase", "E:\qbank\banks", "-Py", $py, "-SkipEval", "-Force",
                 "-NQuestions", "150", "-QFile", "E:\qbank\banks\Llama-3.3-70B-Instruct\code\kd\questions\kd_questions_shared_v2.jsonl",
                 "-QSha256", $v2sha, "-MinCount", "20", "-Banks", "s3", "-PerQuestion", "-PqLayers", "20,36,40,44",
                 "-PqBanks", "s3,s23", "-VecSuffix", "_q150", "-MinDownloadKbps", "40", "-Shards", "$Shards", "-GenBatch", "$GenBatch", "-JudgeBatch", "$JudgeBatch", "-StoreShards", "$StoreShards", "-PqSessionBatches", "$PqSessionBatches", "-BankSessionBatches", "$BankSessionBatches", "-BankBatchSize", "$BankBatchSize", "-BankTokenBudget", "$BankTokenBudget", "-PqTokenBudget", "$PqTokenBudget", "-Attempts", "8") }
    "qwen35_q150" = @{
        chain = "E:\qbank\banks\Qwen3.5-27B\code\run_chain.ps1"
        args = @("-Tag", "qwen35", "-RunBase", "E:\qbank\banks", "-Py", $py, "-SkipEval", "-Force",
                 "-NQuestions", "150", "-QFile", "E:\qbank\banks\Qwen3.5-27B\code\kd\questions\kd_questions_shared_v2.jsonl",
                 "-QSha256", $v2sha, "-MinCount", "20", "-Banks", "s3", "-PerQuestion", "-PqLayers", "16,19,22,26,29,32",
                 "-PqBanks", "s3,s23", "-VecSuffix", "_q150", "-MinDownloadKbps", "40", "-Shards", "$Shards", "-GenBatch", "$GenBatch", "-JudgeBatch", "$JudgeBatch", "-StoreShards", "$StoreShards", "-PqSessionBatches", "$PqSessionBatches", "-BankSessionBatches", "$BankSessionBatches", "-BankBatchSize", "$BankBatchSize", "-BankTokenBudget", "$BankTokenBudget", "-PqTokenBudget", "$PqTokenBudget", "-Attempts", "8") }
    "llama_q30_store" = @{
        chain = "E:\qbank\banks\Llama-3.3-70B-Instruct\code\run_chain.ps1"
        args = @("-Tag", "llama33", "-RunBase", "E:\qbank\banks", "-Py", $py, "-SkipEval", "-Force",
                 "-NQuestions", "30", "-QFile", "E:\qbank\banks\Llama-3.3-70B-Instruct\code\kd\questions\kd_questions_shared.jsonl",
                 "-QSha256", $origsha, "-MinCount", "20", "-Banks", "none", "-PerQuestion", "-PqLayers", "20,36,40,44",
                 "-PqBanks", $PqBanks, "-VecSuffix", "_q30", "-MinDownloadKbps", "40", "-Shards", "$Shards") }
}
$r = $runs[$Which]
$quoted = ($r.args | ForEach-Object { if ($_ -match '[\s,]') { "`"$_`"" } else { $_ } }) -join " "
$cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$($r.chain)`" $quoted"
$startup = (Get-CimClass -ClassName Win32_ProcessStartup).CimClassInstance ; $startup = $null
$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = (Split-Path $r.chain) }
"$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  WMI launch $Which rc=$($res.ReturnValue) pid=$($res.ProcessId): $cmd" | Out-File -Append -Encoding utf8 "E:\qbank\logs\launches.txt"
"launched $Which via WMI: return $($res.ReturnValue), pid $($res.ProcessId)"
