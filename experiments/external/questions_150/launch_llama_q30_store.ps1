# Phase 3b-ii: the per-question store of the 30 ORIGINAL questions on Llama-3.3-70B (GATE-0 input).
# The seeded folder already holds the 5,400 rollouts and 5,250 scores, so the chain's generate and judge
# gates pass at once; stage 3b builds vectors_pq_<keep>_q30 at layers 20/36/40/44 for each keep set in
# -PqBanks (default s3 and s23). No all-layer bank (-Banks none). -Force because an earlier run of this
# launcher leaves chain.status at DONE. Detached, so it survives the launching session.
param([string]$PqBanks = "s3,s23")
$chain = "E:\qbank\banks\Llama-3.3-70B-Instruct\code\run_chain.ps1"
$qfile = "E:\qbank\banks\Llama-3.3-70B-Instruct\code\kd\questions\kd_questions_shared.jsonl"
$args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $chain,
          "-Tag", "llama33", "-RunBase", "E:\qbank\banks", "-Py", "E:\aletheia\venv\Scripts\python.exe", "-SkipEval", "-Force",
          "-NQuestions", "30", "-QFile", $qfile,
          "-QSha256", "1A5F7709E778669BE724475E146679B9D45E0144ED0485AC436A6E931868FFB7",
          "-MinCount", "20", "-Banks", "none", "-PerQuestion", "-PqLayers", "20,36,40,44", "-PqBanks", $PqBanks,
          "-VecSuffix", "_q30", "-MinDownloadKbps", "40", "-Shards", "3")
$p = Start-Process -FilePath "powershell.exe" -ArgumentList $args -WindowStyle Hidden -PassThru
"launched chain pid $($p.Id) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (q30 store, PqBanks $PqBanks)" | Out-File -Append -Encoding utf8 "E:\qbank\logs\launches.txt"
"pid $($p.Id)"
