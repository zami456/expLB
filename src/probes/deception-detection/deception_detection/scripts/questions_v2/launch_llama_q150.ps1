# Phase 3: the 150-question Llama-3.3-70B run. Generates only ids 30-149 (resume by key), judges only the
# new keys, builds the per-question store vectors_pq_s3_q150 (layers 20/36/40/44) and the all-layer bank
# vectors_s3_q150. -Force because the q30 store run leaves chain.status at DONE. Detached.
$chain = "E:\qbank\banks\Llama-3.3-70B-Instruct\code\run_chain.ps1"
$qfile = "E:\qbank\banks\Llama-3.3-70B-Instruct\code\kd\questions\kd_questions_shared_v2.jsonl"
$args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $chain,
          "-Tag", "llama33", "-RunBase", "E:\qbank\banks", "-Py", "E:\aletheia\venv\Scripts\python.exe", "-SkipEval", "-Force",
          "-NQuestions", "150", "-QFile", $qfile,
          "-QSha256", "D5BCDD705D558CE24BED391E4E4ADE1D94AEC21173FF616F96B934797B4E1694",
          "-MinCount", "20", "-Banks", "s3", "-PerQuestion", "-PqLayers", "20,36,40,44", "-PqBanks", "s3",
          "-VecSuffix", "_q150", "-MinDownloadKbps", "40", "-Shards", "3", "-Attempts", "8")
$p = Start-Process -FilePath "powershell.exe" -ArgumentList $args -WindowStyle Hidden -PassThru
"launched chain pid $($p.Id) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Append -Encoding utf8 "E:\qbank\logs\launches.txt"
"pid $($p.Id)"
