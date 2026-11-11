# Phase 7: the 150-question Qwen3.5-27B run in the isolated folder (the Aletheia bank folder is never written).
# Generates ids 30-149, judges the new keys, builds vectors_pq_s3_q150 at the organism grid 16-32 and the
# all-block bank vectors_s3_q150. Detached.
$chain = "E:\qbank\banks\Qwen3.5-27B\code\run_chain.ps1"
$qfile = "E:\qbank\banks\Qwen3.5-27B\code\kd\questions\kd_questions_shared_v2.jsonl"
$args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $chain,
          "-Tag", "qwen35", "-RunBase", "E:\qbank\banks", "-Py", "E:\aletheia\venv\Scripts\python.exe", "-SkipEval", "-Force",
          "-NQuestions", "150", "-QFile", $qfile,
          "-QSha256", "D5BCDD705D558CE24BED391E4E4ADE1D94AEC21173FF616F96B934797B4E1694",
          "-MinCount", "20", "-Banks", "s3", "-PerQuestion", "-PqLayers", "16,19,22,26,29,32", "-PqBanks", "s3",
          "-VecSuffix", "_q150", "-MinDownloadKbps", "40", "-Shards", "3", "-Attempts", "8")
$p = Start-Process -FilePath "powershell.exe" -ArgumentList $args -WindowStyle Hidden -PassThru
"launched chain pid $($p.Id) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Append -Encoding utf8 "E:\qbank\logs\launches.txt"
"pid $($p.Id)"
