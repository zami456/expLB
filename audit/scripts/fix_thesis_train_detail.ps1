# READ-ONLY: what does train.py do (API calls? local GPU?), and has it run since 2026-08-04?
"--- D:/user4fina/results (latest 8)"
Get-ChildItem 'D:\user4fina\results' -File -EA SilentlyContinue | Sort-Object LastWriteTime -Desc | Select-Object -First 8 | ForEach-Object { $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm') + '  ' + $_.Length + '  ' + $_.Name }
"--- secrets present (names only, contents NOT read)"
Get-ChildItem 'D:\user4fina\secrets' -File -EA SilentlyContinue | ForEach-Object { $_.Name + '  ' + $_.Length + ' B' }
"--- train.py: size, and the lines that show what it calls"
$p = 'D:\user4fina\code\train.py'
if (-not (Test-Path $p)) { "NOT FOUND: $p" } else {
  (Get-Item $p).Length.ToString() + " B, mtime " + (Get-Item $p).LastWriteTime
  Select-String -LiteralPath $p -Pattern 'import |openai|anthropic|api_key|API_KEY|requests|http|cuda|device|from_pretrained|Trainer|epochs|batch_size|dataset|model_name|\.to\(' |
    Select-Object -First 40 | ForEach-Object { '  ' + $_.LineNumber + ': ' + $_.Line.Trim() }
}
