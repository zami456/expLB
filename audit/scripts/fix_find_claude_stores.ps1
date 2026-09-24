# READ-ONLY: are there Claude session stores on the lab PC (any user profile)?
foreach ($u in (Get-ChildItem 'C:\Users' -Directory -EA SilentlyContinue)) {
  $c = Join-Path $u.FullName '.claude'
  if (Test-Path $c) {
    $proj = Join-Path $c 'projects'
    $n = 0; if (Test-Path $proj) { $n = @(Get-ChildItem $proj -Recurse -File -Filter *.jsonl -EA SilentlyContinue).Count }
    "  " + $u.Name + ": .claude present, " + $n + " session files"
    if ($n -gt 0) { Get-ChildItem $proj -Directory | ForEach-Object { "      " + $_.Name } }
  }
}
"(done)"
