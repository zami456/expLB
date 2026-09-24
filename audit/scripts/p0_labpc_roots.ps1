# Phase 0, READ-ONLY: compact listing of candidate roots on the lab PC (names only, dirs marked with /).
$ErrorActionPreference = 'SilentlyContinue'
foreach ($root in @('D:\','D:\expLB-acts','C:\Users\user4','C:\Users\user4\expLB')) {
  if (Test-Path -LiteralPath $root) {
    $items = Get-ChildItem -LiteralPath $root -Force
    "=== $root  (" + $items.Count + " entries)"
    ($items | Where-Object { $_.PSIsContainer } | ForEach-Object { $_.Name + '/' }) -join '  '
    "files: " + (($items | Where-Object { -not $_.PSIsContainer }).Count)
  } else { "=== $root : NOT FOUND" }
}
"=== END"
