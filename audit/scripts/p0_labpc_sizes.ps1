# Phase 0, READ-ONLY: per-folder size, file count, result-type counts, newest write time on the lab PC.
# Output: one CSV line per folder (level-1 subfolders of each root, plus loose files of the root).
$ErrorActionPreference = 'SilentlyContinue'
$roots = @('C:\Users\user4\expLB','C:\Users\user4\expLBactivationLAYER22','C:\Users\user4\expLBmask','C:\Users\user4\expLBresults','C:\Users\user4\expLBsrc','C:\Users\user4\run-crossgen','C:\Users\user4\share_deceptive','C:\Users\user4\share_L20_shared','C:\Users\user4\acrivatins','D:\expLB-acts','D:\Thesis','D:\user4','D:\user4fina')
$exts = @('.json','.jsonl','.csv','.npy','.npz','.pt','.log','.parquet','.png','.pdf','.md','.py','.ps1','.safetensors','.bin')
'root,folder,files,bytes,newest,' + (($exts | ForEach-Object { $_.TrimStart('.') }) -join ',')
foreach ($root in $roots) {
  if (-not (Test-Path -LiteralPath $root)) { "$root,NOT_FOUND"; continue }
  $subs = @(Get-ChildItem -LiteralPath $root -Force -Directory | Where-Object { $_.Name -notmatch '^\.venv|^\.pip-cache$|^AppData$' })
  $targets = @($subs | ForEach-Object { $_.FullName }) + @('<loose>')
  foreach ($t in $targets) {
    if ($t -eq '<loose>') { $files = @(Get-ChildItem -LiteralPath $root -Force -File); $name = '<loose files>' }
    else { $files = @(Get-ChildItem -LiteralPath $t -Force -File -Recurse); $name = Split-Path $t -Leaf }
    $bytes = ($files | Measure-Object Length -Sum).Sum; if (-not $bytes) { $bytes = 0 }
    $newest = ($files | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
    $nstr = if ($newest) { $newest.ToString('yyyy-MM-dd HH:mm') } else { '' }
    $counts = foreach ($e in $exts) { @($files | Where-Object { $_.Extension -eq $e }).Count }
    "$root,$name,$($files.Count),$bytes,$nstr," + ($counts -join ',')
  }
}
'END'
