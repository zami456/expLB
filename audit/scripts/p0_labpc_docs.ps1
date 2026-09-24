# Phase 0, READ-ONLY: every .md/.txt/.html/.pdf/.tex/.docx doc on the lab PC store (outside venvs and the big
# run-crossgen / run-coords trees), with size, mtime, md5. Output CSV: path,bytes,mtime,md5
$ErrorActionPreference = 'SilentlyContinue'
$exts = @('.md','.txt','.html','.pdf','.tex','.docx')
$skip = @('.venv-aa','.venv-analysis','.venv','.pip-cache','run-crossgen','run-coords')
$roots = 'C:\Users\user4\expLB','D:\expLB-acts','C:\Users\user4\share_deceptive','C:\Users\user4\share_L20_shared'
'path,bytes,mtime,md5'
foreach ($r in $roots) {
  $top = Get-ChildItem -LiteralPath $r -Force
  $files = @($top | Where-Object { -not $_.PSIsContainer })
  foreach ($d in ($top | Where-Object { $_.PSIsContainer -and ($skip -notcontains $_.Name) })) { $files += @(Get-ChildItem -LiteralPath $d.FullName -Recurse -File -Force) }
  foreach ($f in $files) {
    if (($exts -contains $f.Extension.ToLower()) -and ($f.FullName -notlike '*site-packages*')) {
      '"' + $f.FullName + '",' + $f.Length + ',' + $f.LastWriteTime.ToString('yyyy-MM-dd HH:mm') + ',' + (Get-FileHash -Algorithm MD5 -LiteralPath $f.FullName).Hash
    }
  }
}
'END'
