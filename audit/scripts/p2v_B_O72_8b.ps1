# READ-ONLY: count the 8B rung's rollouts and scores on the lab PC (verify batch B, item O72).
$ErrorActionPreference = 'Stop'
$r = 'C:\Users\user4\expLB\run-scale\8b'
$tot = 0; $nf = 0
Get-ChildItem -LiteralPath "$r\responses" -Filter *.jsonl -File | Sort-Object Name | ForEach-Object {
  $n = @(Get-Content -LiteralPath $_.FullName).Count
  $tot += $n; $nf++
  '{0,-30} {1,5}' -f $_.BaseName, $n
}
"RESPONSES: $nf files, $tot rows"
$st = 0; $sf = 0
Get-ChildItem -LiteralPath "$r\scores" -Filter *.jsonl -File | ForEach-Object {
  $st += @(Get-Content -LiteralPath $_.FullName).Count; $sf++
}
"SCORES: $sf files, $st rows"
"--- which roles have responses but no scores ---"
$a = (Get-ChildItem -LiteralPath "$r\responses" -File).BaseName
$b = (Get-ChildItem -LiteralPath "$r\scores" -File).BaseName
Compare-Object $a $b | ForEach-Object { '{0} {1}' -f $_.SideIndicator, $_.InputObject }
"--- vectors_s23 / vectors_all counts ---"
'vectors_all  = ' + @(Get-ChildItem -LiteralPath "$r\vectors_all" -File).Count
'vectors_s23  = ' + @(Get-ChildItem -LiteralPath "$r\vectors_s23" -File).Count
