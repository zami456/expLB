# W6G part 2: whole small logs / targeted blocks. Read-only.
$ErrorActionPreference = 'Continue'
$L = 'C:\Users\user4\expLB\logs'

Write-Output "#### q8_smoke.log : layer 44 block"
$c = Get-Content (Join-Path $L 'q8_smoke.log')
$i = ($c | Select-String -Pattern 'layer 44 --' | Select-Object -First 1).LineNumber
Write-Output ("  start line " + $i + " of " + $c.Count)
$c[($i-1)..([Math]::Min($i+24, $c.Count-1))] | ForEach-Object { Write-Output ("  | " + $_) }

Write-Output ""
Write-Output "#### scale_8b_axes.log : FULL"
Get-Content (Join-Path $L 'scale_8b_axes.log') | ForEach-Object { Write-Output ("  | " + $_) }

Write-Output ""
Write-Output "#### logs matching scale_8b / cap_axes / cap_vectors"
Get-ChildItem $L -File | Where-Object { $_.Name -match 'scale_8b|cap_axes|cap_vectors|cap_ctrl_axes|cap_ctrl_vectors' } |
  ForEach-Object { Write-Output ("  " + $_.Name + "  " + $_.Length + "  " + $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')) }
