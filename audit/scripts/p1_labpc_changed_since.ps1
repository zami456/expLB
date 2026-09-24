# READ-ONLY: files under the team's folders modified after __SINCE__ (excluding the audit's own scratch folder
# and the expected Drive-sync logs). Used to prove the audit changed nothing of the owner's.
$since = [datetime]::Parse('__SINCE__')
$roots = 'C:/Users/user4/expLB', 'C:/Users/user4/share_deceptive', 'C:/Users/user4/share_L20_shared', 'D:/expLB-acts'
foreach ($r in $roots) {
  Get-ChildItem -LiteralPath $r -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -gt $since -and $_.FullName -notmatch 'site-packages' } |
    ForEach-Object { $_.LastWriteTime.ToString('MM-dd HH:mm:ss') + '  ' + $_.Length + '  ' + $_.FullName }
}
Get-ChildItem -LiteralPath 'C:/Users/user4' -File -Force | Where-Object { $_.LastWriteTime -gt $since } | ForEach-Object { $_.LastWriteTime.ToString('MM-dd HH:mm:ss') + '  ' + $_.Length + '  ' + $_.FullName }
'END'
