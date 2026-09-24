# READ-ONLY: which files did the backup verification flag?
$L = 'C:\Users\user4\audit_scratch\out'
foreach ($f in Get-ChildItem $L -Filter 'backup_check_*.log') {
  "=== " + $f.Name + " (" + $f.Length + " B)"
  Get-Content $f.FullName | Where-Object { $_ -match 'ERROR|differ|missing|NOTICE' } | Select-Object -First 12
  Get-Content $f.FullName -Tail 3
}
