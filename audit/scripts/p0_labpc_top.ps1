# Phase 0, READ-ONLY: list drives and top-level folders of candidate result roots on the lab PC.
Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{n='UsedGB';e={[math]::Round($_.Used/1GB,1)}}, @{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}} | Format-Table -AutoSize | Out-String -Width 200
foreach ($root in @('C:\Users\user4\expLB','D:\expLB-acts','D:\','C:\Users\user4')) {
  if (Test-Path $root) {
    "=== $root"
    Get-ChildItem -LiteralPath $root -Force -ErrorAction SilentlyContinue | Select-Object Mode, LastWriteTime, Length, Name | Format-Table -AutoSize | Out-String -Width 200
  } else { "=== $root : NOT FOUND" }
}
