# W6G: read-only listing of the stores the batch needs.
$ErrorActionPreference = 'Continue'
foreach ($d in @('C:\Users\user4\expLB\run-cap','C:\Users\user4\expLB\run-cap-ctrl','C:\Users\user4\expLB\run-kd','C:\Users\user4\expLB\run-scale\8b','C:\Users\user4\expLB\run-scale\3b','C:\Users\user4\expLB\cap')) {
  Write-Output ("#### " + $d)
  if (-not (Test-Path $d)) { Write-Output "  MISSING"; continue }
  Get-ChildItem $d -File | ForEach-Object { Write-Output ("  F " + $_.Name + "  " + $_.Length + "  " + $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm')) }
  Get-ChildItem $d -Directory | ForEach-Object {
    $n = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Output ("  D " + $_.Name + "/  files=" + $n)
  }
}
Write-Output "#### loose cap/kd geometry scripts in C:\Users\user4\expLB"
Get-ChildItem 'C:\Users\user4\expLB' -File | Where-Object { $_.Name -match '\.py$' -and $_.Name -match 'cap|geom|axes|pca|bank' } |
  ForEach-Object { Write-Output ("  " + $_.Name + "  " + $_.Length + "  " + $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm')) }
