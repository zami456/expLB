# READ-ONLY: list immediate children of __ROOT__ with file counts per subfolder.
$root = (Resolve-Path -LiteralPath '__ROOT__').Path
Get-ChildItem -LiteralPath $root -Force | ForEach-Object {
  if ($_.PSIsContainer) { $n = @(Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Force).Count; 'D ' + $n + ' ' + $_.Name }
  else { 'F ' + $_.Length + ' ' + $_.Name }
}
