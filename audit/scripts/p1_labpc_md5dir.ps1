# READ-ONLY on the lab PC: "md5 size relpath" for every file under __ROOT__ (forward slashes OK).
$root = (Resolve-Path -LiteralPath '__ROOT__').Path.TrimEnd([char]92)
Get-ChildItem -LiteralPath $root -Recurse -File | ForEach-Object {
  (Get-FileHash -Algorithm MD5 -LiteralPath $_.FullName).Hash.ToLower() + ' ' + $_.Length + ' ' + $_.FullName.Substring($root.Length + 1).Replace([char]92, '/')
}
