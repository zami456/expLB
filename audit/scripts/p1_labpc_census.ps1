# Phase 1 census, lab PC. READ-ONLY on the owner's files; writes ONLY to C:\Users\user4\audit_scratch\census\.
# One TSV row per file: root, relpath, bytes, LastWriteTime, CreationTime, md5 (files <= 200 MB; else "SKIP_BIG").
# Roots: the team's project folders under C:\Users\user4 and D:\expLB-acts. Venvs and pip caches are skipped.
$ErrorActionPreference = 'SilentlyContinue'
$out = 'C:\Users\user4\audit_scratch\census\labpc_census.tsv'
$log = 'C:\Users\user4\audit_scratch\census\labpc_census.log'
$roots = @('C:/Users/user4/expLB', 'D:/expLB-acts', 'C:/Users/user4/share_deceptive', 'C:/Users/user4/share_L20_shared',
           'C:/Users/user4/expLBactivationLAYER22', 'C:/Users/user4/tools')
$skipDirs = @('.venv', '.venv-aa', '.venv-analysis', '.pip-cache', '__pycache__', 'site-packages', 'node_modules')
$w = New-Object System.IO.StreamWriter($out, $false, (New-Object System.Text.UTF8Encoding($false)))
$w.WriteLine("root`trelpath`tbytes`tmtime`tctime`tmd5")
$md5 = [System.Security.Cryptography.MD5]::Create()
$n = 0; $t0 = Get-Date
function Walk([string]$dir, [string]$root) {
  foreach ($e in [System.IO.Directory]::EnumerateFileSystemEntries($dir)) {
    $name = [System.IO.Path]::GetFileName($e)
    if ([System.IO.Directory]::Exists($e)) {
      if ($skipDirs -notcontains $name) { Walk $e $root }
      continue
    }
    $fi = New-Object System.IO.FileInfo($e)
    $h = 'SKIP_BIG'
    if ($fi.Length -le 200MB) {
      try { $s = [System.IO.File]::OpenRead($e); $h = ([System.BitConverter]::ToString($md5.ComputeHash($s))).Replace('-', '').ToLower(); $s.Close() } catch { $h = 'ERR' }
    }
    $rel = $e.Substring($root.Length + 1).Replace([char]92, '/')
    $script:w.WriteLine("$root`t$rel`t$($fi.Length)`t$($fi.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))`t$($fi.CreationTime.ToString('yyyy-MM-dd HH:mm:ss'))`t$h")
    $script:n++
    if ($script:n % 5000 -eq 0) { $script:w.Flush(); Add-Content -LiteralPath $log -Value ("{0} files {1}" -f $script:n, (Get-Date).ToString('HH:mm:ss')) }
  }
}
Set-Content -LiteralPath $log -Value ("start " + $t0.ToString('yyyy-MM-dd HH:mm:ss'))
# loose files directly in C:\Users\user4 (drivers such as night_chain.ps1 live there); not recursive
$home4 = (Resolve-Path -LiteralPath 'C:/Users/user4').Path
foreach ($f in [System.IO.Directory]::EnumerateFiles($home4)) {
  $fi = New-Object System.IO.FileInfo($f)
  if ($fi.Extension -in @('.ps1', '.py', '.cmd', '.bat', '.json', '.txt', '.md', '.log', '.csv')) {
    $s = [System.IO.File]::OpenRead($f); $h = ([System.BitConverter]::ToString($md5.ComputeHash($s))).Replace('-', '').ToLower(); $s.Close()
    $w.WriteLine("C:\Users\user4`t$($fi.Name)`t$($fi.Length)`t$($fi.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))`t$($fi.CreationTime.ToString('yyyy-MM-dd HH:mm:ss'))`t$h")
    $n++
  }
}
foreach ($r in $roots) {
  if (-not (Test-Path -LiteralPath $r)) { Add-Content -LiteralPath $log -Value "missing root $r"; continue }
  $root = (Resolve-Path -LiteralPath $r).Path.TrimEnd([char]92)
  Walk $root $root
  Add-Content -LiteralPath $log -Value ("done root {0} at {1} files {2}" -f $root, (Get-Date).ToString('HH:mm:ss'), $n)
}
$w.Close()
Add-Content -LiteralPath $log -Value ("END {0} files, {1:N0} s" -f $n, ((Get-Date) - $t0).TotalSeconds)
"END $n files"
