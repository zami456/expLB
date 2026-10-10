<#
Full backup of the lab PC's thesis data to tamara:expLB/labpc-backup-2026-10-06/
(owner request 2026-10-06, before the lab PC goes off for maintenance).

Owner decisions (2026-10-06): the two run-crossgen trees (215k tiny files) are packed as one
.tar per top-level folder plus a per-file md5 manifest; everything else is copied file by file;
ATresearch is included without its .venv.

Left out on purpose: the two Python venvs (package lists are saved instead), .cache (HF models),
.lmstudio, .vscode*, Downloads, acrivatins, ansel, Favorites, D:\expLB-acts (venv + pip cache),
.ssh, audit_scratch\rclone (login secrets), every .env file (API keys) and SPACE_RESERVE_50GB.bin
(a 50 GB disk-space placeholder in the home folder).

Resumable and idempotent: finished packs and finished copies leave a marker; rclone copy skips
files already on Drive. Run it again after a reboot and it continues.
Writes on the lab PC ONLY under C:\Users\user4\audit_scratch\backup\.
Drive: only `rclone copy` and `rclone check`, only into the new folder above, with --immutable
so nothing that is already there can be overwritten. Never sync / move / delete.
Status: audit_scratch\backup\backup.status (last line "ALL DONE" when finished).
#>
$ErrorActionPreference = "Continue"
$H    = "C:\Users\user4"
$W    = "$H\audit_scratch\backup"
$P    = "$W\pack"
$L    = "$W\logs"
$rc   = "$H\expLB\tools\rclone.exe"
$conf = "$H\AppData\Roaming\rclone\rclone.conf"
$DST  = "tamara:expLB/labpc-backup-2026-10-06"
$status = "$W\backup.status"
New-Item -ItemType Directory -Force $P, $L | Out-Null

function Note($m) { Add-Content -Path $status -Value ((Get-Date).ToString("yyyy-MM-dd HH:mm:ss") + "  " + $m) }

if ((Test-Path $status) -and (Select-String -Path $status -Pattern "ALL DONE" -Quiet)) { exit 0 }
Note "START pid $PID"

$freeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
if ($freeGB -lt 10) { Note "ABORT only $freeGB GB free on C:"; exit 1 }

# ---------------------------------------------------------------- 1. pack the crossgen trees
$tar = "$env:SystemRoot\System32\tar.exe"
$packSrc = @(
    @{ name = "expLB_run-crossgen";         path = "$H\expLB\run-crossgen" },
    @{ name = "expLB-resplit_run-crossgen"; path = "$H\expLB-resplit\run-crossgen" }
)
foreach ($s in $packSrc) {
    $done = "$P\$($s.name).done"
    if (Test-Path $done) { continue }
    $out = "$P\$($s.name)"
    New-Item -ItemType Directory -Force $out | Out-Null
    $ok = $true
    foreach ($it in Get-ChildItem $s.path -Directory) {
        $t = "$out\$($it.Name).tar"
        & $tar -cf $t -C $s.path $it.Name 2>> "$L\tar.err"
        if ($LASTEXITCODE -ne 0) { $ok = $false; Note "PACK FAIL $($s.name)\$($it.Name) tar exit $LASTEXITCODE"; continue }
        $nTar = @(& $tar -tf $t | Where-Object { $_ -notmatch '/$' }).Count
        $nSrc = @(Get-ChildItem $it.FullName -Recurse -File -Force).Count
        if ($nTar -ne $nSrc) { $ok = $false; Note "PACK MISMATCH $($s.name)\$($it.Name) tar=$nTar src=$nSrc" }
        else { Note "pack $($s.name)\$($it.Name).tar files=$nTar" }
    }
    $rootFiles = @(Get-ChildItem $s.path -File -Force | ForEach-Object { $_.Name })
    if ($rootFiles.Count) {
        & $tar -cf "$out\_rootfiles.tar" -C $s.path @rootFiles 2>> "$L\tar.err"
        if ($LASTEXITCODE -ne 0) { $ok = $false; Note "PACK FAIL $($s.name)\_rootfiles" }
        else { Note "pack $($s.name)\_rootfiles.tar files=$($rootFiles.Count)" }
    }
    & $rc md5sum $s.path --config $conf --output-file "$out\_source_files.md5" 2>> "$L\md5sum.err"
    if ($LASTEXITCODE -ne 0) { $ok = $false; Note "MD5SUM FAIL $($s.name)" }
    else { Note ("md5 manifest $($s.name): " + @(Get-Content "$out\_source_files.md5").Count + " files") }
    if ($ok) { New-Item -ItemType File -Force $done | Out-Null; Note "PACK DONE $($s.name)" }
}

# ---------------------------------------------- 2. snapshots: live logs, package lists, tasks
$snap = "$P\snapshots"
if (-not (Test-Path "$P\snapshots.done")) {
    New-Item -ItemType Directory -Force "$snap\expLB-logs", "$snap\scheduled-tasks" | Out-Null
    # expLB\logs is still appended to by the lab's own sync tasks, so copy a frozen snapshot
    robocopy "$H\expLB\logs" "$snap\expLB-logs" /E /R:2 /W:2 /NFL /NDL /NJH /NJS | Out-Null
    $rcopy = $LASTEXITCODE
    foreach ($v in @("$H\expLB\.venv-aa", "$H\expLB\.venv-analysis", "$H\ATresearch\.venv")) {
        $tag = ($v.Replace("$H\", "") -replace '[\\.]+', '_').Trim('_')
        $sp = "$v\Lib\site-packages"
        if (Test-Path $sp) {
            Get-ChildItem $sp -Directory -Filter *.dist-info | ForEach-Object { $_.Name -replace '\.dist-info$', '' } |
                Sort-Object | Set-Content "$snap\packages_$tag.txt"
        }
        if (Test-Path "$v\pyvenv.cfg") { Copy-Item "$v\pyvenv.cfg" "$snap\pyvenv_$tag.cfg" }
    }
    # definitions of every scheduled task that runs our scripts (several exist only on this box)
    foreach ($t in Get-ScheduledTask -TaskPath "\" -EA SilentlyContinue) {
        $a = ($t.Actions | ForEach-Object { "$($_.Execute) $($_.Arguments) $($_.WorkingDirectory)" }) -join " "
        if ($a -match 'user4') { Export-ScheduledTask -TaskName $t.TaskName | Set-Content "$snap\scheduled-tasks\$($t.TaskName).xml" }
    }
    if ($rcopy -lt 8) { New-Item -ItemType File -Force "$P\snapshots.done" | Out-Null; Note "SNAPSHOTS DONE (robocopy $rcopy)" }
    else { Note "SNAPSHOTS FAIL robocopy $rcopy" }
}

# ------------------------------------------------------------------- 3. copy, then check
$common = @("--config", $conf, "--immutable", "--transfers", "8", "--checkers", "16",
            "--retries", "5", "--low-level-retries", "20", "--exclude", "**/.env")
$jobs = @(
    @{ n = "expLB-resplit";          src = "$H\expLB-resplit";          f = @("--exclude", "/run-crossgen/**") },
    # SPACE_RESERVE_50GB.bin is a 50 GB disk-space placeholder, not data (excluded 2026-10-06 18:25)
    @{ n = "home-loose-files";       src = "$H";                        f = @("--max-depth", "1", "--ignore-case", "--exclude", "/ntuser*", "--exclude", "/SPACE_RESERVE*") },
    @{ n = "audit_scratch";          src = "$H\audit_scratch";          f = @("--exclude", "/rclone/**", "--exclude", "/backup/**") },
    @{ n = "share_deceptive";        src = "$H\share_deceptive";        f = @() },
    @{ n = "share_L20_shared";       src = "$H\share_L20_shared";       f = @() },
    @{ n = "expLBactivationLAYER22"; src = "$H\expLBactivationLAYER22"; f = @() },
    @{ n = "ATresearch";             src = "$H\ATresearch";             f = @("--exclude", "/.venv/**") },
    @{ n = "packed";                 src = $P;                          f = @("--exclude", "/*.done") },
    @{ n = "expLB";                  src = "$H\expLB";                  f = @("--exclude", "/.venv-aa/**", "--exclude", "/.venv-analysis/**",
                                                                              "--exclude", "/run-crossgen/**", "--exclude", "/logs/**") }
)
foreach ($j in $jobs) {
    if (Test-Path "$L\$($j.n).copied") { continue }
    if ($j.n -eq "packed" -and -not ((Test-Path "$P\snapshots.done") -and @(Get-ChildItem $P -Filter *_run-crossgen.done).Count -eq 2)) {
        Note "copy packed SKIPPED (packs not complete)"; continue
    }
    Note "copy $($j.n) START"
    $f = $j.f
    & $rc copy $j.src "$DST/$($j.n)" @common @f --log-file "$L\copy_$($j.n).log" --log-level INFO --stats 2m --stats-one-line 2>> "$L\rclone_stderr.log"
    $e = $LASTEXITCODE
    Note "copy $($j.n) exit $e"
    if ($e -eq 0) { New-Item -ItemType File -Force "$L\$($j.n).copied" | Out-Null }
}

$allOk = $true
foreach ($j in $jobs) {
    if (-not (Test-Path "$L\$($j.n).copied")) { $allOk = $false; continue }
    if (Test-Path "$L\$($j.n).checked") { continue }
    $f = $j.f
    & $rc check $j.src "$DST/$($j.n)" --one-way --config $conf --checkers 16 --exclude "**/.env" @f `
        --missing-on-dst "$L\$($j.n).missing.txt" --differ "$L\$($j.n).differ.txt" --error "$L\$($j.n).error.txt" `
        --log-file "$L\check_$($j.n).log" --log-level NOTICE 2>> "$L\rclone_stderr.log"
    $e = $LASTEXITCODE
    $miss = if (Test-Path "$L\$($j.n).missing.txt") { @(Get-Content "$L\$($j.n).missing.txt").Count } else { -1 }
    $diff = if (Test-Path "$L\$($j.n).differ.txt") { @(Get-Content "$L\$($j.n).differ.txt").Count } else { -1 }
    Note "check $($j.n) exit $e missing=$miss differ=$diff"
    if ($e -eq 0) { New-Item -ItemType File -Force "$L\$($j.n).checked" | Out-Null } else { $allOk = $false }
}

if ($allOk) { Note "ALL DONE" } else { Note "PASS INCOMPLETE - the next trigger retries what is not marked done" }
