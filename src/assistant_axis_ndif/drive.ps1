<#
.SYNOPSIS
  Push pipeline output into Google Drive WITHOUT keeping a copy on this laptop,
  and report Drive health.

.DESCRIPTION
  Why rclone and not the Drive desktop app:

    * This laptop has exactly ONE real disk. C: has ~34 GB free; G: and H: are both
      Google Drive mounts, not storage (H: holds a single My Drive.lnk shortcut).
    * The Drive app's streaming mount has already dropped once mid-session while the
      app itself kept running, taking G: with it. A multi-hour run must not depend on it.
    * tamarayushra@gmail.com is in MIRROR mode, which is the opposite of what we want:
      mirrored files live on C: and consume the disk we are short of.
    * rclone talks to the Drive API directly. No mount, nothing to crash, no local
      copy, and -Move deletes the local file only after the upload verifies.

  ONE-TIME SETUP (needs your browser, and you must pick the right account):

      rclone config create tamara drive scope=drive

  A browser opens -- sign in as tamarayushra@gmail.com. Confirm with:

      .\src\assistant_axis_ndif\drive.ps1 -Check

.EXAMPLE
  .\src\assistant_axis_ndif\drive.ps1 -Check
  .\src\assistant_axis_ndif\drive.ps1 -Sync run-paper16 -Dest runs/paper16
  .\src\assistant_axis_ndif\drive.ps1 -Sync run/coords -Dest coords -Move
#>
[CmdletBinding()]
param(
    [switch]$Check,
    [string]$Sync,
    [string]$Dest,
    [string]$Remote = "tamara",
    [string]$Root = "expLB",
    # delete each local file once rclone has verified it landed -- this is what keeps
    # the 20 GB of activations off a 34 GB disk
    [switch]$Move,
    [int]$Transfers = 4
)

function Get-Rclone {
    $cmd = Get-Command rclone -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $glob = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages\Rclone.Rclone*\rclone*\rclone.exe'
    $hit = Get-ChildItem $glob -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { return $hit.FullName }
    throw "rclone not found. Install with: winget install --id Rclone.Rclone"
}

function Invoke-Check {
    param([string]$rclone)

    Write-Host "=== disk ==="
    # Only report free space for REAL disks. G: and H: are Google Drive mounts and the
    # number Windows reports for them is meaningless: H: is a shortcut whose target is a
    # folder on C:, so it just echoes C:'s free space, and G: returns a placeholder the
    # streaming filesystem invents. Neither relates to the account's quota -- that comes
    # only from `rclone about`, printed below.
    $notStorage = @{
        'G' = 'Google Drive STREAM mount (suraiyashanaz1) -- not storage'
        'H' = 'Google Drive shortcut -> tamarayushra mirror folder on C: -- not storage'
    }
    Get-PSDrive -PSProvider FileSystem | ForEach-Object {
        if ($notStorage.ContainsKey($_.Name)) {
            Write-Host ("  {0}:  {1}" -f $_.Name, $notStorage[$_.Name])
        } else {
            Write-Host ("  {0}:  {1,6:N1} GB free   (real disk)" -f $_.Name, ($_.Free / 1GB))
        }
    }

    Write-Host "`n=== Drive desktop app ==="
    $p = Get-Process GoogleDriveFS -ErrorAction SilentlyContinue
    if ($p) { Write-Host ("  running ({0} proc)" -f $p.Count) } else { Write-Host "  NOT running" }
    $mirror = "$env:USERPROFILE\My Drive (tamarayushra@gmail.com)"
    if (Test-Path $mirror) {
        $sz = (Get-ChildItem $mirror -Recurse -File -ErrorAction SilentlyContinue |
               Measure-Object Length -Sum).Sum
        Write-Host ("  tamarayushra MIRROR folder on C: -- {0:N2} GB" -f ($sz / 1GB))
        if ($sz -gt 5GB) {
            Write-Warning "  mirror folder is large and lives on C:; that is laptop disk, not cloud"
        }
    }
    $cache = "$env:LOCALAPPDATA\Google\DriveFS"
    if (Test-Path $cache) {
        $sz = (Get-ChildItem $cache -Recurse -File -ErrorAction SilentlyContinue |
               Measure-Object Length -Sum).Sum
        Write-Host ("  stream cache on C:            -- {0:N2} GB" -f ($sz / 1GB))
    }

    Write-Host "`n=== rclone ==="
    Write-Host "  binary  : $rclone"
    $remotes = & $rclone listremotes 2>$null
    if (-not $remotes) {
        Write-Warning "  no remote configured. Run:  rclone config create $Remote drive scope=drive"
        Write-Host   "  then sign in as tamarayushra@gmail.com when the browser opens."
        return $false
    }
    Write-Host "  remotes : $($remotes -join ' ')"
    if ($remotes -notcontains "${Remote}:") {
        Write-Warning "  remote '${Remote}:' not among them"
        return $false
    }
    $about = & $rclone about "${Remote}:" 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Warning "  '${Remote}:' unreachable: $about"; return $false }
    Write-Host "  quota   :"
    $about | ForEach-Object { Write-Host "    $_" }
    return $true
}

$rclone = Get-Rclone

if ($Check -or (-not $Sync)) {
    $ok = Invoke-Check -rclone $rclone
    if (-not $Sync) { exit $(if ($ok) { 0 } else { 1 }) }
}

if (-not (Test-Path $Sync)) { throw "source not found: $Sync" }
if (-not $Dest) { $Dest = Split-Path $Sync -Leaf }
$target = "${Remote}:$Root/$Dest"

$verb = if ($Move) { "move" } else { "copy" }
Write-Host "`n=== rclone $verb  $Sync  ->  $target ==="
if ($Move) {
    Write-Warning "-Move deletes local files after upload verifies. That is the point on a 34 GB disk."
}

# --checksum: compare by hash, not mtime. Drive rewrites mtimes and we would rather
#   re-upload than silently skip a file that differs.
# --create-empty-src-dirs: keep the run layout intact so resume logic still sees it.
& $rclone $verb $Sync $target `
    --transfers $Transfers --checkers 8 --checksum `
    --create-empty-src-dirs --progress --stats-one-line --stats 20s

if ($LASTEXITCODE -ne 0) { throw "rclone $verb failed with exit code $LASTEXITCODE" }

Write-Host "`n=== verifying ==="
& $rclone check $Sync $target --checksum --one-way 2>&1 | Select-Object -Last 5
if ($LASTEXITCODE -ne 0) { Write-Warning "verification reported differences - inspect before deleting anything" }
else { Write-Host "verified: every local file is present in $target with a matching hash" }
