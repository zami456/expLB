"""Upload a file to the lab PC over ssh, in base64 chunks.

Why chunks: a single `ssh ... powershell -Command "..."` line fails above roughly
8 KB ("The command line is too long"), which a base64-encoded source file exceeds
immediately. Chunked Add-Content then a single FromBase64String decode is the only
route that has worked reliably here. `tar -cf - | tar -xf -` works lab -> laptop
but not laptop -> lab in this setup.

Verifies by md5 on both sides and refuses to leave a partial file in place.

  python audit/scripts/labpc_put.py <local file> <remote path>
"""
import base64
import hashlib
import subprocess
import sys
from pathlib import Path

HOST = "user4@100.85.217.68"
CHUNK = 4000  # base64 chars per ssh call; keeps the command line well under the limit


def ps(cmd, timeout=120):
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST,
         f'powershell -NoProfile -Command "{cmd}"'],
        capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    local, remote = Path(sys.argv[1]), sys.argv[2]
    data = local.read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    b64 = base64.b64encode(data).decode("ascii")
    tmp = remote + ".b64"

    print(f"local  : {local}  ({len(data)} bytes, md5 {md5})")
    print(f"remote : {remote}")
    print(f"chunks : {-(-len(b64) // CHUNK)} x {CHUNK} chars")

    rc, _, err = ps(f"if (Test-Path '{tmp}') {{ Remove-Item '{tmp}' -Force }}; "
                    f"New-Item -ItemType File -Path '{tmp}' -Force | Out-Null")
    if rc != 0:
        raise SystemExit(f"could not create the staging file: {err}")

    for n, i in enumerate(range(0, len(b64), CHUNK), 1):
        part = b64[i:i + CHUNK]
        rc, _, err = ps(f"Add-Content -Path '{tmp}' -Value '{part}' -NoNewline")
        if rc != 0:
            ps(f"Remove-Item '{tmp}' -Force -ErrorAction SilentlyContinue")
            raise SystemExit(f"chunk {n} failed: {err}")
        if n % 5 == 0 or i + CHUNK >= len(b64):
            print(f"  .. {min(i + CHUNK, len(b64))}/{len(b64)} chars")

    rc, out, err = ps(
        f"[IO.File]::WriteAllBytes('{remote}', "
        f"[Convert]::FromBase64String((Get-Content '{tmp}' -Raw))); "
        f"Remove-Item '{tmp}' -Force; "
        f"(Get-FileHash '{remote}' -Algorithm MD5).Hash.ToLower()")
    if rc != 0:
        raise SystemExit(f"decode failed: {err}")

    remote_md5 = out.splitlines()[-1].strip() if out else ""
    if remote_md5 == md5:
        print(f"VERIFIED: remote md5 {remote_md5} == local")
    else:
        ps(f"Remove-Item '{remote}' -Force -ErrorAction SilentlyContinue")
        raise SystemExit(f"MD5 MISMATCH local {md5} != remote {remote_md5}; remote file removed")


if __name__ == "__main__":
    main()
