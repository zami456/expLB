"""Run a PowerShell script on the lab PC, read-only by design of the scripts passed to it.

Usage:  python audit/scripts/labpc_run.py <script.ps1> [KEY=VALUE ...] [--out FILE] [--timeout SEC]

- Every __KEY__ in the script is replaced by VALUE (use forward slashes in paths; scripts normalise
  them with Resolve-Path / .Replace).  This avoids the Bash tool's backslash mangling.
- The script is sent with -EncodedCommand (UTF-16LE base64), with `ssh -o BatchMode=yes`, so it can
  never hang on a password prompt.
- PowerShell CLIXML progress noise is dropped. stdout goes to --out (binary-safe) or to our stdout.
"""
import base64, subprocess, sys

HOST = "user4@100.85.217.68"


def main():
    args = sys.argv[1:]
    out, timeout = None, 600
    if "--out" in args:
        i = args.index("--out"); out = args[i + 1]; del args[i:i + 2]
    if "--timeout" in args:
        i = args.index("--timeout"); timeout = int(args[i + 1]); del args[i:i + 2]
    script = open(args[0], encoding="utf-8").read()
    for kv in args[1:]:
        k, v = kv.split("=", 1)
        script = script.replace(f"__{k}__", v)
    if "__" in script and any(f"__{t}__" in script for t in ("ROOT", "PATH", "OUT", "FILE")):
        sys.exit("unreplaced placeholder left in script")
    enc = base64.b64encode(script.encode("utf-16-le")).decode()
    if len(enc) < 7000:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST,
               f"powershell -NoProfile -NonInteractive -EncodedCommand {enc}"]
    else:
        # Too long for a Windows command line: upload the (substituted) script into the approved scratch
        # folder with tar over ssh (verified byte-faithful), then run it with -File. Logged in LABPC_WRITES.md.
        import io, os, tarfile
        name = os.path.basename(args[0])
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
            data = ("\ufeff" + script).encode("utf-8")  # BOM so Windows PowerShell 5.1 reads UTF-8
            ti = tarfile.TarInfo(name); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
        up = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST,
                             "tar -xf - -C C:/Users/user4/audit_scratch/scripts"], input=buf.getvalue(),
                            capture_output=True, timeout=120)
        if up.returncode != 0:
            sys.exit("upload failed: " + up.stderr.decode("utf-8", "replace"))
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST,
               f"powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File C:/Users/user4/audit_scratch/scripts/{name}"]
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    lines = [l for l in p.stdout.splitlines(keepends=True)
             if not l.startswith(b"#< CLIXML") and not l.startswith(b"<Objs")]
    data = b"".join(lines)
    if out:
        open(out, "wb").write(data)
    else:
        sys.stdout.buffer.write(data)
    err = p.stderr.decode("utf-8", "replace")
    err = "\n".join(l for l in err.splitlines() if "CLIXML" not in l and not l.startswith("<Objs"))
    if err.strip():
        sys.stderr.write(err + "\n")
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
