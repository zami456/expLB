"""Upload local audit files into C:/Users/user4/audit_scratch/scripts/ on the lab PC (tar over ssh, verified
byte-faithful in Phase 1) and append a line to audit/LABPC_WRITES.md for each file.
Usage: python audit/scripts/labpc_upload.py [--dest scripts|census|out] <file> [<file> ...]"""
import datetime, io, os, subprocess, sys, tarfile

HOST = "user4@100.85.217.68"
LOG = os.path.join(os.path.dirname(__file__), "..", "LABPC_WRITES.md")
args = sys.argv[1:]
dest = "scripts"
if args and args[0] == "--dest":
    dest = args[1]; args = args[2:]
assert dest in ("scripts", "census", "out")
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
    for f in args:
        tf.add(f, arcname=os.path.basename(f))
p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST,
                    f"tar -xf - -C C:/Users/user4/audit_scratch/{dest}"], input=buf.getvalue(), capture_output=True, timeout=120)
if p.returncode:
    sys.exit("upload failed: " + p.stderr.decode("utf-8", "replace"))
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
with open(LOG, "a", encoding="utf-8") as log:
    for f in args:
        log.write(f"| {now} | upload | `audit_scratch/{dest}/{os.path.basename(f)}` | audit file (source: {f}) | {os.path.getsize(f)} B |\n")
print("uploaded", len(args), "file(s) to", dest)
