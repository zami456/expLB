"""Upload an audit script into the lab scratch folder C:/Users/user4/audit_scratch/scripts.
Read-only w.r.t. owner files; writes only inside the approved scratch folder.
Usage: python audit/scripts/claims/P9_upload.py <local_file> [...]"""
import io
import os
import subprocess
import sys
import tarfile

HOST = "user4@100.85.217.68"

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
    for src in sys.argv[1:]:
        data = open(src, "rb").read()
        ti = tarfile.TarInfo(os.path.basename(src))
        ti.size = len(data)
        ti.mtime = 0
        tf.addfile(ti, io.BytesIO(data))

cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST,
       "powershell -NoProfile -NonInteractive -Command "
       "\"Set-Location C:/Users/user4/audit_scratch/scripts; tar -xf -\""]
p = subprocess.run(cmd, input=buf.getvalue(), capture_output=True, timeout=300)
print("rc", p.returncode)
print(p.stdout.decode("utf-8", "replace")[:2000])
print(p.stderr.decode("utf-8", "replace")[:2000])
