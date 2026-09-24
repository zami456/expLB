"""Verify batch B: pull result files from the lab PC into audit/cache/labpc/ (read-only on the lab).

Method is the Phase-1 verified one: ssh "tar -cf - -C <dir> <name>" | local tar -x.
Verifies each pulled file's md5 against the lab-side md5 (Get-FileHash) and appends to
audit/cache/MANIFEST.csv.

Usage: python audit/scripts/p2v_B_pull.py <labdir-forward-slashes> <name> [<name> ...]
       (lab dir must be under C:/Users/user4/expLB; files are written to
        audit/cache/labpc/<relative path under expLB>/)
"""
import datetime, hashlib, io, os, subprocess, sys, tarfile

HOST = "user4@100.85.217.68"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LABROOT = "C:/Users/user4/expLB/"


def sh(cmd, **kw):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, cmd],
                          capture_output=True, timeout=600, **kw)


def main():
    labdir = sys.argv[1].rstrip("/")
    names = sys.argv[2:]
    assert labdir.startswith(LABROOT), labdir
    rel = labdir[len(LABROOT):]
    dest = os.path.join(ROOT, "audit", "cache", "labpc", *rel.split("/"))
    os.makedirs(dest, exist_ok=True)
    # lab-side md5 first (read-only)
    q = "; ".join("(Get-FileHash -Algorithm MD5 -LiteralPath '%s/%s').Hash + ' %s'" % (labdir, n, n)
                  for n in names)
    r = sh("powershell -NoProfile -NonInteractive -Command \"%s\"" % q)
    labmd5 = {}
    for line in r.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line:
            h, n = line.split(None, 1)
            labmd5[n.strip()] = h.lower()
    p = sh("tar -cf - -C %s %s" % (labdir, " ".join(names)))
    if p.returncode != 0:
        sys.exit("tar failed: " + p.stderr.decode("utf-8", "replace")[:500])
    with tarfile.open(fileobj=io.BytesIO(p.stdout)) as tf:
        tf.extractall(dest)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    man = os.path.join(ROOT, "audit", "cache", "MANIFEST.csv")
    with open(man, "a", encoding="utf-8") as mf:
        for n in names:
            fp = os.path.join(dest, n)
            b = open(fp, "rb").read()
            m = hashlib.md5(b).hexdigest()
            ok = "MATCH" if labmd5.get(n) == m else "MISMATCH(lab=%s)" % labmd5.get(n)
            print("%-34s %9d B  md5 %s  %s" % (n, len(b), m, ok))
            mf.write("%s,labpc,%s/%s,%d,%s,%s,%d,%s,%s,tar over ssh (verify batch B)\n"
                     % (now, labdir, n, len(b), labmd5.get(n, ""),
                        os.path.relpath(fp, ROOT).replace(os.sep, "/"), len(b), m, ok))


main()
