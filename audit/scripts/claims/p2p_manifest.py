"""Log the P2-pooled cache copies (run-premise/ + six premise logs) in audit/cache/MANIFEST.csv."""
import csv
import hashlib
import os

MAN = "audit/cache/MANIFEST.csv"
BASE = "audit/cache/labpc"
LAB = "C:/Users/user4/expLB"
WHEN = "2026-09-23 03:05"

lab_md5 = {}
with open("audit/cache/_tmp/premise_md5_lab.txt", encoding="utf-8-sig") as f:
    for line in f:
        line = line.rstrip("\r\n")
        if "\t" in line:
            h, p = line.split("\t", 1)
            lab_md5[p.strip()] = h

existing = set()
with open(MAN, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        existing.add(r["local_path"])

rows = []
for sub in ("run-premise", "logs"):
    for root, _, fs in os.walk(os.path.join(BASE, sub)):
        for fn in fs:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, BASE).replace(os.sep, "/")
            local = f"{BASE}/{rel}"
            if local in existing:
                continue
            b = open(fp, "rb").read()
            h = hashlib.md5(b).hexdigest()
            src = lab_md5.get(rel)
            rows.append(dict(copied_at=WHEN, source="labpc", source_path=f"{LAB}/{rel}",
                             source_bytes=len(b) if src else "",
                             source_md5=src or "", local_path=local, local_bytes=len(b),
                             local_md5=h,
                             check="MATCH" if src == h else ("NOT_HASHED" if not src else "DIFF"),
                             method="tar over ssh (batch P2-pooled)"))

with open(MAN, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["copied_at", "source", "source_path", "source_bytes",
                                      "source_md5", "local_path", "local_bytes", "local_md5",
                                      "check", "method"])
    for r in rows:
        w.writerow(r)
print("appended", len(rows), "rows;",
      sum(1 for r in rows if r["check"] == "MATCH"), "verified MATCH,",
      sum(1 for r in rows if r["check"] == "NOT_HASHED"), "logs (not hash-compared),",
      sum(1 for r in rows if r["check"] == "DIFF"), "DIFF")
