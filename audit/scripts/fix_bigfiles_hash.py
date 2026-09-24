"""Hash the lab files that the census skipped (>200 MB) and print md5 + size, so they can be checked against Drive
instead of trusted on name+size. Read-only; runs on the lab PC. Output: audit_scratch/out/bigfiles_md5.tsv"""
import hashlib, os

SRC = "C:/Users/user4/audit_scratch/census/labpc_census.tsv"
OUT = "C:/Users/user4/audit_scratch/out/bigfiles_md5.tsv"
rows = []
with open(SRC, encoding="utf-8") as f:
    next(f)
    for line in f:
        root, rel, size, mtime, ctime, md5 = line.rstrip("\n").split("\t")
        if md5 == "SKIP_BIG":
            rows.append((root, rel, size))
print(len(rows), "files to hash")
with open(OUT, "w", encoding="utf-8", newline="\n") as out:
    out.write("root\trelpath\tbytes\tmd5\n")
    for i, (root, rel, size) in enumerate(rows, 1):
        p = os.path.join(root, rel)
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                h.update(chunk)
        out.write(f"{root}\t{rel}\t{size}\t{h.hexdigest()}\n")
        if i % 10 == 0:
            print(i, "done", flush=True)
print("written:", OUT)
