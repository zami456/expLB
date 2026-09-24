"""Phase 1 second pass, run ON the lab PC. READ-ONLY on owner files; writes only audit_scratch/census/.
For each text-type file in labpc_census.tsv (<= 50 MB): md5 after CRLF->LF, plus whether the raw file had CR.
Lets copies that differ only by line endings (Windows writers vs git LF blobs) be matched."""
import hashlib, os
C = "C:/Users/user4/audit_scratch/census/"
EXT = {".json", ".jsonl", ".csv", ".md", ".txt", ".py", ".ps1", ".yaml", ".yml", ".log", ".sh", ".status", ".tsv", ".html", ".cmd"}
n = 0
with open(C + "labpc_census.tsv", encoding="utf-8") as fin, open(C + "labpc_census_eol.tsv", "w", encoding="utf-8", newline="\n") as fout:
    next(fin)
    fout.write("root\trelpath\tmd5_lf\thas_cr\n")
    for line in fin:
        root, rel, size = line.rstrip("\n").split("\t")[:3]
        if os.path.splitext(rel)[1].lower() not in EXT or int(size) > 50 * 2**20:
            continue
        try:
            b = open(os.path.join(root, rel), "rb").read()
        except OSError:
            continue
        cr = b"\r" in b
        fout.write(f"{root}\t{rel}\t{hashlib.md5(b.replace(b'\r\n', b'\n') if cr else b).hexdigest()}\t{int(cr)}\n")
        n += 1
print("EOL pass done:", n, "text files")
