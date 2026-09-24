"""Phase 1 census of git trees (read-only): one row per blob with md5 of its content.
Usage: python audit/scripts/p1_git_census.py <ref> <out.tsv> [path-prefix]
md5 is computed over the blob bytes as stored in git (LF endings for text), via `git cat-file --batch`."""
import hashlib, subprocess, sys
ref, out = sys.argv[1], sys.argv[2]
prefix = sys.argv[3] if len(sys.argv) > 3 else ""
ls = subprocess.run(["git", "ls-tree", "-r", "-l", ref] + ([prefix] if prefix else []), capture_output=True, check=True).stdout.decode("utf-8", "replace")
entries = []
for line in ls.splitlines():
    meta, path = line.split("\t", 1)
    mode, typ, sha, size = meta.split()
    if typ == "blob":
        entries.append((path, sha, int(size)))
p = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write("ref\tpath\tbytes\tblob_sha\tmd5\n")
    for path, sha, size in entries:
        p.stdin.write((sha + "\n").encode()); p.stdin.flush()
        header = p.stdout.readline().split()
        n = int(header[2]); data = p.stdout.read(n); p.stdout.read(1)
        f.write(f"{ref}\t{path}\t{n}\t{sha}\t{hashlib.md5(data).hexdigest()}\n")
p.stdin.close(); p.wait()
print(ref, len(entries), "blobs ->", out)
