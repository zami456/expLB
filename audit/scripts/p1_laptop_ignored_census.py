"""Phase 1: md5 of every git-IGNORED file in the laptop working tree (tracked files are covered by the git census).
Skips .git, .venv*, audit/, submodules, caches. Chunked reads (low memory)."""
import hashlib, os, subprocess, datetime
root = "C:/Users/SURAIYA/Documents/STUDY/thesis/expLB"
ign = subprocess.run(["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"], cwd=root,
                     capture_output=True, check=True).stdout.decode("utf-8", "replace").split("\0")
skip = (".git/", ".venv", "audit/", "mask/", "pacchiardi/", "soft-trigger/", "assistant_axis/", "__pycache__",
        ".pytest_cache", "dishonesty_bench.egg-info", ".claude/")
rows = 0
with open(root + "/audit/phase1/census_laptop_ignored.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("path\tbytes\tmtime\tmd5\n")
    for rel in ign:
        if not rel or any(s in rel for s in skip):
            continue
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            continue
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        st = os.stat(p)
        f.write(f"{rel}\t{st.st_size}\t{datetime.datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M:%S}\t{h.hexdigest()}\n")
        rows += 1
print(rows, "ignored files hashed")
