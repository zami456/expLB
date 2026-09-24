"""P17 lab-PC READ-ONLY fetch helper.

Sends a small Python program on STDIN to the lab's analysis python over
`ssh -o BatchMode=yes -o ConnectTimeout=15` (no file is uploaded, nothing is written on the lab PC).
The program reads the requested files and prints one JSON bundle
{path: {bytes, mtime, md5, b64}} to stdout. Locally, each file's md5 is recomputed from the decoded
bytes and must equal the lab-side md5 (transport check; this is NOT scp, which truncates at 204,800 B).

Fetched files are cached under audit/p17/labcache/<lab path> with a sidecar .meta.json
(lab md5 + mtime) so re-runs are offline unless --refresh is passed to the caller.
"""
import base64, hashlib, json, os, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
HOST = "user4@100.85.217.68"
PY = r"C:\Users\user4\expLB\.venv-analysis\Scripts\python.exe"
CACHE = ROOT / "audit" / "p17" / "labcache"

_PROG = r'''
import sys, os, json, glob, hashlib, base64, datetime
req = json.loads(sys.stdin.readline())
out = {}
for pat in req["patterns"]:
    hits = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
    for p in hits:
        try:
            b = open(p, "rb").read()
            st = os.stat(p)
            out[p.replace("\\", "/")] = dict(bytes=len(b), mtime=datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                          md5=hashlib.md5(b).hexdigest(), b64=(base64.b64encode(b).decode() if req.get("content", True) else None))
        except Exception as e:
            out[p.replace("\\", "/")] = dict(error=repr(e))
sys.stdout.write(json.dumps(out))
'''


class LabUnreachable(RuntimeError):
    pass


def _run(patterns, content=True, timeout=600):
    # the program itself goes as `-c`; the request goes on stdin
    prog_b64 = base64.b64encode(_PROG.encode()).decode()
    boot = f"import base64;exec(base64.b64decode('{prog_b64}').decode())"
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, f'"{PY}" -c "{boot}"']
    try:
        p = subprocess.run(cmd, input=(json.dumps({"patterns": patterns, "content": content}) + "\n").encode(),
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise LabUnreachable(f"timeout: {e}")
    if p.returncode != 0:
        raise LabUnreachable(p.stderr.decode("utf-8", "replace")[-500:])
    return json.loads(p.stdout.decode("utf-8"))


def _cache_path(labpath):
    rel = labpath.replace("C:/Users/user4/", "").replace(":", "")
    return CACHE / rel


def fetch(patterns, refresh=False):
    """Return {labpath: (bytes_content, meta)}; meta has lab md5/mtime/bytes. Uses cache when possible."""
    if isinstance(patterns, str):
        patterns = [patterns]
    # listing first (no content) so cached files can be matched by md5
    need_listing = refresh or any(any(c in p for c in "*?[") for p in patterns) or \
        not all(_cache_path(p).exists() for p in patterns)
    result = {}
    if not need_listing:
        for p in patterns:
            cp = _cache_path(p)
            meta = json.loads(cp.with_name(cp.name + ".meta.json").read_text())
            result[p] = (cp.read_bytes(), meta)
        return result
    listing = _run(patterns, content=False)
    todo = []
    for p, m in listing.items():
        if "error" in m:
            result[p] = (None, m)
            continue
        cp = _cache_path(p)
        if not refresh and cp.exists() and hashlib.md5(cp.read_bytes()).hexdigest() == m["md5"]:
            result[p] = (cp.read_bytes(), m)
        else:
            todo.append(p)
    for i in range(0, len(todo), 40):
        got = _run(todo[i:i + 40], content=True)
        for p, m in got.items():
            b = base64.b64decode(m.pop("b64"))
            if hashlib.md5(b).hexdigest() != m["md5"] or len(b) != m["bytes"]:
                raise RuntimeError(f"transport md5 mismatch for {p}")
            cp = _cache_path(p)
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(b)
            cp.with_name(cp.name + ".meta.json").write_text(json.dumps(m))
            result[p] = (b, m)
    return result


def reachable():
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, "echo ok"],
                           capture_output=True, timeout=40)
        return r.returncode == 0 and b"ok" in r.stdout
    except Exception:
        return False


if __name__ == "__main__":
    import sys
    r = fetch(sys.argv[1:], refresh=True)
    for p, (b, m) in r.items():
        print(p, m)
