"""Phase 1 Task B, list 2 (ghosts): every file/folder path the Task-B doc corpus cites, checked against every
census source (lab, Drive, asix HEAD, his branch, laptop-ignored). Read-only; streams the census (small memory).

Token extraction: backtick spans and bare path-like words that either start with a known result root
(run-*, run/, kd/run, kd/results, cap/run*, logs/, results/, figures/, share/, staging/, activations/, smoke/,
src/probes/deception-detection/..., persona vectors/..., data_syco, papers/) or end in a result/script extension.
Brace forms are expanded ({a,b} -> a, b; {0..76} -> every integer in the range is NOT enumerated: the range
becomes a digit wildcard). '*', '<...>', '$var', '…' and 'N' placeholders become wildcards.
Matching is case-insensitive on path SUFFIXES at segment boundaries, against the raw path of each source AND
its canonical lab-style form (p1_anom_common.canon), so 'kd11/banks/kd/summary.json' matches lab
'run-crossgen/kd11/banks/kd/summary.json' and Drive 'kd/crossgen11/banks/kd/summary.json'. A token also matches
a DIRECTORY when some file lies under it.
Output: audit/phase1/anomalies_detail/cited_paths.tsv  (token, kind, found_in, n_matches, example_match, docs, first_ref)
Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p1_anom_ghosts.py
"""
import collections, csv, gzip, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_anom_common import canon, load_docs, _expand_token, REPO

csv.field_size_limit(10**9)
P1 = os.path.join(REPO, "audit", "phase1")
ROOTS = ("run-", "run/", "kd/run", "kd/results", "cap/run", "logs/", "results/", "figures/", "share/", "staging/",
         "activations/", "smoke/", "aa-smoke", "src/probes/deception-detection/", "persona vectors/", "data_syco",
         "persona including", "persona knowledge", "deceptive/", "paper16_l20", "thinking/", "coords/", "crossgen/",
         "own100/", "paper16/", "hf275/", "scale/", "sycophancy", "thesis-docs/", "figures-crossgen/", "expLBactivation".lower())
RESEXT = (".json", ".jsonl", ".csv", ".pt", ".npz", ".npy", ".log", ".status", ".parquet", ".pkl", ".txt", ".png",
          ".pdf", ".zip", ".err", ".yaml", ".xlsx")
SCRIPTEXT = (".py", ".ps1", ".sh", ".ipynb")
TICK = re.compile(r"`([^`\n]{2,300})`")
BARE = re.compile(r"(?<![\w/.-])((?:[\w@.\-{},*]+/)+[\w@.\-*{},<>…]+/?|[\w\-*{},<>…]+\.(?:json|jsonl|csv|pt|npz|log|status|parquet|py|ps1))(?![\w/])")


def clean(t):
    t = t.strip().strip("'\"()[],;:").replace("\\", "/")
    for pre in ("c:/users/user4/explb/", "c:/users/user4/", "tamara:explb/", "./", "explb/", "d:/"):
        if t.startswith(pre):
            t = t[len(pre):]
    while t.startswith("../"):
        t = t[3:]
    t = re.sub(r":l?\d+(-\d+)?$", "", t)
    t = re.sub(r"/{2,}", "/", t)
    t = re.sub(r"\s+$", "", t)
    return t.rstrip(".")


def expand_pipes(t):
    """a/b|c/d -> a/b/d, a/c/d (alternation inside one path segment)."""
    segs = t.split("/")
    outs = [""]
    for sg in segs:
        alts = sg.split("|") if "|" in sg else [sg]
        outs = [(o + "/" + a) if o else a for o in outs for a in alts]
    return outs[:50]


def is_candidate(t):
    if not t or " " in t.strip("/") and not t.startswith(("persona vectors", "persona including", "persona knowledge")):
        return False
    if t.startswith(("http", "www.", "arxiv", "doi")) or "://" in t or t.count("/") > 12:
        return False
    low = t.lower()
    if low.startswith(ROOTS):
        return True
    base = low.rstrip("/").split("/")[-1]
    return base.endswith(RESEXT + SCRIPTEXT)


def kind(t):
    b = t.rstrip("/").split("/")[-1]
    if b.endswith(SCRIPTEXT):
        return "script"
    if b.endswith(RESEXT):
        return "file"
    return "dir"


def to_regex(t):
    """literal path with wildcards -> regex that must match a whole path suffix at a segment boundary."""
    s = re.escape(t.rstrip("/"))
    s = s.replace(r"\*", "[^/]*").replace("…", "[^/]*").replace(r"\.\.\.", "[^/]*")
    s = re.sub(r"<[^>]*>", "[^/]+", s.replace(r"\<", "<").replace(r"\>", ">"))
    s = re.sub(r"\\\$\w+|\$\w+", "[^/]+", s)
    s = s.replace(re.escape("#DIGITS#"), r"\d+").replace("#DIGITS#", r"\d+")
    return re.compile(r"(?:^|/)" + s + r"(?:/|$)")


docs = load_docs(expand=False)
tokens = collections.defaultdict(lambda: {"docs": set(), "first": None})
for d, text in docs.items():
    for i, line in enumerate(text.splitlines(), 1):
        cands = [m.group(1) for m in TICK.finditer(line)]
        cands += [m.group(1) for m in BARE.finditer(line)]
        for c in cands:
            for part in re.split(r"\s+(?:and|or|vs|via|\+|,)\s+|;\s*|,\s+", c):
                t = clean(part)
                if not is_candidate(t):
                    continue
                # ranges {0..76} become a digit wildcard; alternatives are expanded
                t2 = re.sub(r"\{(\d+)\.\.(\d+)\}", "#DIGITS#", t)
                for e in [x for y in _expand_token(t2) for x in expand_pipes(y)]:
                    e = clean(e)
                    if is_candidate(e.replace("#DIGITS#", "0")):
                        rec = tokens[e]
                        rec["docs"].add(d.split(":", 1)[1])
                        if rec["first"] is None:
                            rec["first"] = f"{d.split(':', 1)[1]}:{i}"

# index tokens by their longest literal segment
index = collections.defaultdict(list)
matchers = {}
for t in tokens:
    segs = [s for s in t.rstrip("/").split("/") if s]
    lit = [s for s in segs if not re.search(r"[*<>$…]|#DIGITS#|\.\.\.", s)]
    matchers[t] = to_regex(t)
    if not lit:
        index["__ALL__"].append(t)
        continue
    key = max(lit, key=len)
    index[key].append(t)
    matchers[t] = to_regex(t)
found = collections.defaultdict(lambda: {"src": collections.Counter(), "ex": ""})


def test(src, path):
    p = path.lower()
    segs = set(p.split("/")) | {"__ALL__"}
    for s in segs:
        for t in index.get(s, ()):
            if matchers[t].search(p):
                r = found[t]
                r["src"][src] += 1
                if not r["ex"]:
                    r["ex"] = f"{src}:{path}"


def both(src, raw, can):
    test(src, raw)
    if can.lower() != raw.lower():
        test(src, can)


with gzip.open(os.path.join(P1, "labpc", "joined_labpc.tsv.gz"), "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["root"] == "root":
            continue
        both("LAB", r["root"].replace("\\", "/").split("/")[-1] + "/" + r["relpath"], canon("LAB", r["relpath"], r["root"]))
with gzip.open(os.path.join(P1, "labpc", "drive_census.tsv.gz"), "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        both("DRIVE", r["path"], canon("DRIVE", r["path"]))
for src, fn in (("ASIX", "census_asix_git.tsv"), ("PCOOD", "census_pcood.tsv")):
    with open(os.path.join(P1, fn), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            test(src, r["path"])
with open(os.path.join(P1, "census_laptop_ignored.tsv"), encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        test("LAPTOP_IGN", r["path"])

out = os.path.join(P1, "anomalies_detail", "cited_paths.tsv")
n = collections.Counter()
with open(out, "w", encoding="utf-8", newline="\n") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["token", "kind", "found_in", "n_matches", "example_match", "n_docs", "docs", "first_ref"])
    for t in sorted(tokens):
        r = found.get(t)
        fi = ",".join(f"{k}:{v}" for k, v in sorted(r["src"].items())) if r else "NOT-FOUND"
        k = kind(t.replace("#DIGITS#", "0"))
        n[(k, "NOT-FOUND" if fi == "NOT-FOUND" else ("UNINDEXED" if fi == "UNINDEXED" else "found"))] += 1
        w.writerow([t.replace("#DIGITS#", "{N}"), k, fi, sum(r["src"].values()) if r else 0, r["ex"] if r else "",
                    len(tokens[t]["docs"]), ";".join(sorted(tokens[t]["docs"]))[:300], tokens[t]["first"]])
print("tokens:", len(tokens))
for k, v in sorted(n.items()):
    print(k, v)
