"""Phase 1 Task B, list 1 (orphans): group every result file of every source into experiment/result UNITS and
check whether the Task-B doc corpus mentions each unit. Read-only; streams the census files (small memory).
Output: audit/phase1/anomalies_detail/units.tsv
Columns: unit kind sources n_lab n_drive n_drive_not_on_lab n_asix n_pcood n_laptop bytes_max mtime_min mtime_max
         lab_on_drive lab_not_on_drive mention_level mention_docs n_docs secondary_mentions
mention_level: PATH (full canonical path or a Drive-name variant), TAIL (last 2-3 path segments),
               NAME (last segment / file stem only), NONE.
Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p1_anom_units.py
"""
import collections, csv, gzip, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_anom_common import canon, unit, load_docs, load_secondary, norm, git_show, REPO, PCOOD, DRIVE_MAP

csv.field_size_limit(10**9)
P1 = os.path.join(REPO, "audit", "phase1")
OUT = os.path.join(P1, "anomalies_detail")
U = {}


def rec(u, kind):
    r = U.get(u)
    if r is None:
        r = U[u] = {"kind": kind, "n": collections.Counter(), "bytes": collections.Counter(), "mmin": "", "mmax": "",
                    "lab_on_drive": 0, "lab_not_on_drive": 0, "drive_not_on_lab": 0}
    return r


def add(src, cp, nbytes, mtime, extra=None):
    u, kind = unit(cp)
    if u is None:
        return
    r = rec(u, kind)
    r["n"][src] += 1
    r["bytes"][src] += nbytes
    if mtime and src in ("LAB", "LAPTOP_IGN"):
        if not r["mmin"] or mtime < r["mmin"]:
            r["mmin"] = mtime
        if mtime > r["mmax"]:
            r["mmax"] = mtime
    if extra:
        for k, v in extra.items():
            r[k] += v


with gzip.open(os.path.join(P1, "labpc", "drive_not_on_lab.tsv.gz"), "rt", encoding="utf-8") as f:
    dnl = {r["path"] for r in csv.DictReader(f, delimiter="\t")}
with gzip.open(os.path.join(P1, "labpc", "joined_labpc.tsv.gz"), "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["root"] == "root":
            continue
        od = int(r["on_drive"])
        add("LAB", canon("LAB", r["relpath"], r["root"]), int(r["bytes"] or 0), r["mtime"],
            {"lab_on_drive": od, "lab_not_on_drive": 1 - od})
with gzip.open(os.path.join(P1, "labpc", "drive_census.tsv.gz"), "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        add("DRIVE", canon("DRIVE", r["path"]), int(r["bytes"] or 0), "", {"drive_not_on_lab": int(r["path"] in dnl)})
for src, fn in (("ASIX", "census_asix_git.tsv"), ("PCOOD", "census_pcood.tsv")):
    with open(os.path.join(P1, fn), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            add(src, r["path"], int(r["bytes"] or 0), "")
with open(os.path.join(P1, "census_laptop_ignored.tsv"), encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        add("LAPTOP_IGN", r["path"], int(r["bytes"] or 0), r["mtime"])

docs = load_docs()
sec = load_secondary()
for k in list(sec):
    if sec[k] is None:
        ref = "HEAD" if k.startswith("code:asix:") else PCOOD
        sec[k] = norm(git_show(ref, k.split(":", 2)[2]))

REV = [(b, a) for a, b in DRIVE_MAP]
FMAP = {}
with open(os.path.join(P1, "fmap_all.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        FMAP[r["f_id"]] = norm(" ".join(r[k] for k in ("cited_paths", "resolved_primary", "key_path", "other_copies",
                                                       "producing_script", "run_log", "depends_on", "notes")))


def forms(u):
    """(level, regex) candidates, strongest first."""
    out = []
    is_family = "#" in u
    def rx(s):
        s = re.escape(s.lower())
        s = s.replace(r"\#", r"\d+").replace("#", r"\d+")
        return s
    path = u[1:] if u.startswith("@") else u
    variants = {path}
    for lab_pref, drv_pref in REV:
        if (u + "/").startswith(lab_pref) and not lab_pref.startswith("@"):
            variants.add(drv_pref + (u + "/")[len(lab_pref):].rstrip("/"))
        if lab_pref.startswith("@") and (u + "/").startswith(lab_pref):
            variants.add(drv_pref.rstrip("/"))
    if u.startswith("src/probes/deception-detection/"):
        variants.add(u[len("src/probes/deception-detection/"):])
    for v in sorted(variants):
        out.append(("PATH", r"(?<![\w-])" + rx(v) + r"(?![\w-])"))
    segs = path.split("/")
    for k in (3, 2):
        if len(segs) > k:
            out.append(("TAIL", r"(?<![\w-])" + rx("/".join(segs[-k:])) + r"(?![\w-])"))
    last = segs[-1]
    if last:
        out.append(("NAME", r"(?<![\w-])" + rx(last) + r"(?![\w-])"))
    return out


rows = []
for u, r in sorted(U.items()):
    fs = forms(u)
    level, hits = "NONE", []
    own = None
    if u.startswith("src/probes/deception-detection/results/"):
        own = "pcood:" + u + "/summary.md"
    for lv, pat in fs:
        c = re.compile(pat)
        hits = [d for d, t in docs.items() if d != own and c.search(t)]
        if hits:
            level = lv
            break
    sec_hits = []
    for lv, pat in fs:
        c = re.compile(pat)
        sh = [d for d, t in sec.items() if t and c.search(t)]
        if sh:
            sec_hits = [f"{lv}:" + ",".join(x.split(":", 1)[1] for x in sh[:4]) + (f" (+{len(sh) - 4})" if len(sh) > 4 else "")]
            break
    fids = []
    for lv, pat in fs:
        if lv == "NAME" and len(u.split("/")[-1]) < 6:
            continue
        c = re.compile(pat)
        fids = [k for k, t in FMAP.items() if c.search(t)]
        if fids:
            fids = [lv + ":" + ",".join(fids)]
            break
    srcs = ",".join(s for s in ("LAB", "DRIVE", "ASIX", "PCOOD", "LAPTOP_IGN") if r["n"][s])
    rows.append([u, r["kind"], srcs, r["n"]["LAB"], r["n"]["DRIVE"], r["drive_not_on_lab"], r["n"]["ASIX"], r["n"]["PCOOD"],
                 r["n"]["LAPTOP_IGN"], max(r["bytes"].values()), r["mmin"], r["mmax"], r["lab_on_drive"],
                 r["lab_not_on_drive"], level, ";".join(h.split(":", 1)[1] for h in hits[:5]) + (f" (+{len(hits) - 5})" if len(hits) > 5 else ""),
                 len(hits), ";".join(sec_hits), ";".join(fids)])

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "units.tsv"), "w", encoding="utf-8", newline="\n") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["unit", "kind", "sources", "n_lab", "n_drive", "n_drive_not_on_lab", "n_asix", "n_pcood", "n_laptop",
                "bytes_max_source", "mtime_min", "mtime_max", "lab_on_drive", "lab_not_on_drive", "mention_level",
                "mention_docs", "n_docs", "secondary_mentions", "fmap_links"])
    w.writerows(rows)
print("units:", len(rows), "docs:", len(docs), "secondary:", len(sec))
print(collections.Counter(r[14] for r in rows))
