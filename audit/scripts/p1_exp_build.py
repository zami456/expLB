"""Phase 1 Task A: build audit/experiments.csv from the hand-written definitions in p1_exp_defs.py plus
values COMPUTED here from raw census files, the lab log index and git.

Read-only on everything except the two outputs:
  audit/experiments.csv
  audit/phase1/exp_work/experiments_computed.tsv   (per-experiment computed numbers + doc-mention hits)

Inputs (all built earlier in the audit, md5-verified):
  audit/phase1/labpc/joined_labpc.tsv.gz   (streamed; one row per lab-PC file)
  audit/phase1/census_asix_git.tsv, census_pcood.tsv, census_laptop_ignored.tsv
  audit/phase1/labpc/drive_census.tsv.gz   (streamed)
  audit/phase1/labpc/log_index.tsv          (NUL bytes stripped)
  git log (read-only) for script commit dates
  docs for mention counts: tracked project docs on asix, the auto-memory dir, draft_text.txt, his summary.md files

Usage (repo root):  PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 python audit/scripts/p1_exp_build.py
"""
import csv, gzip, io, os, re, subprocess, sys, glob

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1_exp_defs as D  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
AUD = os.path.join(REPO, "audit")
P1 = os.path.join(AUD, "phase1")
BS = chr(92)
HIS = "origin/feat/persona-pc-ood"
MEM = "C:/Users/SURAIYA/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB/memory"

EXPS = D.EXPS
ROOTKEY = {
    "C:/Users/user4/expLB": "expLB", "C:/Users/user4": "user4", "C:/Users/user4/expLBactivationLAYER22": "LAYER22",
    "C:/Users/user4/share_L20_shared": "share_L20_shared", "C:/Users/user4/share_deceptive": "share_deceptive",
    "D:/expLB-acts": "D_acts", "C:/Users/user4/tools": "tools",
}


def lit_prefix(rx):
    """literal prefix of an anchored regex (for a cheap startswith pre-filter)."""
    if not rx.startswith("^"):
        return ""
    out = []
    i = 1
    while i < len(rx):
        c = rx[i]
        if c == BS and i + 1 < len(rx) and rx[i + 1] in "./-_":
            out.append(rx[i + 1]); i += 2; continue
        if c in ".^$*+?{}[]()|" + BS:
            break
        out.append(c); i += 1
    return "".join(out)


def comp(key):
    res = {}
    for e in EXPS:
        pats = e.get(key) or []
        if pats:
            res[e["id"]] = [(lit_prefix(p), re.compile(p)) for p in pats]
    return res


def matcher(pats, s):
    for pre, rx in pats:
        if pre and not s.startswith(pre):
            continue
        if rx.search(s):
            return True
    return False


def new_acc():
    return {"n": 0, "bytes": 0, "min": "9999", "max": "0000", "drive": 0, "asix": 0, "asixdiff": 0, "pcood": 0, "wave": 0, "wave_git": 0}


WAVE_LO, WAVE_HI = "2026-09-22 15:40", "2026-09-22 16:31"  # RF-13 re-run wave window (lab local time)


# ---------------------------------------------------------------- lab census
lab_p = comp("lab")
lab = {k: new_acc() for k in lab_p}
with gzip.open(os.path.join(P1, "labpc", "joined_labpc.tsv.gz"), "rt", encoding="utf-8", errors="replace") as f:
    hdr = f.readline().rstrip("\n").split("\t")
    ix = {h: i for i, h in enumerate(hdr)}
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < len(hdr):
            p += [""] * (len(hdr) - len(p))
        rk = ROOTKEY.get(p[ix["root"]].replace(BS, "/"), "other")
        s = rk + ":" + p[ix["relpath"]].replace(BS, "/")
        for eid, pats in lab_p.items():
            if matcher(pats, s):
                a = lab[eid]
                a["n"] += 1
                try:
                    a["bytes"] += int(p[ix["bytes"]] or 0)
                except ValueError:
                    pass
                mt = p[ix["mtime"]]
                if mt:
                    a["min"] = min(a["min"], mt); a["max"] = max(a["max"], mt)
                    if WAVE_LO <= mt[:16] < WAVE_HI:
                        a["wave"] += 1
                        a["wave_git"] += p[ix["in_asix"]] == "1"
                a["drive"] += p[ix["on_drive"]] == "1"
                a["asix"] += p[ix["in_asix"]] == "1"
                a["pcood"] += p[ix["in_pcood"]] == "1"
                a["asixdiff"] += bool(p[ix["asix_same_name_diff_content"]].strip())


def count_tsv(path, key, col, gz=False, skip_header=True):
    pats = comp(key)
    acc = {k: {"n": 0, "bytes": 0} for k in pats}
    if not pats:
        return acc
    op = gzip.open if gz else open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        if skip_header:
            f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) <= col:
                continue
            s = p[col]
            for eid, pp in pats.items():
                if matcher(pp, s):
                    acc[eid]["n"] += 1
                    try:
                        acc[eid]["bytes"] += int(p[col + 1])
                    except (ValueError, IndexError):
                        pass
    return acc


git = count_tsv(os.path.join(P1, "census_asix_git.tsv"), "git", 1)
his = count_tsv(os.path.join(P1, "census_pcood.tsv"), "pcood", 1)
drv = count_tsv(os.path.join(P1, "labpc", "drive_census.tsv.gz"), "drive_only", 0, gz=True)
lap = count_tsv(os.path.join(P1, "census_laptop_ignored.tsv"), "laptop", 0)

# ---------------------------------------------------------------- log index
logs = {}
raw = open(os.path.join(P1, "labpc", "log_index.tsv"), encoding="utf-8", errors="replace").read().replace("\x00", "")
for d in csv.DictReader(io.StringIO(raw), delimiter="\t"):
    name = d["log"].replace(" [utf16]", "").strip()
    logs[name] = d


def norm_ts(t):
    t = (t or "").strip()
    if not t:
        return ""
    t = t.replace("/", "-").replace("T", " ")
    if re.match(r"^\d\d-\d\d \d", t):
        t = "2026-" + t
    return t[:16]


# ---------------------------------------------------------------- git commits
_gc = {}


def git_hist(path, ref):
    key = (path, ref)
    if key in _gc:
        return _gc[key]
    try:
        out = subprocess.run(["git", "-C", REPO, "log", ref, "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M", "--", path],
                             capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
    except Exception:
        out = []
    r = (out[-1], out[0], len(out)) if out else None
    _gc[key] = r
    return r


def short(path):
    return path.replace("src/probes/deception-detection/", "DD/")


# ---------------------------------------------------------------- docs for mention counts
docs = {}
tracked = subprocess.run(["git", "-C", REPO, "ls-files", "kd/docs", "kd/README.md", "docs", "cap/docs"], capture_output=True, text=True).stdout.splitlines()
skip = re.compile(r"^docs/(concepts|developing|errors|gotchas|models|patterns|reference|remote|usage)/|NNSIGHT_UPSTREAM")
for p in tracked:
    if skip.search(p) or not (p.endswith(".md") or p.endswith(".csv")):
        continue
    try:
        docs[p] = open(os.path.join(REPO, p), encoding="utf-8", errors="replace").read().lower()
    except OSError:
        pass
for p in glob.glob(MEM + "/*.md"):
    docs["memory/" + os.path.basename(p)] = open(p, encoding="utf-8", errors="replace").read().lower()
docs["draft_text.txt"] = open(os.path.join(AUD, "phase0", "draft_text.txt"), encoding="utf-8", errors="replace").read().lower()
his_md = subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "--name-only", HIS], capture_output=True, text=True).stdout.splitlines()
for p in his_md:
    if p.startswith("src/probes/deception-detection/") and (p.endswith("summary.md") or p.endswith("deceptive/README.md")):
        txt = subprocess.run(["git", "-C", REPO, "show", HIS + ":" + p], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
        docs["his:" + short(p)] = txt.lower()


def doc_hits(keys):
    hits = []
    for name, txt in docs.items():
        if any(k.lower() in txt for k in keys):
            hits.append(name)
    return hits


# ---------------------------------------------------------------- build rows
COLS = ["exp_id", "what_it_tests", "who", "when_first", "when_last", "when_source", "branch", "commit", "script",
        "config_or_args", "scheduled_task", "results_location", "n_result_files", "status", "status_evidence",
        "depends_on", "f_ids", "notes"]
rows, comp_rows = [], []
for e in EXPS:
    eid = e["id"]
    ref = HIS if e.get("branch", "").startswith("origin/feat/persona-pc-ood") else "HEAD"
    # scripts + commits
    scr_txt, com_txt = [], []
    nrepo = 0
    main_first = None
    for s in e.get("scripts", []):
        if s.startswith("LAB:"):
            base = s[4:]
            full = ("C:/Users/user4/expLB/" + base[6:]) if base.startswith("expLB/") else ("C:/Users/user4/" + base[6:] if base.startswith("user4/") else base)
            scr_txt.append("LAB-ONLY " + full)
            continue
        scr_txt.append(short(s))
        h = git_hist(s, ref)
        if nrepo == 0 and h:
            main_first = (os.path.basename(s), h[0])
        if nrepo < 4:
            if h:
                com_txt.append(f"{os.path.basename(s)}: +{h[0]} / last {h[1]} ({h[2]} commits)")
            else:
                com_txt.append(f"{os.path.basename(s)}: NOT in git on {ref}")
        nrepo += 1
    if nrepo > 4:
        com_txt.append(f"(+{nrepo - 4} more scripts)")
    # when
    wf = wl = ws = ""
    w = e.get("when")
    if isinstance(w, tuple):
        wf, wl, ws = w
    elif isinstance(w, str) and w.startswith("GIT:"):
        h = git_hist("src/probes/deception-detection/results/" + w[4:], HIS)
        if h:
            wf, wl = h[0].split(" ", 1)[1], h[1].split(" ", 1)[1]
            ws = f"git commit dates of the folder on his branch (first {h[0].split()[0]}, last {h[1].split()[0]}, {h[2]} commits); run dates not recorded"
    else:
        fs = [norm_ts(logs[l]["first_ts"]) for l in e.get("logs", []) if l in logs and logs[l]["first_ts"]]
        ls = [norm_ts(logs[l]["last_ts"]) for l in e.get("logs", []) if l in logs and logs[l]["last_ts"]]
        if fs or ls:
            wf, wl = min(fs or ls), max(ls or fs)
            nlog = len([l for l in e.get("logs", []) if l in logs and (logs[l]["first_ts"] or logs[l]["last_ts"])])
            ws = f"internal timestamps of {nlog} lab log(s) (log_index.tsv)"
            if wl >= "2026-09-22 15:40" and wl <= "2026-09-22 16:30" and wf < "2026-09-22":
                ws += "; when_last = the 09-22 15:40-16:26 re-run wave (RF-13), original run ended earlier (see status_evidence)"
        elif eid in lab and lab[eid]["n"]:
            wf, wl, ws = lab[eid]["min"][:16], lab[eid]["max"][:16], "lab file mtimes (no log with timestamps)"
    if e.get("laptop_logs"):
        ws += ("; " if ws else "") + "laptop logs: " + e["laptop_logs"]
    # results / counts
    parts = []
    nres = ""
    if eid in lab:
        a = lab[eid]
        parts.append(f"lab {a['n']} files {a['bytes'] / 1e6:.1f} MB, mtimes {a['min'][:16]}..{a['max'][:16]}; identical copy on Drive {a['drive']}/{a['n']}; identical in asix git {a['asix']}; same-name-different-content in asix {a['asixdiff']}; mtime inside the 09-22 15:40-16:30 wave {a['wave']}")
        nres = a["n"]
    if eid in git:
        parts.append(f"asix git {git[eid]['n']} files")
        nres = nres if nres != "" else git[eid]["n"]
    if eid in his:
        parts.append(f"his branch {his[eid]['n']} files")
        nres = nres if nres != "" else his[eid]["n"]
    if eid in drv:
        parts.append(f"Drive paths matching {drv[eid]['n']}")
        nres = nres if nres != "" else drv[eid]["n"]
    if eid in lap:
        parts.append(f"laptop ignored files {lap[eid]['n']}")
        nres = nres if nres != "" else lap[eid]["n"]
    if nres == "":
        nres = 0
    loc = e.get("loc", "") + (" [computed: " + " | ".join(parts) + "]" if parts else "")
    hits = doc_hits(e.get("doc_keys", [])) if e.get("doc_keys") else []
    notes = e.get("notes", "")
    if main_first and re.match(r"^\d{4}-\d\d-\d\d \d\d:\d\d", wf or "") and not eid.startswith(("C", "BG")):
        cdate = main_first[1].split(" ", 1)[1]
        if cdate > wf:
            notes += f" [main script {main_first[0]} first entered git {main_first[1]}, AFTER when_first {wf}: the run-time version was not in git when it ran]"
    if eid in lab and lab[eid]["wave"]:
        notes += f" [RF-13: {lab[eid]['wave']} of {lab[eid]['n']} lab files have mtimes in the 09-22 15:40-16:30 re-run wave; {lab[eid]['wave_git']} of those are byte-identical (md5_lf) to some asix git blob]"
    if e.get("doc_keys"):
        notes += f" [doc mentions (keys {e['doc_keys']}): {len(hits)} doc(s)" + (": " + ", ".join(hits[:6]) + (" ..." if len(hits) > 6 else "") if hits else "") + "]"
    rows.append({
        "exp_id": eid, "what_it_tests": e["what"], "who": e["who"], "when_first": wf, "when_last": wl, "when_source": ws,
        "branch": e.get("branch", ""), "commit": "; ".join(com_txt), "script": "; ".join(scr_txt),
        "config_or_args": e.get("args", ""), "scheduled_task": e.get("task", ""), "results_location": loc,
        "n_result_files": nres, "status": e["status"], "status_evidence": e.get("evidence", ""),
        "depends_on": e.get("depends", ""), "f_ids": " ".join(e.get("f", [])), "notes": notes.strip(),
    })
    comp_rows.append([eid, nres, lab.get(eid, {}).get("n", ""), git.get(eid, {}).get("n", ""), his.get(eid, {}).get("n", ""),
                      drv.get(eid, {}).get("n", ""), lap.get(eid, {}).get("n", ""), lab.get(eid, {}).get("wave", ""),
                      lab.get(eid, {}).get("drive", ""), (main_first[1] if main_first else ""), len(hits), ",".join(hits), " ".join(e.get("f", []))])

with open(os.path.join(AUD, "experiments.csv"), "w", encoding="utf-8", newline="") as o:
    w = csv.DictWriter(o, fieldnames=COLS, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(P1, "exp_work", "experiments_computed.tsv"), "w", encoding="utf-8", newline="\n") as o:
    o.write("exp_id\tn_primary\tlab_n\tgit_n\this_n\tdrive_only_n\tlaptop_n\tlab_wave_n\tlab_on_drive_n\tmain_script_first_commit\tdoc_hits\tdoc_hit_names\tf_ids\n")
    for r in comp_rows:
        o.write("\t".join(str(x) for x in r) + "\n")
print(f"wrote {len(rows)} rows -> audit/experiments.csv; docs scanned: {len(docs)}")
