"""Shared helpers for the Phase 1 Task B (anomalies) scripts. Read-only; imports only the standard library.
- canon(source, path): map a path from any census source to one canonical, lab-style repo-relative path.
- unit(canon_path): the experiment/result unit a file belongs to (None = not a result of the persona work).
- load_docs(): the doc corpus named in audit/phase1/INVENTORY_INSTRUCTIONS.md (Task B), lower-cased,
  backslashes turned into '/', as {doc_label: text}.
"""
import os, re, subprocess

BS = chr(92)
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MEM = "C:/Users/SURAIYA/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB/memory"
PCOOD = "origin/feat/persona-pc-ood"

# Drive (tamara:expLB/) prefix -> lab-style prefix. Longest prefix wins.
DRIVE_MAP = [
    ("kd/crossgen11/", "run-crossgen/kd11/"), ("kd/run-kd/", "run-kd/"), ("kd/figures/crossgen11/", "figures/crossgen/kd11/"),
    ("own100/run/", "run/"), ("own100/", "run/"), ("paper16/run/", "run-paper16/"), ("paper16/logs/", "logs/"),
    ("paper16-shared/", "run-paper16-shared/"), ("paper16/", "run-paper16/"), ("coords/", "run-coords/"),
    ("crossgen/", "run-crossgen/"), ("scale/", "run-scale/"), ("sycophancy_mt/", "run-sycophancy-mt/"),
    ("sycophancy/", "run-sycophancy/"), ("hf275/", "run-hf275/"), ("figures-crossgen/", "figures/crossgen/"),
    ("figures/", "figures/persona-coords/"), ("thesis-docs/", "docs/thesis/"), ("share/deceptive/", "@share_deceptive/"),
    ("share/paper16_L20_sharedquestions/", "@share_L20_shared/"),
    ("kd/", "kd/"), ("staging/", "drive:staging/"), ("share/", "drive:share/"), ("_selftest/", "drive:_selftest/"),
]
DRIVE_MAP.sort(key=lambda kv: -len(kv[0]))


def canon(source, path, root=None):
    if source == "LAB":
        r = (root or "").replace(BS, "/")
        return path if r.endswith("/expLB") else "@" + r.split("/")[-1] + "/" + path
    if source == "DRIVE":
        for a, b in DRIVE_MAP:
            if path.startswith(a):
                return b + path[len(a):]
        return "drive:" + path
    return path


STRUCT = {"responses", "scores", "vectors", "vectors_all", "vectors_s3", "vectors_s23", "vectors_all_fp32",
          "vectors_s3_fp32", "vectors_fp32", "vectors_275", "vectors_100", "csv", "matrices", "chain"}
RES_EXT = (".json", ".jsonl", ".csv", ".pt", ".npz", ".npy", ".log", ".status", ".parquet", ".pkl", ".txt",
           ".yaml", ".yml", ".png", ".pdf", ".zip", ".predupe", ".stale-576", ".tmp", ".err", ".xlsx")
FAMILY_ROOTS = ("run-coords", "run-premise", "kd/results")
FILE_ROOTS = ("run-kd", "run-paper16", "run-paper16-shared", "run-hf275", "run", "run-cap", "run-cap-ctrl",
              "run-scale/3b", "run-scale/8b")


def family(stem):
    return re.sub(r"\d+", "#", stem)


def unit(p):
    """Return (unit, kind) or (None, reason)."""
    parts = p.split("/")
    name = parts[-1]
    low = name.lower()
    ext = os.path.splitext(low)[1]
    marker = name in (".launched", "DELIVERABLE_READY", ".done") or ext in (".done",)
    if not (low.endswith(RES_EXT) or marker or "." not in name):
        return None, "not-result-ext"
    d = parts[:-1]
    top = d[0] if d else ""
    if p.startswith(("results/", "data/", "figures/barcharts", "figures/heatmaps", "figures/violin_plots",
                     "figures/radarcharts", "figures/examples", "figures/figure_1", "figures/harm-pressure")):
        return None, "liars-bench-background"
    if top in ("assistant_axis", "mask", "pacchiardi", "soft-trigger", "tools", "@tools", "@user4", "thinking",
               "docs", "src", "scale", "kd", "cap") and not p.startswith(("kd/run/", "kd/results/", "cap/run",
               "src/probes/deception-detection/results/", "src/probes/deception-detection/deceptive/",
               "src/probes/deception-detection/paper16_L20_sharedquestions/",
               "src/probes/deception-detection/persona including knowlede denial/", "docs/thesis/")):
        return None, "code-or-input"
    if p.startswith("src/probes/deception-detection/"):
        rest = parts[3:]
        if rest[0] == "results":
            return "/".join(parts[:5]), "his-result"
        return "/".join(parts[:4]), "his-input"
    if p.startswith("docs/thesis/"):
        return None, "doc"
    if top == "@expLB-acts":
        return ("@expLB-acts/logs", "lab-logs") if len(d) > 1 and d[1] == "logs" else (None, "vendored-copy")
    if top.startswith("@"):
        return "/".join(d[:2]) if len(d) > 1 else top, "lab-other-root"
    if top == "logs":
        return "logs/" + family(os.path.splitext(name)[0]), "log"
    if top == "run-crossgen":
        keep = []
        for s in d:
            if s in ("csv", "matrices"):
                break
            keep.append(s)
        return "/".join(keep), "crossgen"
    if top == "run-coords":
        if len(d) == 1:
            return "run-coords/" + family(os.path.splitext(name)[0]), "coords-analysis"
        return "/".join(d[:2]), "coords-store"
    if top == "run-scale":
        if len(d) >= 3 and d[2] == "coords":
            return "/".join(d[:3]), "scale-coords"
        if len(d) == 2:
            return "/".join(d[:2]) + "/" + os.path.splitext(name)[0], "file"
        return "/".join(d[:2]), "scale"
    if top == "run-premise":
        if len(d) == 1:
            return "run-premise/" + family(os.path.splitext(name)[0]), "premise"
        return "/".join(d[:2]), "premise"
    if p.startswith("kd/results/"):
        return "kd/results/" + os.path.splitext(name)[0], "kd-results"
    if p.startswith("kd/run/"):
        return "/".join(parts[:3]), "kd-run"
    if top.startswith("cap"):
        if top == "cap" and len(d) >= 2:
            if d[1] == "run":
                return "/".join(d[:3]) if len(d) >= 3 else "cap/run", "cap-run"
            return "/".join(d[:2]), "cap-run"
        return top, "cap"
    if top in ("run-sycophancy", "run-sycophancy-mt", "data_syco", "smoke", "aa-smoke", "aa-smoke-win"):
        return top, "store"
    if top in FILE_ROOTS or "/".join(d[:2]) in FILE_ROOTS:
        root = "/".join(d[:2]) if "/".join(d[:2]) in FILE_ROOTS else top
        depth = root.count("/") + 1
        if len(d) == depth:
            return root + "/" + os.path.splitext(name)[0], "file"
        sub = d[depth]
        return (root if sub in STRUCT else root + "/" + sub), "store"
    if top == "figures":
        return "/".join(d[:3]) if len(d) >= 3 and d[1] == "crossgen" else "/".join(d[:2]), "figures"
    if p.startswith("drive:"):
        q = p[len("drive:"):].split("/")
        return "drive:" + "/".join(q[:2] if len(q) > 2 else q[:-1] or q), "drive-only-area"
    return top or "(root)", "other"


def _read(p):
    try:
        return open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


def git_show(ref, path):
    r = subprocess.run(["git", "-C", REPO, "show", f"{ref}:{path}"], capture_output=True)
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else ""


def git_ls(ref):
    r = subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "--name-only", ref], capture_output=True)
    return r.stdout.decode("utf-8", "replace").splitlines()


_BRACE = re.compile(r"[\w./@-]*\{[^{}\n]{1,160}\}[\w./@{},-]*")


def _expand_token(tok, depth=0):
    m = re.search(r"\{([^{}]*)\}", tok)
    if not m or depth > 3:
        return [tok]
    inner = m.group(1)
    alts = [inner.split("..")[0]] if ".." in inner else inner.split(",")
    out = []
    for a in alts[:60]:
        out.extend(_expand_token(tok[:m.start()] + a.strip() + tok[m.end():], depth + 1))
    return out[:200]


def norm(t):
    """lower-case, backslash -> '/', and append brace expansions (a{x,y}b -> axb ayb; {0..76} -> 0)."""
    t = t.lower().replace(BS, "/")
    extra = []
    for tok in set(_BRACE.findall(t)):
        extra.extend(_expand_token(tok))
    nl = chr(10)
    return t + (nl + nl.join(extra) if extra else "")


def load_docs(expand=True):
    nz = norm if expand else (lambda t: t.lower().replace(BS, "/"))
    docs = {}
    for p in git_ls("HEAD"):
        top_docs = p.startswith("docs/") and p.count("/") == 1 and p.endswith(".md")
        if top_docs or p.startswith(("kd/docs/", "cap/docs/", "docs/thesis/")):
            if p.endswith((".xlsx",)):
                continue
            docs["asix:" + p] = nz(_read(os.path.join(REPO, p)))
    for f in sorted(os.listdir(MEM)):
        if f.endswith(".md"):
            docs["memory:" + f] = nz(_read(os.path.join(MEM, f)))
    for p in git_ls(PCOOD):
        if p.endswith("/summary.md"):
            docs["pcood:" + p] = nz(git_show(PCOOD, p))
    docs["draft:draft_text.txt"] = nz(_read(os.path.join(REPO, "audit", "phase0", "draft_text.txt")))
    return docs


def load_secondary():
    """Prose outside the Task-B doc set, used only for a context column: CLAUDE.md, READMEs, the plan file
    (= lab/Drive thinking/plan.md), the collaborator abstract, and every tracked script on both branches."""
    sec = {}
    for p in ("CLAUDE.md", "kd/README.md", "README.md", "audit/phase0/collab_abstract_2026-09-21.md"):
        sec["prose:" + p] = norm(_read(os.path.join(REPO, p)))
    plan = os.path.expanduser("~/.claude/plans/harmonic-cuddling-kay.md")
    sec["prose:plan harmonic-cuddling-kay.md"] = norm(_read(plan))
    for ref, lab in (("HEAD", "asix"), (PCOOD, "pcood")):
        for p in git_ls(ref):
            if p.endswith((".py", ".ps1", ".sh", ".yaml")) and not p.startswith(("assistant_axis/", "mask/", "pacchiardi/",
                                                                                 "soft-trigger/")):
                if lab == "pcood" and not p.startswith("src/probes/deception-detection/"):
                    continue
                if lab == "asix" and p.startswith("src/probes/deception-detection/"):
                    continue
                sec[f"code:{lab}:{p}"] = None  # lazily read
    return sec
