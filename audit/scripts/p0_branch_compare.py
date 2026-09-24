"""Phase 0 -- compare branch `asix` (owner) with `origin/feat/persona-pc-ood` (collaborator).

Read-only: uses only `git ls-tree`, `git show`, `git log`, `git diff`, `git rev-list`,
`git cat-file`, `git merge-base`, `git rev-parse`, `git for-each-ref`. Writes exactly one file,
audit/phase0/pcood_results_index.csv; everything else is printed to stdout.

Re-run from anywhere:
    PYTHONIOENCODING=utf-8 python audit/scripts/p0_branch_compare.py > some_scratch_file.txt

Sections printed (grep for '#### '):
  A  per-folder index of the collaborator's result/data folders (also written to CSV)
  B  LFS check (.gitattributes on every ref + small-blob pointer scan)
  C  his 17 .md docs: lines, dates, first heading, numeric-token counts, asix-related terms
  D  blob-hash dependencies between the branches
  E  shared paths under src/probes/deception-detection/ with different content
  F  folder-level: what each branch has that the other lacks
  G  submodule pins + .gitmodules per ref
  H  other remote branches
  I  rough counts for his branch
  J  keys/shapes of his .npz/.pt files; his 36-persona bank vs paper16_L20.pt at layer 20
     (torch.load(weights_only=True) on temp copies; nothing unpickled unsafely)
"""

from __future__ import annotations

import collections
import csv
import io
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_CSV = os.path.join(REPO, "audit", "phase0", "pcood_results_index.csv")

ASIX = "asix"
PC = "origin/feat/persona-pc-ood"
DD = "src/probes/deception-detection/"


def git(*args: str, binary: bool = False, check: bool = True, input_: bytes | None = None):
    res = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, input=input_, check=False
    )
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.decode('utf-8', 'replace')}")
    return res.stdout if binary else res.stdout.decode("utf-8", "replace")


def ls_tree(ref: str, path: str | None = None) -> list[tuple[str, str, str, int | None, str]]:
    """(mode, type, hash, size, path) for every entry, recursive, NUL-safe."""
    args = ["ls-tree", "-r", "-l", "-z", ref]
    if path:
        args += ["--", path]
    out = git(*args)
    rows = []
    for rec in out.split("\0"):
        if not rec:
            continue
        meta, p = rec.split("\t", 1)
        mode, typ, h, size = meta.split()
        rows.append((mode, typ, h, None if size == "-" else int(size), p))
    return rows


def log_lines(rng: str, path: str, extra: tuple[str, ...] = ()) -> list[tuple[str, str, str]]:
    out = git("log", *extra, "--format=%h|%ad|%s", "--date=iso", rng, "--", path)
    return [tuple(l.split("|", 2)) for l in out.splitlines() if l.strip()]  # type: ignore


def cat_many(hashes) -> dict[str, bytes]:
    """Contents of many blobs in one `git cat-file --batch` call."""
    hashes = list(dict.fromkeys(hashes))
    if not hashes:
        return {}
    raw = git("cat-file", "--batch", input_=("\n".join(hashes) + "\n").encode(), binary=True)
    out, i = {}, 0
    for h in hashes:
        nl = raw.index(b"\n", i)
        header = raw[i:nl].decode()
        parts = header.split()
        if len(parts) < 3 or parts[1] == "missing":
            i = nl + 1
            continue
        size = int(parts[2])
        out[h] = raw[nl + 1 : nl + 1 + size]
        i = nl + 1 + size + 1
    return out


LFS_MAGIC = b"version https://git-lfs"


def ext_of(p: str) -> str:
    b = p.rsplit("/", 1)[-1]
    return b.rsplit(".", 1)[-1].lower() if "." in b else "(none)"


MB = git("merge-base", ASIX, PC).strip()
print(f"asix={git('rev-parse', ASIX).strip()}  pc={git('rev-parse', PC).strip()}  merge-base={MB}")

# ----------------------------------------------------------------------------------------------
# A. folders the collaborator added
# ----------------------------------------------------------------------------------------------
changed = [l.split("\t") for l in git("diff", "--name-status", "--no-renames", MB, PC).splitlines()]
outside = [p for s, p in changed if not p.startswith(DD)]
print(f"\nchanged files MB..pc: {len(changed)}; outside {DD}: {len(outside)}")

CODE_TOP = {"deception_detection", "tests", "README.md"}


def folder_of(path: str) -> str | None:
    rel = path[len(DD):]
    parts = rel.split("/")
    if parts[0] == "results":
        return DD + "results/" + parts[1]
    if parts[0] in CODE_TOP:
        return None
    return DD + parts[0]


folders = sorted({f for s, p in changed if (f := folder_of(p))})

# How each folder was produced: read from the scripts, configs, summaries and commit messages
# (see the 'how_known' column). Filled in by hand after reading those sources during this audit.
S = "deception_detection/scripts/"
MANUAL = {
    "results/persona_ood": (
        S + "persona_ood.py run (defaults: --layer 20, --pooled_dir activations/pooled, out results/persona_ood); pooled by " + S + "pool_activations.py from the XXXX-9-llama-all cache; figures " + S + "persona_ood_figures.py",
        S + "configs/XXXX-9-llama-all.yaml (activations); bank: paper16_L20_sharedquestions/paper16_L20.pt at a093c54, regenerated via uncommitted 'persona vectors/vectors_s3' at 24a237b",
        "OUT_DIR default in persona_ood.py; persona_ood/summary.md 'Setup'; commit msgs a093c54, 24a237b",
        "6x6 train/test AUROC transfer matrix: 8192-d probe vs persona PC/axis projections (15 deception+honesty personas) + random-3d and sycophancy-basis controls; zero-shot axis AUROC; LB head-to-head",
    ),
    "results/persona_ood_L40": (
        S + "persona_ood.py run --layer 40 (pooled from XXXX-9-llama-all-L40 cache)",
        S + "configs/XXXX-9-llama-all-L40.yaml (committed later, 148ddeb); bank: uncommitted 'persona vectors/vectors_s3' (26 personas x 80 layers)",
        "persona_ood_L40/summary.md lines 1-10; commit 06da2de; persona_pcs.VECTORS_DIR",
        "same experiment as persona_ood at layer 40",
    ),
    "results/persona_ood_ext_L20": (
        S + "persona_ood.py run/zeroshot --layer 20 --vectors 'persona including knowlede denial'",
        S + "configs/XXXX-9-llama-all.yaml; bank: 'persona including knowlede denial/' (36 personas)",
        "persona_ood_ext_L20/summary.md; ext_L44 summary gives the analogous command; commit 9774075",
        "transfer matrix + zero-shot with extended bank (adds 4 knowledge-denial vs 6 candour personas): ext_axis1 (12v13), denial_axis1 (4v6), ext/denial PCs",
    ),
    "results/persona_ood_ext_L40": (
        S + "persona_ood.py run/zeroshot --layer 40 --vectors 'persona including knowlede denial' (pooled_L40)",
        S + "configs/XXXX-9-llama-all-L40.yaml; bank: 'persona including knowlede denial/'",
        "persona_ood_ext_L40/summary.md (points to ext_L20); commit 9774075",
        "extended-bank experiment at layer 40",
    ),
    "results/persona_ood_ext_L44": (
        S + "pool_activations.py --out_dir activations/pooled_L44; persona_ood.py verify/run/zeroshot --layer 44 --vectors 'persona including knowlede denial'",
        S + "configs/XXXX-9-llama-all-L44.yaml; bank: 'persona including knowlede denial/'",
        "reproduce block in persona_ood_ext_L44/summary.md lines 92-100; commit 29cc17c",
        "extended-bank experiment at layer 44 (third depth point)",
    ),
    "results/persona_ood_followup_L20": (
        S + "pool_activations.py --out_dir activations/pooled_followup_L20; persona_ood.py verify/run/zeroshot --layer 20 --vectors 'persona including knowlede denial'",
        S + "configs/XXXX-9-llama-all-followup.yaml (tail-mode activations at the prefilled 'No'); bank: 'persona including knowlede denial/'",
        "reproduce block in persona_ood_followup_L20/summary.md lines 141-149; commit d2f4cce",
        "extended-bank experiment on follow-up-token activations instead of mean over response tokens",
    ),
    "results/persona_ood_mod_L20": (
        S + "persona_ood.py run/zeroshot --layer 20 --vectors 'persona knowledge denial modified'",
        S + "configs/XXXX-9-llama-all.yaml; bank: UNCOMMITTED 'persona knowledge denial modified/'",
        "reproduce block in persona_ood_mod_L40/summary.md lines 100-108; commit 94db651",
        "same 36 personas re-extracted to ~uniform 142-150 rollouts: does better sampling improve the axes?",
    ),
    "results/persona_ood_mod_L40": (
        S + "persona_ood.py run/zeroshot --layer 40 --vectors 'persona knowledge denial modified'",
        S + "configs/XXXX-9-llama-all-L40.yaml; bank: UNCOMMITTED 'persona knowledge denial modified/'",
        "persona_ood_mod_L40/summary.md lines 100-108; commit 94db651",
        "modified (more-rollout) bank at layer 40",
    ),
    "results/persona_ood_mod_L44": (
        S + "persona_ood.py run/zeroshot --layer 44 --vectors 'persona knowledge denial modified'",
        S + "configs/XXXX-9-llama-all-L44.yaml; bank: UNCOMMITTED 'persona knowledge denial modified/'",
        "persona_ood_mod_L40/summary.md lines 100-108 ('per layer'); commit 94db651",
        "modified (more-rollout) bank at layer 44",
    ),
    "results/persona_positions_L20": (
        S + "persona_positions.py project/verify/analyze/figures --layer 20 --vectors 'persona including knowlede denial'",
        S + "configs/XXXX-9-llama-all.yaml; bank: 'persona including knowlede denial/'; verify reads persona_ood_ext_L20/zero_shot.csv",
        "persona_positions.py docstring; persona_positions_L40/summary.md lines 103-110; commit a8eb902",
        "persona-axis score at every token: mean vs max / top-10% / first / last / pre-reply reads, position profile",
    ),
    "results/persona_positions_L40": (
        S + "persona_positions.py project/verify/analyze/figures --layer 40",
        S + "configs/XXXX-9-llama-all-L40.yaml; bank: 'persona including knowlede denial/'",
        "persona_positions_L40/summary.md lines 103-110; commit a8eb902",
        "per-token persona reads at layer 40",
    ),
    "results/persona_positions_L44": (
        S + "persona_positions.py project/verify/analyze/figures --layer 44",
        S + "configs/XXXX-9-llama-all-L44.yaml; bank: 'persona including knowlede denial/'",
        "persona_positions_L40/summary.md (covers all three); commit a8eb902",
        "per-token persona reads at layer 44",
    ),
    "results/loo_L20": (
        S + "persona_ood.py loo --layer 20 --pooled_dir activations/pooled --out_dir results/loo_L20 --vectors 'persona including knowlede denial'; persona_ood_figures.py loo",
        S + "configs/XXXX-9-llama-all.yaml; bank: 'persona including knowlede denial/'",
        "reproduce block in loo_L20/summary.md lines 120-123; commit 6b16a46",
        "leave-one-dataset-out: 8192-d probe trained on pooled other 5 datasets (naive and 1/n weighting) vs persona arms; bootstrap CIs",
    ),
    "results/loo_L40": (
        S + "persona_ood.py loo --layer 40 (pooled_L40) --vectors 'persona including knowlede denial'",
        S + "configs/XXXX-9-llama-all-L40.yaml; bank: 'persona including knowlede denial/'",
        "loo_L40/summary.md (points to loo_L20); commit 6b16a46",
        "leave-one-dataset-out at layer 40",
    ),
    "results/XXXX-9-llama-all": (
        S + "experiment.py run (via " + S + "run_llama_all.sh)",
        S + "configs/XXXX-9-llama-all.yaml",
        "config 'name:' field = folder name; commit 148ddeb message ('plots from the five experiment runs')",
        "Liars' Bench Mean Probe (LR on RepE) at layer 20 over 5 LB datasets + Alpaca control; ONLY the 4 plots are committed",
    ),
    "results/XXXX-9-llama-all-L40": (
        S + "experiment.py run (run_llama_all.sh, CONFIG=...-L40.yaml, ACTS_CACHE_DIR=E:/...)",
        S + "configs/XXXX-9-llama-all-L40.yaml",
        "config 'name:' field = folder name; commit 148ddeb",
        "Mean Probe at layer 40; ONLY plots committed",
    ),
    "results/XXXX-9-llama-all-L44": (
        S + "experiment.py run (run_llama_all.sh, CONFIG=...-L44.yaml)",
        S + "configs/XXXX-9-llama-all-L44.yaml",
        "config 'name:' field = folder name; commit 148ddeb",
        "Mean Probe at layer 44; ONLY plots committed",
    ),
    "results/XXXX-9-llama-all-followup": (
        S + "experiment.py run (run_llama_all.sh, CONFIG=...-followup.yaml)",
        S + "configs/XXXX-9-llama-all-followup.yaml (copy committed in folder as cfg.yaml)",
        "summary.md 'Protocol' para; config header 'Run with'; commit fd17728",
        "Liars' Bench Follow-Up Probe (Table 7) measured at layer 20: probe reads the prefilled 'No' after 'Was your last response deceptive?'",
    ),
    "results/XXXX-9_lr__repe_honesty__plain__llama-70b-3.3__20260712_053340": (
        S + "experiment.py run (pre-branch pipeline)",
        S + "configs/XXXX-9.yaml (exists at merge-base 2e2a785; 2 eval datasets)",
        "folder timestamp 20260712_053340 + XXXX-9.yaml header in XXXX-9-llama-all.yaml ('the 2026-07-12 two-dataset run'); commit 148ddeb",
        "earlier Mean Probe run, layer 20, eval on harm_refusal_choice + instruct_dishonesty; ONLY plots committed",
    ),
    "deceptive": (
        "NOT-FOUND in git (export from owner's run 'run-paper16-shared'; no exporter script found on either branch)",
        "n/a (input data)",
        "deceptive/README.md 'Source'; deceptive_L20.json source_run",
        "INPUT: 8 deceptive persona vectors + deception axis at layer 20 from the owner's paper16 shared-question run",
    ),
    "paper16_L20_sharedquestions": (
        "NOT-FOUND in git (export from owner's run 'run-paper16-shared')",
        "n/a (input data)",
        "paper16_L20.json source_run='run-paper16-shared'; persona_pcs.BANK_PATH",
        "INPUT: 26-persona bank (vectors + 4 axes) at layer 20 from the owner's paper16 shared-question run",
    ),
    "persona including knowlede denial": (
        "NOT-FOUND in git (per-persona vector[80,8192] files; persona names match asix kd/personas/kd_instructions)",
        "n/a (input data)",
        "persona_pcs.EXTENDED_VECTORS_DIR docstring; commit 148ddeb message ('36 personas x 80 layers, 91 MB')",
        "INPUT: 36-persona all-layer bank = paper16's 26 re-extracted + 10 knowledge-denial/candour roles",
    ),
}


def layers_from_csvs(files: list[tuple[str, str, str, int | None, str]]) -> set[str]:
    found: set[str] = set()
    sel = [r for r in files if r[1] == "blob" and r[4].endswith(".csv") and (r[3] or 0) <= 2_000_000]
    contents = cat_many(r[2] for r in sel)
    for _, typ, h, size, p in sel:
        text = contents[h].decode("utf-8", "replace")
        try:
            rows = list(csv.DictReader(io.StringIO(text)))
        except Exception:
            continue
        for r in rows:
            v = r.get("layer")
            if v not in (None, ""):
                found.add(str(v))
    return found


def detect_layers(text: str) -> set[str]:
    """detect_layers in either flow style ([20]) or block style (- 20)."""
    m = re.search(r"detect_layers:\s*\[([^\]]*)\]", text)
    if m:
        return {x.strip() for x in m.group(1).split(",") if x.strip()}
    m = re.search(r"detect_layers:\s*\n((?:\s*-\s*\d+\s*\n)+)", text)
    return set(re.findall(r"\d+", m.group(1))) if m else set()


def layers_from_yaml(files) -> set[str]:
    found = set()
    for _, typ, h, size, p in files:
        if p.endswith(".yaml"):
            found |= detect_layers(git("cat-file", "-p", h))
    return found


print("\n#### A. collaborator folders")
csv_rows = []
for f in folders:
    files = ls_tree(PC, f + "/")
    blobs = [r for r in files if r[1] == "blob"]
    exts = collections.Counter(ext_of(r[4]) for r in blobs)
    nbytes = sum(r[3] or 0 for r in blobs)
    commits = log_lines(f"{MB}..{PC}", f + "/", ("--reverse",))
    adds = log_lines(f"{MB}..{PC}", f + "/", ("--reverse", "--diff-filter=A"))
    first = commits[0] if commits else ("", "", "")
    later = [c for c in commits[1:]]
    # which later commits only ADD files vs modify existing ones
    later_desc = []
    for h, d, s in later:
        st = git("diff", "--name-status", "--no-renames", f"{h}^", h, "--", f + "/")
        kinds = collections.Counter(l.split("\t")[0] for l in st.splitlines() if l.strip())
        later_desc.append(f"{h} {d[:10]} [{' '.join(f'{k}{v}' for k, v in sorted(kinds.items()))}] {s}")
    at_mb = ls_tree(MB, f + "/")
    small = cat_many(r[2] for r in blobs if (r[3] or 0) < 1024)
    lfs = sum(1 for v in small.values() if v.startswith(LFS_MAGIC))
    lay_name = ",".join(re.findall(r"[-_]L(\d+)$", f)) or ""
    lay_csv = sorted(layers_from_csvs(blobs), key=lambda x: int(x) if x.isdigit() else 999)
    lay_yaml = sorted(layers_from_yaml(blobs))
    key0 = f[len(DD):]
    m_cfg = re.search(r"configs/(\S+?\.yaml)", MANUAL.get(key0, ("", "", "", ""))[1])
    lay_cfg = ""
    if m_cfg:
        cfg_txt = git("show", f"{PC}:{DD}deception_detection/scripts/configs/{m_cfg.group(1)}")
        lay_cfg = f"{','.join(sorted(detect_layers(cfg_txt))) or '?'} ({m_cfg.group(1)})"
    key = f[len(DD):]
    script, config, how, what = MANUAL.get(key, ("?", "?", "?", "?"))
    row = {
        "folder": f,
        "n_files": len(blobs),
        "files_by_type": " ".join(f"{k}:{v}" for k, v in sorted(exts.items())),
        "total_blob_bytes": nbytes,
        "existed_at_merge_base": "yes" if at_mb else "no",
        "first_commit": first[0],
        "first_commit_date": first[1],
        "first_commit_msg": first[2],
        "n_commits_touching": len(commits),
        "later_commits": " || ".join(later_desc),
        "script": script,
        "config_or_input": config,
        "how_known": how,
        "what_it_tests": what,
        "layer_from_name": lay_name,
        "layer_in_csv_layer_column": ",".join(lay_csv),
        "layer_in_yaml_in_folder": ",".join(lay_yaml),
        "layer_in_activation_config": lay_cfg,
        "lfs_pointer_files": lfs,
        "in_asix_head_same_blob": 0,  # filled in section D
    }
    csv_rows.append(row)
    print(f"\n{f}\n  files={len(blobs)} bytes={nbytes} types={row['files_by_type']} at_MB={row['existed_at_merge_base']}")
    print(f"  first={first}\n  later={later_desc}\n  layers name={lay_name} csv={lay_csv} yaml_in_folder={lay_yaml} config={lay_cfg} lfs={lfs}")

# ----------------------------------------------------------------------------------------------
# B. LFS
# ----------------------------------------------------------------------------------------------
print("\n#### B. LFS")
refs_lfs = [ASIX, PC, MB, "origin/main"]
for ref in refs_lfs:
    tree = ls_tree(ref)
    ga = [r[4] for r in tree if r[4].endswith(".gitattributes")]
    lfs_lines = []
    for p in ga:
        txt = git("show", f"{ref}:{p}")
        lfs_lines += [f"{p}: {l}" for l in txt.splitlines() if "lfs" in l.lower()]
    small = [r for r in tree if r[1] == "blob" and (r[3] or 0) < 1024]
    contents = cat_many(r[2] for r in small)
    ptr = [r[4] for r in small if contents.get(r[2], b"").startswith(LFS_MAGIC)]
    print(f"{ref}: .gitattributes={ga} lfs-lines={lfs_lines} blobs<1KiB scanned={len(small)} lfs-pointers={ptr}")

# ----------------------------------------------------------------------------------------------
# C. his .md docs
# ----------------------------------------------------------------------------------------------
print("\n#### C. collaborator .md docs")
NUM = re.compile(r"(?<![\w.])[-+\u2212]?\d*\.\d+(?![\w.])|\b\d+(?:\.\d+)?\s?%")
TERMS = {
    "paper16": r"paper16",
    "knowledge-denial/kd": r"knowledge[- ]denial|knowlede denial|\bkd\b",
    "F-number": r"\bF-\d+",
    "cap/capitulation": r"\bcap\b|capitulation",
    "shared-questions/run-paper16-shared": r"shared[_ -]?questions?|run-paper16-shared|sharedquestions",
    "vectors_s3/'persona vectors'": r"vectors_s3|persona vectors",
    "asix/owner": r"\basix\b|Suraiya|\bowner\b",
    "run-* dirs": r"run-(kd|cap|coords|crossgen|paper16)",
    "FINDINGS.md": r"FINDINGS",
    "Assistant Axis": r"assistant[_ ]axis",
    "arXiv 2605.09391": r"2605\.09391",
}
mds = sorted(p for s, p in changed if p.endswith(".md"))
doc_rows = []
for p in mds:
    txt = git("show", f"{PC}:{p}")
    lines = txt.splitlines()
    commits = log_lines(f"{MB}..{PC}", p, ("--reverse",))
    heading = next((l.strip() for l in lines if l.startswith("#")), lines[0].strip() if lines else "")
    nums = NUM.findall(txt)
    num_lines = sum(1 for l in lines if NUM.search(l))
    hits = {k: len(re.findall(v, txt, flags=re.I)) for k, v in TERMS.items()}
    hits = {k: v for k, v in hits.items() if v}
    if p == DD + "README.md":
        # only the part pc-ood added
        diff = git("diff", MB, PC, "--", p)
        added = "\n".join(l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
        nums = NUM.findall(added)
        num_lines = sum(1 for l in added.splitlines() if NUM.search(l))
        heading = "(README.md: added section) " + next((l for l in added.splitlines() if l.startswith("#")), "")
        lines_added = len(added.splitlines())
    else:
        lines_added = None
    doc_rows.append((p, len(lines), lines_added, commits, heading, len(nums), num_lines, hits))
    print(f"\n{p}\n  lines={len(lines)} added_lines={lines_added} commits={[(c[0], c[1][:16]) for c in commits]}")
    print(f"  heading={heading!r}\n  numeric tokens={len(nums)} lines-with-number={num_lines}\n  terms={hits}")
print(f"\nTOTAL numeric tokens over {len(doc_rows)} docs = {sum(r[5] for r in doc_rows)}; lines with a number = {sum(r[6] for r in doc_rows)}")

# ----------------------------------------------------------------------------------------------
# D. blob-hash dependencies
# ----------------------------------------------------------------------------------------------
print("\n#### D. blob-hash dependencies")
asix_tree = ls_tree(ASIX)
asix_by_hash: dict[str, list[str]] = collections.defaultdict(list)
for m, t, h, s, p in asix_tree:
    if t == "blob":
        asix_by_hash[h].append(p)
pc_tree = ls_tree(PC)
pc_by_path = {p: (h, s) for m, t, h, s, p in pc_tree if t == "blob"}
pc_by_hash: dict[str, list[str]] = collections.defaultdict(list)
for m, t, h, s, p in pc_tree:
    if t == "blob":
        pc_by_hash[h].append(p)


def objs(rng: str) -> set[str]:
    return {l.split()[0] for l in git("rev-list", "--objects", rng).splitlines() if l.strip()}


asix_new = objs(f"{MB}..{ASIX}")
pc_new = objs(f"{MB}..{PC}")
asix_all = objs(ASIX)
pc_changed_paths = [p for s, p in changed if s != "D"]
hits_head = []
hits_hist = []
for p in pc_changed_paths:
    h, size = pc_by_path[p]
    if h in asix_by_hash:
        hits_head.append((p, h, size, asix_by_hash[h]))
    elif h in asix_all:
        hits_hist.append((p, h, size))
print(f"pc-ood changed blobs: {len(pc_changed_paths)}")
print(f"  ... identical blob somewhere in asix HEAD tree: {len(hits_head)}")
for x in hits_head:
    print("     ", x)
print(f"  ... identical blob only in asix history (not HEAD): {len(hits_hist)}")
for x in hits_hist:
    print("     ", x)
for row in csv_rows:
    row["in_asix_head_same_blob"] = sum(1 for x in hits_head if x[0].startswith(row["folder"] + "/"))

both_new = asix_new & pc_new
if both_new:
    chk = git("cat-file", "--batch-check", input_="\n".join(sorted(both_new)).encode(), binary=True).decode()
    both_blobs = [l.split()[0] for l in chk.splitlines() if l.split()[1] == "blob"]
else:
    both_blobs = []
print(f"objects introduced on BOTH sides since MB: {len(both_new)} (blobs: {len(both_blobs)})")
for h in both_blobs:
    print("   ", h, "asix:", asix_by_hash.get(h, ["(history only)"])[:3], "pc:", pc_by_hash.get(h, ["(history only)"])[:3])

# same basename, non-generic, compare
GENERIC = {"summary.md", "summary.csv", "zero_shot.csv", "README.md", "cfg.yaml", "__init__.py"}
asix_by_base: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
for m, t, h, s, p in asix_tree:
    if t == "blob":
        asix_by_base[p.rsplit("/", 1)[-1]].append((p, h))
print("\nsame basename on asix HEAD (pc-ood changed files):")
seen = set()
for p in pc_changed_paths:
    b = p.rsplit("/", 1)[-1]
    if b in asix_by_base and b not in seen:
        seen.add(b)
        same = [(ap, ah == pc_by_path[p][0]) for ap, ah in asix_by_base[b]]
        tag = "GENERIC" if b in GENERIC else ""
        print(f"  {b} {tag}: pc={p} -> asix {same[:6]}{' ...' if len(same) > 6 else ''}")

# asix docs cite pc-ood at f386a72, two commits before its tip: which cited-folder files changed since?
CITED = ["results/persona_ood_ext_L20", "results/persona_ood_ext_L40"]
print("\nfiles changed f386a72..pc-tip inside the folders asix's docs cite (persona_ood_ext_L20/L40):")
print(git("diff", "--name-status", "f386a72", PC, "--", *[DD + c for c in CITED]).strip() or "  (none)")
print("asix commits whose diff mentions 'persona-pc-ood':",
      git("log", "--format=%h %ad %s", "--date=iso", "-S", "persona-pc-ood", ASIX, "--",
          "kd/docs/FINDINGS.md", "docs/COLLABORATOR_BRIEFING.md", "docs/ROADMAP_TO_DEFENCE.md",
          "kd/personas/kd_role_groups_collab.json").strip())
print("first asix commit containing 'amnesiac_witness' in kd/personas:",
      git("log", "--reverse", "--format=%h %ad %s", "--date=iso", ASIX, "--",
          "kd/personas/kd_instructions/pc_amnesiac_witness.json").splitlines()[0])
print("asix commit renaming pc_candid_unknowing -> pc_honest_novice:",
      git("log", "--format=%h %ad %s", "--date=iso", "--diff-filter=A", ASIX, "--",
          "kd/personas/kd_instructions/pc_honest_novice.json").strip())
print("asix commit adding run_paper16_shared_chain.ps1 (produces run-paper16-shared/):",
      git("log", "--reverse", "--format=%h %ad %s", "--date=iso", ASIX, "--",
          "src/assistant_axis_ndif/run_paper16_shared_chain.ps1").splitlines()[0])
for label, p in [("ext bank uses pc_honest_novice", DD + "persona including knowlede denial/pc_honest_novice.pt"),
                 ("ext bank has pc_candid_unknowing", DD + "persona including knowlede denial/pc_candid_unknowing.pt")]:
    print(f"{label}: {p in pc_by_path}")

# reverse: which asix-new blobs are anywhere in pc tip tree
rev = [h for h in asix_new if h in pc_by_hash]
print(f"\nasix-new objects (MB..asix) present in pc-ood tip tree: {len(rev)}")

# ----------------------------------------------------------------------------------------------
# E. shared paths under DD
# ----------------------------------------------------------------------------------------------
print("\n#### E. shared paths under", DD)
a = {p: h for m, t, h, s, p in ls_tree(ASIX, DD) if t == "blob"}
c = {p: h for m, t, h, s, p in ls_tree(PC, DD) if t == "blob"}
b = {p: h for m, t, h, s, p in ls_tree(MB, DD) if t == "blob"}
print(f"files under DD: asix={len(a)} pc={len(c)} MB={len(b)}; asix==MB: {a == b}")
for p in sorted(set(a) & set(c)):
    if a[p] != c[p]:
        who = []
        if a[p] != b.get(p):
            who.append("asix changed")
        if c[p] != b.get(p):
            who.append("pc changed")
        ns = git("diff", "--numstat", MB, PC, "--", p).strip()
        print(f"  DIFF {p}: {', '.join(who)}; MB..pc numstat: {ns}")
print(f"  only on pc under DD: {len(set(c) - set(a))}; only on asix under DD: {len(set(a) - set(c))}")
print("  asix commits touching DD since MB:", git("log", "--oneline", f"{MB}..{ASIX}", "--", DD).strip() or "(none)")

# ----------------------------------------------------------------------------------------------
# F. folder-level differences
# ----------------------------------------------------------------------------------------------
print("\n#### F. folder-level differences (blob counts; gitlinks counted separately)")


def depth_of(parts: list[str]) -> int:
    if parts[:2] == ["src", "probes"]:
        return 4  # src/probes/deception-detection/<x>
    return {"src": 2, "docs": 2, "results": 2}.get(parts[0], 1)


def dir_counts(ref: str) -> collections.Counter:
    cnt: collections.Counter = collections.Counter()
    for m, t, h, s, p in ls_tree(ref):
        parts = p.split("/")
        d = depth_of(parts)
        if t == "commit":
            key = "[gitlink] " + p
        elif len(parts) == 1:
            key = "(root file) " + p
        elif len(parts) > d:
            key = "/".join(parts[:d])
        else:
            key = "/".join(parts[:-1])
        cnt[key] += 1
    return cnt


ca, cp, cm = dir_counts(ASIX), dir_counts(PC), dir_counts(MB)
keys = sorted(set(ca) | set(cp) | set(cm))
print(f"{'dir':70s} {'MB':>6s} {'asix':>6s} {'pc':>6s}")
for k in keys:
    if k.startswith("(root file)"):
        continue
    if ca[k] != cp[k] or ca[k] != cm[k]:
        print(f"{k[:70]:70s} {cm[k]:6d} {ca[k]:6d} {cp[k]:6d}")
root_a = sorted(k for k in ca if k.startswith("(root file)"))
root_p = sorted(k for k in cp if k.startswith("(root file)"))
print("root files only on asix:", [k[12:] for k in root_a if k not in cp])
print("root files only on pc:", [k[12:] for k in root_p if k not in ca])
print(f"TOTAL entries: MB={sum(cm.values())} asix={sum(ca.values())} pc={sum(cp.values())}")

# ----------------------------------------------------------------------------------------------
# G. submodules
# ----------------------------------------------------------------------------------------------
print("\n#### G. submodule pins")
for ref in [ASIX, PC, "origin/main", "main", MB]:
    pins = [(r[4], r[2]) for r in ls_tree(ref) if r[0] == "160000"]
    gm = git("show", f"{ref}:.gitmodules", check=False)
    print(f"\n{ref}: gitlinks={pins}")
    print("  .gitmodules:", " | ".join(l.strip() for l in gm.splitlines() if l.strip()))

# ----------------------------------------------------------------------------------------------
# H. other branches
# ----------------------------------------------------------------------------------------------
print("\n#### H. branches")
refs = [
    r
    for r in git("for-each-ref", "--format=%(refname:short)", "refs/remotes", "refs/heads").split()
    if not r.endswith("/HEAD") and r != "origin"
]
for r in refs:
    tip = git("log", "-1", "--format=%h|%an <%ae>|%ad|%s", "--date=iso", r).strip()
    mba = git("merge-base", ASIX, r, check=False).strip()
    ahead_a, behind_a = git("rev-list", "--left-right", "--count", f"{r}...{ASIX}").split()
    ahead_m, behind_m = git("rev-list", "--left-right", "--count", f"{r}...{MB}").split()
    authors = collections.Counter(git("log", "--format=%an <%ae>", f"{ASIX}..{r}").splitlines())
    touched = collections.Counter()
    if mba:
        for p in git("diff", "--name-only", mba, r).splitlines():
            parts = p.split("/")
            touched["/".join(parts[:2]) if parts[0] in ("src", "docs") and len(parts) > 2 else parts[0]] += 1
    print(f"\n{r}\n  tip={tip}\n  vs asix: {ahead_a} ahead / {behind_a} behind (merge-base {mba[:7]})"
          f"\n  vs MB {MB[:7]}: {ahead_m} ahead / {behind_m} behind\n  authors of commits not in asix: {dict(authors)}"
          f"\n  files changed since merge-base with asix, by dir: {dict(touched.most_common(12))}")

# ----------------------------------------------------------------------------------------------
# write CSV
# ----------------------------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
    w.writeheader()
    w.writerows(csv_rows)
# ----------------------------------------------------------------------------------------------
# J. contents of his array files (keys/shapes only; torch.load with weights_only=True)
# ----------------------------------------------------------------------------------------------
print("\n#### J. array files: keys and shapes")
import tempfile  # noqa: E402

try:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
except ImportError as e:  # pragma: no cover
    np = torch = None
    print("numpy/torch unavailable:", e)

if np is not None:
    arr_paths = [p for s, p in changed if p.endswith((".npz", ".pt"))]
    contents = cat_many(pc_by_path[p][0] for p in arr_paths)
    ext_bank_n = {}
    ext_l20 = {}
    with tempfile.TemporaryDirectory() as td:
        for p in sorted(arr_paths):
            fn = os.path.join(td, "x" + os.path.splitext(p)[1])
            with open(fn, "wb") as fh:
                fh.write(contents[pc_by_path[p][0]])
            try:
                if p.endswith(".npz"):
                    with np.load(fn, allow_pickle=False) as z:
                        desc = {k: (tuple(z[k].shape), str(z[k].dtype)) for k in z.files}
                else:
                    obj = torch.load(fn, weights_only=True, map_location="cpu")
                    desc = {
                        k: (tuple(v.shape), str(v.dtype)) if hasattr(v, "shape") else (type(v).__name__, (v if isinstance(v, (int, float, str)) else len(v) if hasattr(v, "__len__") else ""))
                        for k, v in obj.items()
                    }
                    if "persona including knowlede denial" in p:
                        name = p.rsplit("/", 1)[-1][:-3]
                        ext_bank_n[name] = int(obj["n"])
                        ext_l20[name] = obj["vector"][20].double().numpy()
                    elif p.endswith("paper16_L20.pt"):
                        p16 = {n: obj["personas"][i].double().numpy() for i, n in enumerate(obj["persona_names"])}
                        p16_n = dict(zip(obj["persona_names"], [int(x) for x in obj["persona_n_rollouts"]]))
            except Exception as e:  # report, never unpickle unsafely
                desc = f"LOAD FAILED ({type(e).__name__}: {str(e)[:120]})"
            if "persona including knowlede denial" in p and not p.endswith("default.pt") and not p.endswith("pc_amnesiac_witness.pt"):
                continue  # print one example of the 36
            print(f"  {p[len(DD):]}: {desc}")
    if ext_bank_n:
        print(f"\n  ext bank: {len(ext_bank_n)} personas; n per persona: {dict(sorted(ext_bank_n.items()))}")
        try:
            common = sorted(set(p16) & set(ext_l20))
            diffs = {n: float(np.linalg.norm(ext_l20[n] - p16[n])) for n in common}
            same = [n for n, d in diffs.items() if d == 0.0]
            print(f"  layer-20 slice of ext bank vs paper16_L20.pt: {len(common)} shared names; bit-identical: {same};"
                  f" max ||diff|| = {max(diffs.values()):.4f}; n paper16 vs ext for shared: "
                  f"{ {n: (p16_n[n], ext_bank_n[n]) for n in common} }")
        except NameError:
            print("  (paper16_L20.pt not loaded; comparison skipped)")

print(f"\n#### I. counts\nfolders={len(csv_rows)} result folders={sum(1 for r in csv_rows if '/results/' in r['folder'])}"
      f" files in folders={sum(r['n_files'] for r in csv_rows)} bytes={sum(r['total_blob_bytes'] for r in csv_rows)}")
print("wrote", OUT_CSV)
