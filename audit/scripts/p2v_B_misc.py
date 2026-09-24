"""Verify batch B: the file-level facts for C08-ext-L20, O72, O12, OR-04, OR-05 and F-22.

Everything here is recomputed from raw evidence available on the laptop:
  audit/phase1/labpc/labpc_census.tsv   (lab PC file census: path, size, mtime, md5)
  audit/phase1/census_pcood.tsv         (git blobs of origin/feat/persona-pc-ood)
  git show                              (committed file versions)
  audit/cache/labpc/...                 (md5-verified copies of lab result files)

Run: python audit/scripts/p2v_B_misc.py
"""
import csv, os, re, subprocess, sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CENSUS = os.path.join(ROOT, "audit", "phase1", "labpc", "labpc_census.tsv")
PCOOD = os.path.join(ROOT, "audit", "phase1", "census_pcood.tsv")
csv.field_size_limit(10 ** 7)
LAB = "C:" + chr(92) + "Users" + chr(92) + "user4" + chr(92) + "expLB"


def lab_rows():
    with open(CENSUS, encoding="utf-8", newline="") as f:
        for d in csv.DictReader(f, delimiter="\t"):
            if d["root"] == LAB:
                yield d


LABALL = list(lab_rows())
print("lab census rows under %s : %d" % (LAB, len(LABALL)))

# ---------------------------------------------------------------- C08-ext-L20
print("\n=== C08-ext-L20 (collaborator branch) ===")
pc = list(csv.DictReader(open(PCOOD, encoding="utf-8"), delimiter="\t"))
res = [r for r in pc if "results/persona_ood_ext_L20/" in r["path"]]
print("files in results/persona_ood_ext_L20 :", len(res))
print("  csv=%d png=%d md=%d" % (sum(r["path"].endswith(".csv") for r in res),
                                 sum(r["path"].endswith(".png") for r in res),
                                 sum(r["path"].endswith(".md") for r in res)))
his = {r["path"].split("/")[-1]: r["md5"] for r in pc
       if "persona including knowlede denial/" in r["path"] and r["path"].endswith(".pt")}
lab = {d["relpath"].split("/")[-1]: d["md5"] for d in LABALL
       if d["relpath"].startswith("run-kd/vectors_s3/") and d["relpath"].endswith(".pt")}
print("  his bank .pt = %d ; lab run-kd/vectors_s3 .pt = %d ; md5-identical = %d"
      % (len(his), len(lab), sum(1 for k, v in his.items() if lab.get(k) == v)))
src = subprocess.run(["git", "show",
                      "origin/feat/persona-pc-ood:src/probes/deception-detection/"
                      "deception_detection/persona_pcs.py"],
                     cwd=ROOT, capture_output=True).stdout.decode("utf-8", "replace")
for g in ("DECEPTION_PERSONAS", "HONESTY_PERSONAS", "KNOWLEDGE_DENIAL_PERSONAS",
          "KNOWLEDGE_CANDOUR_PERSONAS", "SYCOPHANCY_PERSONAS", "ANTI_SYCOPHANCY_PERSONAS"):
    m = re.search(g + r"\s*=\s*\((.*?)\)", src, re.S)
    print("  %-28s %d personas" % (g, m.group(1).count('"') // 2 if m else -1))

# -------------------------------------------------------------------- O72 8B
print("\n=== O72-scale-8b-bank (file side; row counts come from p2v_B_O72_8b.ps1) ===")
sc = [d for d in LABALL if d["relpath"].startswith("run-scale/8b")]
print("run-scale/8b files:", len(sc),
      Counter(d["relpath"].split("/")[2] for d in sc if d["relpath"].count("/") > 1).most_common())
h = {d["relpath"]: d["md5"] for d in LABALL if d["relpath"] == "scale/scripts/run_scale_chain.ps1"}
import hashlib
loc = os.path.join(ROOT, "scale", "scripts", "run_scale_chain.ps1")
print("run_scale_chain.ps1 lab md5 %s | repo md5 %s | %s"
      % (list(h.values())[0], hashlib.md5(open(loc, "rb").read()).hexdigest(),
         "SAME" if list(h.values())[0] == hashlib.md5(open(loc, "rb").read()).hexdigest() else "DIFFERENT"))
for ln, line in enumerate(open(loc, encoding="utf-8", errors="replace").read().splitlines(), 1):
    if "Llama-3.1-8B" in line or "judge_model" in line or "held constant" in line:
        print("   %d: %s" % (ln, line.strip()[:110]))

# ------------------------------------------------------------ O12 passB_fine
print("\n=== O12-extra-P0c-passBfine ===")
pb = [d for d in LABALL if d["relpath"].startswith("run-crossgen/passB_fine")]
lay = Counter()
for d in pb:
    m = re.search(r"_L(\d+)[_.]", d["relpath"])
    if m:
        lay[int(m.group(1))] += 1
print("files: %d | layers in filenames: %s" % (len(pb), sorted(lay)))
print("by mtime date:", sorted(Counter(d["mtime"][:10] for d in pb).items()))
print("banks:", Counter(d["relpath"].split("/")[3] for d in pb
                        if d["relpath"].startswith("run-crossgen/passB_fine/banks/")).most_common())

# ------------------------------------------------------------- OR-04 coverage
print("\n=== OR-04 run-crossgen root + banks coverage ===")
for store in ["run-crossgen/matrices/"] + ["run-crossgen/banks/%s/matrices/" % b for b in
                                           sorted({d["relpath"].split("/")[2] for d in LABALL
                                                   if d["relpath"].startswith("run-crossgen/banks/")})]:
    L, S, P = set(), set(), set()
    n = 0
    for d in LABALL:
        if d["relpath"].startswith(store):
            m = re.search(r"(mean|last)_L(\d+)_s(\d+)\.json$", d["relpath"])
            if m:
                n += 1
                P.add(m.group(1)); L.add(int(m.group(2))); S.add(int(m.group(3)))
    print("  %-46s %4d files | %2d layers | seeds %s | pooling %s" % (store, n, len(L), sorted(S), sorted(P)))

# ---------------------------------------------------------------- OR-05 files
print("\n=== OR-05 length-control files + whether any doc names them ===")
for name in ("confounds_L44_last.json", "p1_L44_last_lengthctl.json",
             "q5_semantics_lengthctl.json", "p1_L44_last_withsyc.json"):
    for d in LABALL:
        if d["relpath"] == "run-coords/" + name:
            print("  %-32s %8s B  %s  md5 %s" % (name, d["bytes"], d["mtime"], d["md5"][:8]))
    hits = subprocess.run(["git", "grep", "-l", name.replace(".json", ""), "HEAD", "--",
                           "docs", "kd", "cap", "scale", "src", "README.md"],
                          cwd=ROOT, capture_output=True).stdout.decode("utf-8", "replace").split()
    print("        named in (HEAD, docs/kd/cap/scale/src):", hits or "NOWHERE")

# --------------------------------------------------------------------- F-22
print("\n=== F-22 1_generate_ndif.py exception handler ===")
for rev in ("107879d", "106d09f", "3dfdb67", "HEAD"):
    out = subprocess.run(["git", "show", rev + ":src/assistant_axis_ndif/1_generate_ndif.py"],
                         cwd=ROOT, capture_output=True).stdout.decode("utf-8", "replace").splitlines()
    marks = [(i + 1, l.strip()[:90]) for i, l in enumerate(out) if "str(e)[:200]" in l]
    print("  %-8s %d lines | str(e)[:200] at %s" % (rev, len(out), marks))
labmd5 = [d["md5"] for d in LABALL if d["relpath"] == "src/assistant_axis_ndif/1_generate_ndif.py"]
locmd5 = hashlib.md5(open(os.path.join(ROOT, "src", "assistant_axis_ndif",
                                       "1_generate_ndif.py"), "rb").read()).hexdigest()
print("  lab md5 %s | repo md5 %s | %s" % (labmd5[0], locmd5, "SAME" if labmd5[0] == locmd5 else "DIFFERENT"))
