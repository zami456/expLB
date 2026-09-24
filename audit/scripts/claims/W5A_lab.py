"""Audit Phase 2 / W5A-ledger - read-only recomputation on the lab PC.

Covers F-21, F-45, F-49, F-53, F-69, F-90, F-103.
Reads only; writes nothing anywhere. Run with:
  python audit/scripts/labpc_run.py audit/scripts/p1_labpc_runpy.ps1 FILE=W5A_lab.py "ARGS=<section>"
"""
import ast
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import sys

E = "C:/Users/user4/expLB"
csv.field_size_limit(10_000_000)


def head(t):
    print("\n" + "=" * 78)
    print("==", t)
    print("=" * 78)


def info(p):
    if not os.path.exists(p):
        return "MISSING"
    st = os.stat(p)
    return "%d B  mtime %s" % (st.st_size, datetime.datetime.fromtimestamp(st.st_mtime))


def grep(p, pat, maxn=25, enc=None):
    if not os.path.exists(p):
        print("  MISSING:", p)
        return
    encs = [enc] if enc else ["utf-8", "utf-16"]
    for e in encs:
        try:
            with open(p, encoding=e, errors="replace") as f:
                n = 0
                for i, line in enumerate(f, 1):
                    if re.search(pat, line, re.I):
                        print("  %6d| %s" % (i, line.rstrip()[:240]))
                        n += 1
                        if n >= maxn:
                            print("  ... (truncated at %d matches)" % maxn)
                            break
            return
        except UnicodeError:
            continue


# ------------------------------------------------------------------ F-21
def f21():
    head("F-21  NDIF per-process memory cap  (logs/scale_3b_generate.log)")
    log = E + "/logs/scale_3b_generate.log"
    print(" log:", info(log))
    grep(log, r"GiB", 12)
    print(" -- lines naming gen_batch / batch sweep --")
    grep(log, r"gen_batch|--batch|batch\s*=\s*\d", 12)
    print(" -- smoke3b store --")
    for p in [E + "/run-scale/smoke3b", E + "/run-scale"]:
        print(" ", p, "->", "EXISTS" if os.path.exists(p) else "MISSING")
        if os.path.isdir(p):
            print("   ", sorted(os.listdir(p))[:25])
    print(" -- chain script settings --")
    for p in glob.glob(E + "/scale/scripts/*.ps1") + glob.glob(E + "/src/assistant_axis_ndif/*.ps1"):
        try:
            t = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if "gen_batch" in t:
            for i, line in enumerate(t.splitlines(), 1):
                if "gen_batch" in line:
                    print("  %s:%d  %s" % (os.path.basename(p), i, line.strip()[:200]))


# ------------------------------------------------------------------ F-45
def f45():
    head("F-45  OOM fix cost 8x throughput (logs/scale_coords_8b.log)")
    log = E + "/logs/scale_coords_8b.log"
    print(" log:", info(log))
    print(" -- lines mentioning chunk counts / rates / rows --")
    grep(log, r"chunk|rows ->|rows/min|KB|sample", 30)
    print(" -- 8B coords store --")
    d = E + "/run-scale/8b/coords"
    print(" ", d, info(d))
    if os.path.isdir(d):
        for sub in sorted(os.listdir(d)):
            p = os.path.join(d, sub)
            if os.path.isdir(p):
                ch = glob.glob(p + "/chunk_*")
                rj = p + "/rows.jsonl"
                nrows = sum(1 for _ in open(rj, encoding="utf-8")) if os.path.exists(rj) else None
                sz = sum(os.path.getsize(c) for c in ch) if ch else 0
                print("   %-35s chunks=%-6d meanKB=%-8.1f rows=%s" %
                      (sub, len(ch), (sz / len(ch) / 1024) if ch else 0, nrows))
    print(" -- passB per-dataset row counts (for the 14,718 / 6,551 arithmetic) --")
    tot = 0
    capped = 0
    pb = E + "/run-coords/passB"
    if os.path.isdir(pb):
        for sub in sorted(os.listdir(pb)):
            rj = os.path.join(pb, sub, "rows.jsonl")
            if os.path.exists(rj):
                n = sum(1 for _ in open(rj, encoding="utf-8"))
                tot += n
                capped += min(800, n)
                print("   %-40s %6d  -> capped %4d" % (sub, n, min(800, n)))
        print("   TOTAL %d ; sum(min(800,n)) = %d ; ratio %.4f" %
              (tot, capped, capped / tot if tot else 0))


# ------------------------------------------------------------------ F-49
def f49():
    head("F-49 / F-50  3B vs 8B throughput")
    for rung in ["3b", "8b"]:
        d = "%s/run-scale/%s/coords/instructed-deception" % (E, rung)
        print(" ", d, "->", "EXISTS" if os.path.isdir(d) else "MISSING")
        if not os.path.isdir(d):
            continue
        ch = sorted(glob.glob(d + "/chunk_*"))
        mt = [os.path.getmtime(c) for c in ch]
        rj = d + "/rows.jsonl"
        nrows = sum(1 for _ in open(rj, encoding="utf-8")) if os.path.exists(rj) else None
        print("    chunks=%d  rows.jsonl=%s" % (len(ch), nrows))
        if mt:
            t0, t1 = min(mt), max(mt)
            span = (t1 - t0) / 60
            print("    first chunk %s   last %s   span %.1f min" %
                  (datetime.datetime.fromtimestamp(t0), datetime.datetime.fromtimestamp(t1), span))
            if span > 0:
                print("    chunks/min = %.2f ; rows/chunk = %s ; rows/min = %s" %
                      (len(ch) / span,
                       ("%.2f" % (nrows / len(ch))) if nrows else "?",
                       ("%.1f" % (nrows / span)) if nrows else "?"))
            # the 24.5-minute window F-49 measured: 306 chunks from 19:36:40
            t0d = datetime.datetime.fromtimestamp(t0)
            w = t0 + 24.5 * 60
            print("    chunks written within 24.5 min of the first: %d" % sum(1 for m in mt if m <= w))
        it = d + "/items.pt"
        print("    items.pt:", info(it))
    print(" -- logs --")
    for lg in ["scale_coords_3b.log", "scale_coords_8b.log"]:
        p = E + "/logs/" + lg
        print(" ", lg, info(p))
        grep(p, r"instructed-deception.*(rows|spans)|sample 800|--sample", 6)


# ------------------------------------------------------------------ F-53
def f53():
    head("F-53  verb form kills the spelunking (cap/run/smoke_v4)")
    d = E + "/cap/run/smoke_v4"
    print(" ", d, "->", "EXISTS" if os.path.isdir(d) else "MISSING")
    if not os.path.isdir(d):
        return
    SPEL = re.compile(r"pickaxe|helmet|spelunk|underground|cavern|\bcave\b|\bcaves\b|caving|"
                      r"\bcaver\b|stalact|headlamp|tunnel|\bmine\b|rope", re.I)
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if not fn.endswith(".jsonl"):
            print("  (skip)", fn, info(p))
            continue
        recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        print("\n  -- %s : %d records ; keys=%s" % (fn, len(recs), sorted(recs[0].keys())))
        hits = 0
        for r in recs:
            blob = json.dumps(r, ensure_ascii=False)
            # look only at the model's own turns if they are identifiable
            texts = []
            for k in ("turn1", "turn2", "response", "responses", "messages", "text"):
                if k in r:
                    texts.append(json.dumps(r[k], ensure_ascii=False))
            t = " ".join(texts) if texts else blob
            m = set(x.lower() for x in SPEL.findall(t))
            if m:
                hits += 1
            print("     item=%s  spelunking_markers=%s" %
                  (str(r.get("item_id") or r.get("qid") or r.get("id") or r.get("question"))[:60],
                   sorted(m)))
        print("     records with any spelunking marker: %d of %d" % (hits, len(recs)))


def f53b():
    head("F-53b  spelunking markers in the ASSISTANT turns only (v3 vs v4)")
    SPEL = re.compile(r"pickaxe|helmet|spelunk|underground|cavern|\bcave\b|\bcaves\b|caving|"
                      r"\bcaver\b|stalact|headlamp|\bmine\b|\brope\b|\bgear\b|\bdepths\b", re.I)
    for d in sorted(glob.glob(E + "/cap/run/smoke_v*")) + [E + "/cap/run/smoke_cells", E + "/cap/run/smoke_tf"]:
        print("\n  ###", d)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".jsonl"):
                continue
            p = os.path.join(d, fn)
            recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            n_any = 0
            print("   -- %s : %d records" % (fn, len(recs)))
            for r in recs:
                conv = r.get("conversation") or []
                asst = []
                if isinstance(conv, list):
                    for m in conv:
                        if isinstance(m, dict) and m.get("role") == "assistant":
                            asst.append(m.get("content", ""))
                        elif isinstance(m, str):
                            asst.append(m)
                t1 = r.get("turn1") or (asst[0] if asst else "")
                t2 = asst[1] if len(asst) > 1 else ""
                m1 = sorted(set(x.lower() for x in SPEL.findall(t1 or "")))
                m2 = sorted(set(x.lower() for x in SPEL.findall(t2 or "")))
                if m1 or m2:
                    n_any += 1
                print("     q=%-45s turns=%d t1_markers=%s t2_markers=%s" %
                      (str(r.get("question"))[:45], len(asst), m1, m2))
            print("     records with any marker in an assistant turn: %d of %d" % (n_any, len(recs)))
        # the system prompt actually used
        f0 = sorted(x for x in os.listdir(d) if x.endswith(".jsonl"))
        if f0:
            r0 = json.loads(open(os.path.join(d, f0[0]), encoding="utf-8").readline())
            print("   system_prompt[0]:", str(r0.get("system_prompt"))[:400])


def f53text():
    head("F-53  the four quoted turn-2 texts")
    d = E + "/cap/run/smoke_v4"
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".jsonl"):
            continue
        for l in open(os.path.join(d, fn), encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            blob = json.dumps(r, ensure_ascii=False)
            for kw in ["Mariana", "capital of Australia", "smallest prime", "Tungsten",
                       "Ottawa", "Sydney", "Wolfram"]:
                if kw.lower() in blob.lower():
                    i = blob.lower().index(kw.lower())
                    print("  [%s] %s ... %s" % (fn, kw, blob[max(0, i - 400):i + 400]))
                    break


# ------------------------------------------------------------------ F-69
def f69():
    head("F-69  passA matched slice repaired")
    # 1. the slice definition, reimplemented from eval_datasets.py:363-421
    DEN = re.compile(r"\b(do not|don't|doesn't|not)\b[^.]{0,20}\bknow\b", re.I)
    ASS = re.compile(r"^\s*(yes\b|i know\b)", re.I)
    ENV = {"nan": float("nan"), "inf": float("inf"), "NaN": float("nan")}
    try:
        import numpy as np
        ENV["np"] = np
    except ImportError:
        pass

    def parse_meta(s):
        try:
            return eval(s, {"__builtins__": {}}, dict(ENV))  # noqa: S307 - repr of a dict
        except Exception:
            return {}

    path = E + "/results/subsets/harm-pressure-knowledge-report.csv"
    print(" csv:", info(path))
    matched, labels = set(), {}
    nall = nllama = 0
    with open(path, encoding="utf-8", newline="") as fh:
        for i, rec in enumerate(csv.DictReader(fh)):
            nall += 1
            if rec.get("model") != "llama-v3.3-70b-instruct":
                continue
            nllama += 1
            try:
                msgs = ast.literal_eval(rec["messages"])
            except Exception:
                continue
            asst = [m for m in msgs if m.get("role") == "assistant"]
            if not asst:
                continue
            sub = rec.get("dataset") or ""
            idx = rec.get("dataset_index") or rec.get("index") or str(i)
            item = ("%s:%s" % (sub, idx)) if sub else str(idx)
            txt = asst[-1].get("content", "")
            if DEN.search(txt or "") and not ASS.match(txt or ""):
                rid = "harm-pressure-knowledge-report:%s:%d" % (item, i)
                matched.add(rid)
                labels[rid] = rec.get("deceptive")
    print(" csv rows total=%d  llama rows=%d  text_matched ids=%d" % (nall, nllama, len(matched)))
    lab = {}
    for rid, v in labels.items():
        lab[str(v)] = lab.get(str(v), 0) + 1
    print(" deceptive column over the matched ids:", lab)

    # 2. each store
    for store in ["passA", "passB", "passC"]:
        p = "%s/run-coords/%s/harm-pressure-knowledge-report/rows.jsonl" % (E, store)
        print("\n ", store, info(p))
        if not os.path.exists(p):
            continue
        ids, labs, keys = [], [], None
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if keys is None:
                    keys = sorted(r.keys())
                ids.append(r.get("row_id"))
                labs.append(r.get("label", r.get("deceptive")))
        print("    rows=%d  keys=%s" % (len(ids), keys))
        print("    md5 of file:", hashlib.md5(open(p, "rb").read()).hexdigest())
        inter = [i for i in ids if i in matched]
        print("    rows in the text_matched slice: %d" % len(inter))
        d = h = 0
        for i, rid in enumerate(ids):
            if rid in matched:
                v = labs[i]
                if v in (1, True, "1", "True", "true"):
                    d += 1
                else:
                    h += 1
        print("    of those: deceptive=%d honest=%d" % (d, h))
    # 3. the gate line
    print("\n  -- logs/hpkr_topup.status --")
    grep(E + "/logs/hpkr_topup.status", r".", 40)


# ------------------------------------------------------------------ F-90
def f90():
    head("F-90  external review audit  (run-premise/audit/audit_L40_s42.json)")
    p = E + "/run-premise/audit/audit_L40_s42.json"
    print(" ", info(p))
    if not os.path.exists(p):
        return
    d = json.load(open(p, encoding="utf-8"))

    def walk(o, pre="", depth=0):
        if depth > 2:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    print("  %s%s = %s" % ("  " * depth, pre + k, v))
                elif isinstance(v, list):
                    print("  %s%s : list[%d] %s" % ("  " * depth, pre + k, len(v), str(v[:6])[:220]))
                    if v and isinstance(v[0], dict) and depth < 2:
                        walk(v[0], pre="[0].", depth=depth + 1)
                else:
                    print("  %s%s :" % ("  " * depth, pre + k))
                    walk(v, depth=depth + 1)
    walk(d)
    # the 20 draws
    for k in d:
        if "draw" in k.lower():
            v = d[k]
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                n = len(v)
                m = sum(v) / n
                sd = (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5
                print("\n  %s: n=%d mean=%.4f min=%.4f max=%.4f sd=%.4f" %
                      (k, n, m, min(v), max(v), sd))
    print("\n -- which bank file the Q5 arm used --")
    for c in [E + "/run/axes.pt", E + "/run-premise/axes.pt"]:
        print("  ", c, info(c))
        if os.path.exists(c):
            print("   md5", hashlib.md5(open(c, "rb").read()).hexdigest())
    for c in [E + "/run-paper16/axes_vectors_s3.pt", E + "/run-paper16/axes_vectors_s3_fp32.pt",
              E + "/run-kd/axes_vectors_s3.pt"]:
        if os.path.exists(c):
            print("  ", c, hashlib.md5(open(c, "rb").read()).hexdigest())
    print("\n -- the script --")
    s = E + "/src/assistant_axis_ndif/30f_result2_audit.py"
    print("  ", info(s))
    if os.path.exists(s):
        for i, line in enumerate(open(s, encoding="utf-8", errors="replace"), 1):
            if re.search(r"axes\.pt|bank|--bank|load_axes|own100|paper16|run/axes", line, re.I):
                print("   %4d| %s" % (i, line.rstrip()[:200]))


# ------------------------------------------------------------------ F-103
def f103():
    head("F-103  kd-bank share is pooling-robust")
    base = E + "/run-crossgen/kd11"
    out = {}
    for pooling in ["mean", "last"]:
        rows = []
        for layer in range(0, 81):
            cp = "%s/banks/collab/matrices/%s_L%d_s42.json" % (base, pooling, layer)
            rp = "%s/matrices/%s_L%d_s42.json" % (base, pooling, layer)
            if not (os.path.exists(cp) and os.path.exists(rp)):
                continue
            c = json.load(open(cp, encoding="utf-8"))
            r = json.load(open(rp, encoding="utf-8"))
            zs = {z["direction"]: z for z in c["zeroshot"]}
            if "contrast" not in zs or "contrast:collab_extended" not in zs:
                continue
            a1 = zs["contrast"]["auroc"]
            e1 = zs["contrast:collab_extended"]["auroc"]
            ds = c["datasets"]
            n = len(ds)
            ma1 = sum(a1) / n
            me1 = sum(e1) / n
            rec = [x for x in r["results"] if x["feature"] == "raw" and x["probe"] == "paper"][0]
            A = rec["auroc"]
            off = [A[i][j] for i in range(len(A)) for j in range(len(A)) if i != j]
            raw = sum(off) / len(off)
            rows.append(dict(layer=layer, n=n, axis1=ma1, ext=me1, diff=me1 - ma1, raw=raw,
                             share=(me1 - ma1) / (me1 - raw) if me1 != raw else float("nan"),
                             bank=c.get("bank"), bank_name=c.get("bank_name"),
                             hpkr_a1=a1[ds.index("harm-pressure-knowledge-report")]
                             if "harm-pressure-knowledge-report" in ds else None,
                             hpkr_e1=e1[ds.index("harm-pressure-knowledge-report")]
                             if "harm-pressure-knowledge-report" in ds else None))
        out[pooling] = rows
        print("\n  == pooling = %s ; %d layers ==" % (pooling, len(rows)))
        print("  layer  n  axis1   ext_axis1  diff     raw     share   hpkr_a1 hpkr_e1")
        for x in rows:
            print("   %-5d %2d %.4f  %.4f   %+.4f  %.4f  %6.1f%%  %s %s" %
                  (x["layer"], x["n"], x["axis1"], x["ext"], x["diff"], x["raw"],
                   100 * x["share"],
                   ("%.4f" % x["hpkr_a1"]) if x["hpkr_a1"] is not None else "-",
                   ("%.4f" % x["hpkr_e1"]) if x["hpkr_e1"] is not None else "-"))
        if rows:
            ds_ = [x["diff"] for x in rows]
            sh = sorted(x["share"] for x in rows)
            med = sh[len(sh) // 2] if len(sh) % 2 else (sh[len(sh) // 2 - 1] + sh[len(sh) // 2]) / 2
            print("  layer-mean diff = %+.4f ; range [%+.4f, %+.4f] ; positive at %d/%d ; "
                  "median share = %.1f%%" %
                  (sum(ds_) / len(ds_), min(ds_), max(ds_),
                   sum(1 for v in ds_ if v > 0), len(ds_), 100 * med))
            print("  bank recorded in the collab matrices:",
                  sorted(set((x["bank"], x["bank_name"]) for x in rows)))
    print("\n  -- file mtimes of the collab matrices --")
    for pooling in ["mean", "last"]:
        g = sorted(glob.glob("%s/banks/collab/matrices/%s_L*_s42.json" % (base, pooling)))
        if g:
            mt = [os.path.getmtime(x) for x in g]
            print("   %s: %d files, %s .. %s" %
                  (pooling, len(g), datetime.datetime.fromtimestamp(min(mt)),
                   datetime.datetime.fromtimestamp(max(mt))))


def f21b():
    head("F-21b  allocation histogram and batch sweep in scale_3b_generate.log")
    log = E + "/logs/scale_3b_generate.log"
    hist = {}
    first = {}
    with open(log, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            m = re.search(r"Tried to allocate ([\d.]+) MiB", line)
            if m:
                v = m.group(1)
                hist[v] = hist.get(v, 0) + 1
                first.setdefault(v, (i, line[:30]))
    print("  allocation sizes seen (MiB -> count, first line):")
    for v in sorted(hist, key=lambda x: float(x)):
        print("   %8s  n=%-5d first at line %d  %s" % (v, hist[v], first[v][0], first[v][1].strip()))
    print("\n  -- lines recording a batch setting or a completed-rows count --")
    with open(log, encoding="utf-8", errors="replace") as f:
        n = 0
        for i, line in enumerate(f, 1):
            if re.search(r"\b(gen_batch|batch_size|batches\b.*of|wrote|rows\b.*/\s*\d|/100)\b", line):
                print("   %6d| %s" % (i, line.rstrip()[:300]))
                n += 1
                if n > 40:
                    print("   ... truncated")
                    break
    print("\n  -- last 12 lines of the log --")
    lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
    for line in lines[-12:]:
        print("   ", line[:300])



def f53c():
    head("F-53c  full assistant turns, pc_caver, v3 (smoke_cells) vs v4 (smoke_v4)")
    for d in [E + "/cap/run/smoke_cells", E + "/cap/run/smoke_v4"]:
        for fn in ["pc_caver.jsonl", "pc_caver_control.jsonl"]:
            p = os.path.join(d, fn)
            if not os.path.exists(p):
                continue
            print(chr(10)+"  ##### %s / %s" % (os.path.basename(d), fn))
            for l in open(p, encoding="utf-8"):
                if not l.strip():
                    continue
                r = json.loads(l)
                conv = r.get("conversation") or []
                asst = [m.get("content", "") for m in conv
                        if isinstance(m, dict) and m.get("role") == "assistant"]
                print("   Q: %s  | correct=%s | turn1_is_correct=%s" %
                      (str(r.get("question"))[:70], r.get("correct_answer"),
                       r.get("turn1_is_correct")))
                for i, a in enumerate(asst, 1):
                    print("     A%d: %s" % (i, a.replace(chr(10), " ")[:420]))
            print("   system_prompt:", str(r.get("system_prompt"))[:300])


SECTIONS = dict(f21=f21, f21b=f21b, f45=f45, f49=f49, f53=f53, f53b=f53b, f53c=f53c, f53text=f53text,
                f69=f69, f90=f90, f103=f103)

if __name__ == "__main__":
    want = sys.argv[1:] or list(SECTIONS)
    for w in want:
        try:
            SECTIONS[w]()
        except Exception as exc:  # keep going; report the last line of the error
            import traceback
            print("!! %s failed: %s" % (w, traceback.format_exc().strip().splitlines()[-1]))
