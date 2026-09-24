"""W5E-ownerdoc lab-side checks. READ-ONLY on C:/Users/user4/expLB.

O-014  briefing:304  "+0.055-0.059, or +0.081 at L20 mean-pooled"
       -> run-crossgen/kd11/banks/collab/matrices/{mean,last}_L*_s42.json, zero-shot rows
          `contrast` (axis1, 8v7) and `contrast:collab_extended` (ext_axis1, 12v13),
          baseline raw from run-crossgen/kd11/matrices/{mean,last}_L*_s42.json.
O-022  briefing:354  single-source baseline variants table, sign-oriented mean 0.735
O-038  ROADMAP:28    the same 0.735
       -> run-premise/audit/audit_L40_s42.json (30f_result2_audit.py)
O-065  thesis:377    own100 bank: 100 roles, 240 shared questions x 5 prompts,
                     ~1,200 rollouts per role, kept median 1,163, judge Llama-3.1-70B
O-082  thesis:702    label lift +0.032 (log-only in 17_validate.py)
VERSION checks: lab vs laptop copies of the run-coords files behind O-075/O-082/O-086/O-088.
"""
import glob, hashlib, json, os, re, statistics, sys

E = "C:/Users/user4/expLB"


def info(p):
    if not os.path.exists(p):
        return f"{p}  MISSING"
    s = os.stat(p)
    import datetime
    return f"{p}  size={s.st_size}  mtime={datetime.datetime.fromtimestamp(s.st_mtime)}"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def offdiag_raw(path):
    d = json.load(open(path, encoding="utf-8"))
    ds = d["datasets"]
    for r in d["results"]:
        if r["feature"] == "raw" and r["probe"] == "paper":
            v = [r["auroc"][i][j] for i in range(len(ds)) for j in range(len(ds))
                 if i != j and r["auroc"][i][j] is not None]
            return statistics.fmean(v), len(v)
    return None, 0


def zmean(path, direction):
    d = json.load(open(path, encoding="utf-8"))
    ds = d["datasets"]
    z = next((x for x in d["zeroshot"] if x["direction"] == direction), None)
    if z is None:
        return None, None, d
    per = {ds[i]: z["auroc"][i] for i in range(len(ds)) if z["auroc"][i] is not None}
    return statistics.fmean(per.values()), per, d


print("#" * 78)
print("O-014 -- ext_axis1 minus axis1 over every stored layer, BOTH poolings")
for pool in ("mean", "last"):
    rows = []
    files = sorted(glob.glob(f"{E}/run-crossgen/kd11/banks/collab/matrices/{pool}_L*_s42.json"),
                   key=lambda s: int(re.search(r"_L(\d+)_", s).group(1)))
    print(f"-- pooling={pool}: {len(files)} layer files")
    for fp in files:
        L = int(re.search(r"_L(\d+)_", fp).group(1))
        bp = f"{E}/run-crossgen/kd11/matrices/{pool}_L{L}_s42.json"
        if not os.path.exists(bp):
            print(f"   L{L}: baseline MISSING {bp}"); continue
        a1, pa, d = zmean(fp, "contrast")
        e1, pe, _ = zmean(fp, "contrast:collab_extended")
        raw, n = offdiag_raw(bp)
        share = 100 * (e1 - a1) / (e1 - raw) if (e1 - raw) else float("nan")
        rows.append((L, a1, e1, raw, e1 - a1, share,
                     pa.get("harm-pressure-knowledge-report"), pe.get("harm-pressure-knowledge-report")))
        print(f"   L{L:<3} axis1={a1:.4f} ext={e1:.4f} raw={raw:.4f}(n={n}) | ext-axis1={e1-a1:+.4f}"
              f" share={share:7.1f}% | HPKR {pa.get('harm-pressure-knowledge-report'):.3f}"
              f"->{pe.get('harm-pressure-knowledge-report'):.3f}"
              + ("  axis1<raw" if a1 < raw else ""))
    if rows:
        d_ = [r[4] for r in rows]; sh = [r[5] for r in rows]
        print(f"   ==> layers={len(rows)}  positive at {sum(1 for x in d_ if x > 0)}")
        print(f"   ==> ext-axis1 mean {statistics.fmean(d_):+.4f}  "
              f"min {min(d_):+.4f} (L{rows[d_.index(min(d_))][0]})  "
              f"max {max(d_):+.4f} (L{rows[d_.index(max(d_))][0]})")
        print(f"   ==> value at L20 = {[r[4] for r in rows if r[0]==20]}")
        print(f"   ==> share median {statistics.median(sh):.1f}%  min {min(sh):.1f}%  max {max(sh):.1f}%")
    # provenance of the collab bank file
    if files:
        d0 = json.load(open(files[0], encoding="utf-8"))
        print(f"   bank={d0.get('bank')} bank_name={d0.get('bank_name')} k={d0.get('k')} "
              f"datasets={len(d0.get('datasets', []))} seed={d0.get('seed')}")
        print("   " + info(files[0]))

print()
print("#" * 78)
print("O-022 / O-038 -- single-source baseline variants, sign-oriented mean = 0.735")
for p in [f"{E}/run-premise/audit/audit_L40_s42.json",
          f"{E}/run-premise/audit/audit_L20_s42.json"]:
    print(info(p))
    if not os.path.exists(p):
        continue
    print("  md5", md5(p))
    d = json.load(open(p, encoding="utf-8"))

    def walk(o, pre=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (dict, list)):
                    walk(v, pre + "." + str(k))
                else:
                    print(f"    {pre}.{k} = {v!r}")
        elif isinstance(o, list):
            if o and not isinstance(o[0], (dict, list)):
                print(f"    {pre} = list[{len(o)}] {o[:6]}{'...' if len(o) > 6 else ''}")
            else:
                for i, v in enumerate(o[:60]):
                    walk(v, pre + f"[{i}]")
                if len(o) > 60:
                    print(f"    {pre} ... {len(o)-60} more")
    walk(d, "root")

print()
print("#" * 78)
print("O-065 -- own100 bank rollout counts")
for cand in [f"{E}/run", f"{E}/run-hf275", f"{E}/run-own100"]:
    if os.path.isdir(cand):
        print(f"  {cand}: {sorted(os.listdir(cand))[:40]}")
for resp in [f"{E}/run/responses", f"{E}/run-own100/responses"]:
    if os.path.isdir(resp):
        fs = sorted(os.listdir(resp))
        print(f"  {resp}: {len(fs)} files; first 5 {fs[:5]}")
        n_rows, n_kept = [], []
        for fn in fs:
            fp = os.path.join(resp, fn)
            if not fn.endswith((".jsonl", ".json")):
                continue
            c = 0
            with open(fp, encoding="utf-8") as f:
                for _ in f:
                    c += 1
            n_rows.append((fn, c))
        print(f"    rollout line counts: n_files={len(n_rows)} "
              f"min={min(x[1] for x in n_rows) if n_rows else '-'} "
              f"median={statistics.median([x[1] for x in n_rows]) if n_rows else '-'} "
              f"max={max(x[1] for x in n_rows) if n_rows else '-'} "
              f"total={sum(x[1] for x in n_rows)}")
for sc in [f"{E}/run/scores", f"{E}/run-own100/scores"]:
    if os.path.isdir(sc):
        fs = [f for f in sorted(os.listdir(sc)) if f.endswith((".jsonl", ".json"))]
        print(f"  {sc}: {len(fs)} files")
        kept = []
        for fn in fs:
            k = t = 0
            with open(os.path.join(sc, fn), encoding="utf-8") as f:
                for line in f:
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    t += 1
                    s = o.get("score", o.get("judge_score"))
                    if s == 3:
                        k += 1
            kept.append((fn, t, k))
        if kept:
            print(f"    judged rows per role: median={statistics.median([x[1] for x in kept])} "
                  f"total={sum(x[1] for x in kept)}")
            print(f"    score-3 kept per role: median={statistics.median([x[2] for x in kept])} "
                  f"min={min(x[2] for x in kept)} max={max(x[2] for x in kept)} total={sum(x[2] for x in kept)}")
            for fn, t, k in kept[:5]:
                print(f"      {fn}: judged={t} kept3={k}")
# the vectors themselves record how many rollouts they were built from
for vd in [f"{E}/run-hf275/vectors_100", f"{E}/run/vectors_s3", f"{E}/run/vectors_all"]:
    if os.path.isdir(vd):
        fs = sorted(os.listdir(vd))
        print(f"  {vd}: {len(fs)} files; first 5 {fs[:5]}")

print()
print("#" * 78)
print("O-082 -- the '+0.032 label lift' printed by 17_validate.py (log only)")
for pat in ["*valid*", "*17_*"]:
    for fp in sorted(glob.glob(f"{E}/logs/{pat}")):
        print(" ", info(fp))
        try:
            txt = open(fp, encoding="utf-8", errors="replace").read()
        except Exception as ex:
            print("   read error", ex); continue
        if "lift" in txt:
            for line in txt.splitlines():
                if "lift" in line or "decodab" in line or "VERDICT" in line:
                    print("    >", line.strip())

print()
print("#" * 78)
print("VERSION -- lab vs laptop copies of the run-coords files behind O-075/O-082/O-086/O-088")
for fn in ["q6_layers.json", "layer_sweep_passB_mms.json", "dotproducts_mean.json",
           "q3_q4_bootstrap.json", "validate_L40.json", "erase_test.json", "significance.json"]:
    p = f"{E}/run-coords/{fn}"
    if os.path.exists(p):
        print(f"  {info(p)}  md5={md5(p)}")
    else:
        print(f"  {p}  MISSING")
