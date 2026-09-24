"""Phase 1 (extra task): build session_experiments.csv from the raw transcript events.

Input : audit/phase1/sessions/run_events2_raw.tsv  (from p1_sess_events2.py)
        audit/experiments.csv                      (114 rows, for matches_experiments_csv)
Output: audit/phase1/sessions/session_experiments.csv      (tests/experiments only)
        audit/phase1/sessions/session_events_all.csv       (superset incl. infra rows)

Category of each event:
  measure  an analysis/statistics run (numbered analysis script, probe/axis/bootstrap
           script, or inline python that calls a measurement AND printed a number)
  run      a compute chain launched: run_*.ps1 driver, night_chain, rebuild, or a
           Register-/Start-ScheduledTask for a compute task
  infra    venv/rclone/network/task-plumbing/doc edits -> kept only in the superset

Repeats of the same (session, signature, args) collapse to the first occurrence
(`repeats`). Events with the same timestamp+command in two sessions are marked
`dup_session` (the 09-10 pair 2394bebd/40b4a5dd is one conversation recorded twice).

Usage:  PYTHONIOENCODING=utf-8 python audit/scripts/p1_sess_expcsv.py
"""
import csv
import io
import re
from collections import OrderedDict

RAW = "audit/phase1/sessions/run_events2_raw.tsv"
EXPC = "audit/experiments.csv"
OUT_TESTS = "audit/phase1/sessions/session_experiments.csv"
OUT_ALL = "audit/phase1/sessions/session_events_all.csv"

MEAS = re.compile(r"roc_auc|AUROC|bootstrap|LogisticRegression|np\.dot|corrcoef|ttest|"
                  r"permutation|cosine|np\.percentile|significan|\bauc\b", re.I)
NUMOUT = re.compile(r"\d\.\d{3}")
ANALYSIS_PY = re.compile(
    r"^(?:\d+[a-z]?_|pc_|cap_|kd_|make_|build_|check_|score_|judge_|audit_|repro_|"
    r"paper_comparison|matched_profile|layer_profile|len_confound|response_lengths|"
    r"ceiling_null|probe_limits|factcheck|cov|len|eval_datasets|show_|read_|verify_|"
    r"inspect_|fingerprint|bardchk|joinerr|bankchk|b2|f)\w*\.py$", re.I)
RUN_PS1 = re.compile(r"^(run_|night_chain|rebuild_|launch_|vectors_loop|watch_and_launch|"
                     r"remote_run_|reextract|restart_|prep_|runq|q\d+_smoke|build_label_banks)", re.I)
COMPUTE_TASK = re.compile(r"Coords|AA-Supervisor|Cap-|KD-|Scale-|Crossgen|Premise|Night-Chain|"
                          r"PassC|PassA|Sarc|Falsify|HPKR|Q8|Q11|Extract|Acts", re.I)
ARG = re.compile(r"--[a-z_][\w-]*(?:[ =][^\s\"';|&]+)?", re.I)
PATHS = re.compile(r"(?:run-[\w\-]+|kd|cap|results|data|figures)[/\\][\w./\\-]+")
LAYER = re.compile(r"--layers?[ =]([\w,]+)|\bL(\d{1,2})\b")
LABHINT = re.compile(r"user4|100\.85\.217\.68|labpc_run|ssh |psrun\.sh|pyrun\.sh|\.venv-analysis|\.venv-aa", re.I)
NDIFHINT = re.compile(r"\bndif\b|nnsight|remote=True|_ndif\b|_ndif\.py", re.I)
DOC_EDIT = re.compile(r"\.md\b|FINDINGS|memory[\\/]|plans?[\\/]|\bplan\.md", re.I)
INFRA = re.compile(r"venv|rclone|tailscale|scp |Copy-Item|winget|install|uv\.exe|Get-Content|"
                   r"disk|Test-Path|netsh|Get-Process", re.I)
TASKNAME = re.compile(r"-TaskName\s+['\"]?([\w\-]+)|/tn\s+['\"]?([\w\-]+)")


def one(s, n):
    return re.sub(r"\s+", " ", (s or "").replace("\x00", "")).strip()[:n]


def load_exp_index():
    by_script, by_path, by_task = {}, {}, {}
    with io.open(EXPC, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            eid = r["exp_id"]
            blob = " ".join((r.get(k) or "") for k in
                            ("script", "config_or_args", "scheduled_task", "results_location", "notes"))
            for n in set(re.findall(r"[\w\-]+\.(?:py|ps1|ipynb)", blob)):
                by_script.setdefault(n.split("/")[-1].split("\\")[-1], set()).add(eid)
            for p in set(re.findall(r"run-[\w\-]+(?:/[\w.\-]+){0,2}", blob.replace("\\", "/"))):
                by_path.setdefault(p, set()).add(eid)
            for t in set(re.findall(r"\b[A-Z][\w]*(?:-[A-Z0-9][\w]*)+\b", (r.get("scheduled_task") or ""))):
                by_task.setdefault(t.lower(), set()).add(eid)
    return by_script, by_path, by_task


def categorise(kind, sig, det, res, say):
    if kind == "sched_task":
        return "run" if COMPUTE_TASK.search(det) else "infra"
    if kind == "run_ps1":
        return "run" if RUN_PS1.match(sig) else "infra"
    if kind in ("run_script", "run_lab"):
        if ANALYSIS_PY.match(sig):
            return "measure"
        return "run" if re.search(r"_ndif\.py$|chain|pipeline", sig, re.I) else "infra"
    if kind == "run_inline":
        if DOC_EDIT.search(det[:200]) and not NUMOUT.search(res + say):
            return "infra"
        if MEAS.search(det) and (NUMOUT.search(res) or NUMOUT.search(say)):
            return "measure"
        return "infra"
    if kind == "notebook":
        return "run"
    return "infra"


def main():
    by_script, by_path, by_task = load_exp_index()
    rows = [l.rstrip("\n").split("\t") for l in io.open(RAW, encoding="utf-8")][1:]
    keep = OrderedDict()
    for r in rows:
        if len(r) < 9:
            continue
        sess, ln, ts, kind, sig, desc, det, res, nxt = r[:9]
        if kind in ("poll_ps1", "write_script", "inline_inspect"):
            continue
        args = " ".join(ARG.findall(det))[:120]
        # distinguishing tokens: flags, bank files, output roots. Without these two runs
        # of the same script with different banks collapse into one row.
        key_args = " ".join(sorted(set(ARG.findall(det))
                                   | set(re.findall(r"[\w:./\-]+\.pt", det))
                                   | set(re.findall(r"run-[\w]+(?:[/\][\w.-]+){0,3}", det))))[:240]
        key = (sess, kind, sig, key_args if kind != "run_inline" else one(det, 90))
        if key in keep:
            e = keep[key]
            e["n"] += 1
            if not NUMOUT.search(e["res"]) and res:
                e["res"] = res
            if not e["say"] and nxt:
                e["say"] = nxt
            continue
        keep[key] = dict(sess=sess, line=ln, ts=ts, kind=kind, sig=sig, desc=desc,
                         det=det, res=res, say=nxt, args=args, n=1)

    seen_global = {}
    out = []
    for i, (key, e) in enumerate(sorted(keep.items(), key=lambda kv: (kv[1]["ts"], kv[1]["sess"])), 1):
        det = e["det"]
        cat = categorise(e["kind"], e["sig"], det, e["res"], e["say"])
        gk = (e["ts"], e["sig"], one(det, 80))
        dup = seen_global.get(gk, "")
        if not dup:
            seen_global[gk] = e["sess"]
        paths = "; ".join(sorted(set(p.replace("\\", "/") for p in PATHS.findall(det)))[:4])[:130]
        lay = ""
        m = LAYER.search(det)
        if m:
            lay = "L" + (m.group(1) or m.group(2))
        where = "lab" if LABHINT.search(det) else "laptop"
        if NDIFHINT.search(det + " " + e["desc"]):
            where += "+NDIF"
        eids = set()
        if e["sig"].endswith((".py", ".ps1")):
            eids |= by_script.get(e["sig"], set())
        for p in re.findall(r"run-[\w\-]+(?:/[\w.\-]+){0,2}", det.replace("\\", "/")):
            eids |= by_path.get(p, set())
        tn = TASKNAME.search(det)
        if tn:
            eids |= by_task.get((tn.group(1) or tn.group(2) or "").lower(), set())
        what = {"INLINE": "inline python", "INLINE-LAB": "inline python (on lab PC)"}.get(e["sig"], e["sig"])
        if tn:
            what += " [" + (tn.group(1) or tn.group(2)) + "]"
        if e["args"]:
            what += " " + e["args"][:70]
        res, say = e["res"], e["say"]
        outcome = res if NUMOUT.search(res) else (say if NUMOUT.search(say) else (res or say))
        out.append(dict(
            event_id="T%03d" % i, when=e["ts"], session=e["sess"], line=e["line"],
            category=cat, kind=e["kind"], repeats=e["n"], dup_session=dup,
            what_ran=one(what, 150), desc=one(e["desc"], 110),
            target=one((paths + (" " + lay if lay else "")).strip(), 130),
            where_run=where, outcome_stated=one(outcome, 200),
            matches_experiments_csv=";".join(sorted(eids)) if eids else "NEW",
            evidence_quote=one(det, 200)))

    cols = ["event_id", "when", "session", "line", "category", "kind", "repeats", "dup_session",
            "what_ran", "desc", "target", "where_run", "outcome_stated",
            "matches_experiments_csv", "evidence_quote"]
    with io.open(OUT_ALL, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow(r)
    tests = [r for r in out if r["category"] in ("measure", "run")]
    with io.open(OUT_TESTS, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in tests:
            w.writerow(r)
    print("all", len(out), "tests", len(tests),
          "NEW", sum(1 for r in tests if r["matches_experiments_csv"] == "NEW"),
          "dupes", sum(1 for r in tests if r["dup_session"]))


if __name__ == "__main__":
    main()
