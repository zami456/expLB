#!/usr/bin/env python
"""Preflight: show NDIF's live state and fail fast before a long run.

    src/assistant_axis_ndif/0_preflight_ndif.py [--require REPO ...] [--warn-only] [--json]

NDIF availability drifts -- a model that is HOT+pinned in the morning can be COLD by
afternoon, and the subject model is not pinned so it can be evicted. Starting a
multi-day generation against a cold/absent model, or against a service that is down,
wastes hours before the failure surfaces. This checks the status endpoint (read-only,
no model load, no trace -- instant and cheap) and prints a verdict with an exit code
the driver gates on.

Verdicts / exit codes:
    READY      all required models RUNNING                     -> exit 0
    WARMING    a required model is DEPLOYING/WARM (coming up)   -> exit 0 (+ warning)
    NOT READY  a required model COLD/absent, or service DOWN    -> exit 2
With --warn-only the report is printed but the exit is always 0 (used for the judge,
which is needed hours later and is resumable, so it should not block stage 1).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]

load_dotenv(PIPELINE_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")
os.environ.setdefault("NDIF_HOST", "https://api.ndif.us")

# Windows (uv standalone Python): stale CA store breaks the status HTTPS call, and the
# cp1252 console can't encode status glyphs. Same two fixes the NDIF stages self-apply.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:  # noqa: BLE001
    pass

SUBJECT_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct"


def fetch_deployments() -> dict | None:
    """repo_id -> aggregated {state, level, pinned}; None if NDIF is unreachable.

    A repo can have several replicas; aggregate to the most-available: RUNNING beats
    other states, HOT beats WARM beats COLD, pinned if any replica is pinned.
    """
    import nnsight
    from nnsight import CONFIG

    host = os.environ["NDIF_HOST"]
    if not host.startswith("http"):
        host = "https://" + host
    CONFIG.API.HOST = host

    raw = nnsight.status(raw=True)  # returns {} on any request failure (does not raise)
    if not raw or "deployments" not in raw:
        return None

    def level_rank(lv: str) -> int:
        return {"HOT": 3, "WARM": 2, "COLD": 1}.get(lv or "COLD", 0)

    agg: dict[str, dict] = {}
    for v in raw["deployments"].values():
        if not isinstance(v, dict):
            continue
        repo = v.get("repo_id")
        if not repo:
            continue
        state = v.get("application_state") or "NOT_DEPLOYED"
        level = v.get("deployment_level") or "COLD"
        pinned = bool(v.get("pinned"))
        cur = agg.get(repo)
        if cur is None:
            agg[repo] = {"state": state, "level": level, "pinned": pinned}
            continue
        if state == "RUNNING":
            cur["state"] = "RUNNING"
        if level_rank(level) > level_rank(cur["level"]):
            cur["level"] = level
        cur["pinned"] = cur["pinned"] or pinned
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require", nargs="+", default=[SUBJECT_DEFAULT],
                    help="repo ids that must be RUNNING (default: the subject model)")
    ap.add_argument("--warn-only", action="store_true",
                    help="report and always exit 0 (use for the judge, needed later)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    agg = fetch_deployments()

    if agg is None:
        verdict, code = "NOT READY", 2
        if args.json:
            print(json.dumps({"service": "DOWN", "verdict": verdict, "require": {}}))
        else:
            print(f"NDIF host : {os.environ['NDIF_HOST']}")
            print("Service   : DOWN / unreachable (status endpoint returned nothing)")
            print(f"\nVERDICT: {verdict} -- cannot reach NDIF; check the network / NDIF_API_KEY "
                  "/ https://login.ndif.us, then retry.")
        return 0 if args.warn_only else code

    service = "UP" if any(d["state"] == "RUNNING" for d in agg.values()) else "REDEPLOYING"

    def classify(repo: str) -> str:
        d = agg.get(repo)
        if d is None:
            return "ABSENT"        # not in the payload at all
        if d["state"] == "RUNNING":
            return "RUNNING"
        if d["state"] in ("DEPLOYING",) or d["level"] in ("HOT", "WARM"):
            return "WARMING"
        return "COLD"

    req = {repo: {"class": classify(repo), **(agg.get(repo) or {"state": None, "level": "COLD", "pinned": False})}
           for repo in args.require}

    if all(v["class"] == "RUNNING" for v in req.values()):
        verdict, code = "READY", 0
    elif any(v["class"] in ("COLD", "ABSENT") for v in req.values()):
        verdict, code = "NOT READY", 2
    else:
        verdict, code = "WARMING", 0

    hot = sorted(r for r, d in agg.items() if d["state"] == "RUNNING" and d["level"] == "HOT")

    if args.json:
        print(json.dumps({"service": service, "verdict": verdict, "require": req,
                          "hot_running": hot}, indent=2))
    else:
        print(f"NDIF host : {os.environ['NDIF_HOST']}")
        print(f"Service   : {service}")
        print("\nRequired models:")
        for repo, v in req.items():
            pin = "pinned" if v["pinned"] else "unpinned"
            mark = "OK " if v["class"] == "RUNNING" else "!! "
            print(f"  {mark}{repo:44} {v['class']:8} ({v.get('state') or 'n/a'} - "
                  f"{v['level']} - {pin})")
        print(f"\nOther HOT+RUNNING models now ({len(hot)}):")
        for r in hot:
            pin = " [pinned]" if agg[r]["pinned"] else ""
            print(f"    {r}{pin}")
        print(f"\nVERDICT: {verdict}")
        if verdict == "READY":
            print("  All required models are up. Safe to run.")
        elif verdict == "WARMING":
            print("  A required model is deploying/warm -- it should come up shortly. "
                  "Every stage is resumable, so proceeding is safe.")
        else:
            cold = [r for r, v in req.items() if v["class"] in ("COLD", "ABSENT")]
            print(f"  Cold/absent: {', '.join(cold)}. Submitting now triggers a slow cold "
                  "deploy or errors. Wait and re-check, or use a HOT model above "
                  "(e.g. set JUDGE_MODEL to a HOT pinned one).")

    return 0 if args.warn_only else code


if __name__ == "__main__":
    sys.exit(main())
