"""Progress and completeness counts for one xmodel run, read from disk only.

The chain gates every stage on these numbers rather than on an exit code, because a stage can
exit 0 having produced nothing (gates-must-assert-their-own-number), and the monitor reads the
same output so what it reports is what the chain decided on.

    python xmodel_status.py --root D:/user4fina/Qwen2.5-72B-Instruct [--json]

Counts:
  responses   rollouts per role (responses/<role>.jsonl lines)
  scores      judged rollouts per role (scores/<role>.json keys)
  vectors_*   persona vector files present, and their layer count / width
  eval        rows pooled per dataset against the rows its items.pt says exist
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_lines(p: Path) -> int:
    with p.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def persona_counts(persona: Path) -> dict:
    out: dict = {"responses": {}, "scores": {}}
    for f in sorted((persona / "responses").glob("*.jsonl")):
        out["responses"][f.stem] = count_lines(f)
    for f in sorted((persona / "scores").glob("*.json")):
        try:
            out["scores"][f.stem] = len(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 -- a file mid-write reads as unknown, not zero
            out["scores"][f.stem] = -1
    out["n_rollouts"] = sum(out["responses"].values())
    out["n_scored"] = sum(v for v in out["scores"].values() if v > 0)
    for tag in ("vectors_s3", "vectors_all"):
        files = sorted((persona / tag).glob("*.pt"))
        out[tag] = {"n": len(files), "shape": None}
        if files:
            import torch
            v = torch.load(files[0], weights_only=False, map_location="cpu")["vector"]
            out[tag]["shape"] = list(v.shape)
    return out


def eval_counts(ev: Path) -> dict:
    import torch
    out = {}
    for d in sorted(p for p in ev.glob("*") if p.is_dir()):
        items_p = d / "items.pt"
        n_items = len(torch.load(items_p, weights_only=False)) if items_p.exists() else None
        done, shape = set(), None
        for c in sorted(d.glob("chunk_*.pt")):
            try:
                blob = torch.load(c, weights_only=False, map_location="cpu")
            except Exception:  # noqa: BLE001
                continue
            done |= set(blob["row_ids"])
            shape = list(blob["pooled"].shape[1:])
        out[d.name] = {"items": n_items, "done": len(done), "row_shape": shape}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    rep = {"persona": persona_counts(root / "persona"), "eval": eval_counts(root / "eval")}
    if a.json:
        print(json.dumps(rep))
        return
    p = rep["persona"]
    print(f"rollouts {p['n_rollouts']} in {len(p['responses'])} role files; "
          f"scored {p['n_scored']} in {len(p['scores'])} files")
    for tag in ("vectors_s3", "vectors_all"):
        print(f"{tag}: {p[tag]['n']} files, shape {p[tag]['shape']}")
    for k, v in rep["eval"].items():
        print(f"eval {k}: {v['done']}/{v['items']} rows, row shape {v['row_shape']}")


if __name__ == "__main__":
    main()
