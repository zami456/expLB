#!/usr/bin/env python
"""Materialise the published Assistant-Axis role vectors as a vectors_dir 6_axes can read.

`lu-christina/assistant-axis-vectors` stores one bare tensor per role, (80, 8192) bf16, plus
`default_vector.pt` and the published `assistant_axis.pt`. 6_axes.load_vectors accepts a bare
tensor and takes the role name from the filename, so the only work is copying them into one
folder and naming the default `default.pt` (the Assistant Axis needs an anchor whose name
contains "default").

    python 31_fetch_hf_vectors.py --out run-hf275/vectors_275
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "lu-christina/assistant-axis-vectors"
MODEL_DIR = "llama-3.3-70b"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roles", nargs="*", default=None,
                    help="only these roles (default: every published role)")
    args = ap.parse_args()

    base = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=[
        f"{MODEL_DIR}/role_vectors/*.pt", f"{MODEL_DIR}/default_vector.pt",
        f"{MODEL_DIR}/assistant_axis.pt"])) / MODEL_DIR
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.roles) if args.roles else None
    n = 0
    for src in sorted((base / "role_vectors").glob("*.pt")):
        if want is None or src.stem in want:
            shutil.copyfile(src, out / src.name)
            n += 1
    if want is None or "default" in want:
        shutil.copyfile(base / "default_vector.pt", out / "default.pt")
        n += 1
    shutil.copyfile(base / "assistant_axis.pt", out.parent / "published_assistant_axis.pt")
    print(f"{n} vectors in {out}")


if __name__ == "__main__":
    main()
