#!/usr/bin/env python
"""The cosine gate: is paper16 less Assistant-Axis-contaminated than own100?

This is the number the whole paper16 bank exists to produce. Our 100-role substitute
bank and the published HF vectors both put the deception contrast at roughly -0.6 to
-0.73 against the assistant axis, under two different judges. If paper16 -- built from
Mahadik & Skapars' own personas, which are institutional on both sides -- lands far
below that, the contamination was the persona set. If it lands near it, deception and
un-Assistant-ness are substantially the same direction in Llama-3.3-70B, which is a
finding about the model rather than the method.

    python src/assistant_axis_ndif/pc_gate.py run-paper16/axes_vectors_all.pt
    python src/assistant_axis_ndif/pc_gate.py            # every run-paper16/axes_*.pt
"""

from __future__ import annotations

import glob
import sys

import torch
import torch.nn.functional as F

# measured 2026-09-14; see docs and the plan's diagnostics table
REFERENCE = {
    20: (-0.729, -0.686),
    22: (-0.710, -0.662),
    40: (-0.660, -0.610),
    60: (-0.573, -0.516),
}


def report(path: str) -> None:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    axes = blob.get("axes", {})
    missing = [k for k in ("deception", "assistant") if k not in axes]
    if missing:
        print(f"\n{path}: missing {', '.join(missing)} -- cannot run the gate")
        return

    print(f"\n{path}")
    print(f"{'layer':>6}{'paper16':>10}{'own100':>10}{'hf275':>9}   reading")
    for layer, (own, hf) in REFERENCE.items():
        if layer >= axes["deception"].shape[0]:
            continue
        c = F.cosine_similarity(axes["deception"][layer], axes["assistant"][layer], dim=0).item()
        # judged against own100, the bank paper16 is meant to replace
        if abs(c) < abs(own) - 0.25:
            note = "much cleaner -- persona set was the problem"
        elif abs(c) < abs(own) - 0.10:
            note = "cleaner"
        elif abs(c) > abs(own) + 0.10:
            note = "worse"
        else:
            note = "unchanged -- looks like a property of the model"
        print(f"{layer:>6}{c:>10.3f}{own:>10.3f}{hf:>9.3f}   {note}")

    if "sycophancy" in axes:
        s = F.cosine_similarity(axes["sycophancy"][40], axes["assistant"][40], dim=0).item()
        d = F.cosine_similarity(axes["deception"][40], axes["sycophancy"][40], dim=0).item()
        print(f"  layer 40: cos(sycophancy, assistant) = {s:+.3f}   "
              f"cos(deception, sycophancy) = {d:+.3f}")


def main() -> None:
    paths = sys.argv[1:] or sorted(glob.glob("run-paper16/axes_*.pt"))
    if not paths:
        print("no axes files found; run 6_axes.py first")
        raise SystemExit(1)
    for p in paths:
        report(p)


if __name__ == "__main__":
    main()
