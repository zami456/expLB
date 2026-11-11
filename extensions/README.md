# Extensions

These are finished follow-up studies that use the paper's detectors but are not part of the paper. Each one was pre-registered, frozen before its test reads, and reported with its own verdict.

| Folder | Question | Report |
|---|---|---|
| [`narcbench/`](narcbench/) | Do white-box collusion probes (Rose et al., NARCBench) read collusion or the collusion instruction? Includes the Qwen3-32B pipeline runs and the Verdict study on Shi et al.'s emergent-collusion runs. | [`RESULTS.md`](narcbench/RESULTS.md), [`VERDICT.md`](narcbench/VERDICT.md), `QWEN_D4_*.md` |
| [`emotion_coords/`](emotion_coords/) | Are emotion coordinates useful lie-probe features on Liars' Bench? | [`PREREGISTRATION.md`](emotion_coords/PREREGISTRATION.md), [`RESULTS.md`](emotion_coords/RESULTS.md) |

Run them as modules from the repository root, for example `python -m extensions.emotion_coords.report_tables`. Their large reads are not distributed; the scripts' path constants name the stores they were run from.
