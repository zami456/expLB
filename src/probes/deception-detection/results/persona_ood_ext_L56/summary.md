# Layer 56 (stride-4 sweep)

One layer of the persona study's depth sweep; `../persona_depth/summary.md` has every layer side
by side. The analysis matches `persona_ood_ext_L20/`, `_L40/` and `_L44/`: Llama-3.3-70B-Instruct,
six Liars' Bench datasets, 36 personas from `persona including knowlede denial/`. Two inputs were
made differently, and both were checked against layer 40 (`layer_sweep.py check40`):

- **Pooled activations were computed on NDIF** (`layer_sweep.py pool`): each example's mean over
  its detected tokens, rather than pooled from a per-token cache.
- **The RepE base probe was trained by the runner** on RepE's statement tokens only
  (`acts_positions: tail`) and never scored by it. It joins `zero_shot_operating.csv` without the
  check that it reproduces its run's own scores.

| transfer, off-diagonal mean | AUROC | balanced accuracy @1% FPR |
|---|---|---|
| extended axis (12v13) | 0.859 | 0.666 |
| knowledge-denial axis (4v6) | 0.852 | 0.715 |
| extended PCs (k=3) | 0.814 | 0.613 |
| unified axis (13v12) | 0.545 | 0.556 |
| deception axis (8v7) | 0.547 | 0.585 |
| single-source probe (8192-d) | 0.730 | 0.575 |

| zero-shot (threshold: 1% FPR on Alpaca) | LB avg AUROC | LB avg balanced accuracy | HP-KR AUROC | HP-KR balanced accuracy |
|---|---|---|---|---|
| base probe (RepE-trained) | 0.752 | 0.627 | 0.916 | 0.500 |
| deception axis (8v7) | 0.720 | 0.588 | 0.434 | 0.503 |
| unified axis (13v12) | 0.725 | 0.557 | 0.377 | 0.500 |
| knowledge-denial axis (4v6) | 0.823 | 0.661 | 0.869 | 0.746 |
| extended axis (12v13) | 0.831 | 0.601 | 0.820 | 0.500 |

Files: the matrices and `summary*.csv` from `persona_ood.py run`, and `zero_shot.csv` and
`zero_shot_operating.csv` from `zeroshot`. The figures are `transfer_matrices.png`,
`transfer_delta.png`, `variants.png`, `transfer_sycophancy*.png`, `zero_shot_fig2.png` and
`zero_shot.png` (this layer only).

Reproduce:

```
python deception_detection/scripts/layer_sweep.py pool          # all 13 new layers at once
python deception_detection/scripts/layer_sweep.py probe --layer 56
python deception_detection/scripts/layer_sweep.py analyse --layer 56 --bank ext
```
