# Layer 64 (stride-4 sweep)

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
| extended axis (12v13) | 0.855 | 0.651 |
| knowledge-denial axis (4v6) | 0.825 | 0.713 |
| extended PCs (k=3) | 0.856 | 0.606 |
| unified axis (13v12) | 0.536 | 0.570 |
| deception axis (8v7) | 0.544 | 0.567 |
| single-source probe (8192-d) | 0.718 | 0.575 |

| zero-shot (threshold: 1% FPR on Alpaca) | LB avg AUROC | LB avg balanced accuracy | HP-KR AUROC | HP-KR balanced accuracy |
|---|---|---|---|---|
| base probe (RepE-trained) | 0.759 | 0.626 | 0.924 | 0.500 |
| deception axis (8v7) | 0.708 | 0.549 | 0.416 | 0.486 |
| unified axis (13v12) | 0.704 | 0.558 | 0.374 | 0.500 |
| knowledge-denial axis (4v6) | 0.790 | 0.663 | 0.844 | 0.757 |
| extended axis (12v13) | 0.826 | 0.584 | 0.829 | 0.500 |

Files: the matrices and `summary*.csv` from `persona_ood.py run`, and `zero_shot.csv` and
`zero_shot_operating.csv` from `zeroshot`. The figures are `transfer_matrices.png`,
`transfer_delta.png`, `variants.png`, `transfer_sycophancy*.png`, `zero_shot_fig2.png` and
`zero_shot.png` (this layer only).

Reproduce:

```
python deception_detection/scripts/layer_sweep.py pool          # all 13 new layers at once
python deception_detection/scripts/layer_sweep.py probe --layer 64
python deception_detection/scripts/layer_sweep.py analyse --layer 64 --bank ext
```
