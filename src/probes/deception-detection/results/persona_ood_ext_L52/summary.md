# Layer 52 (stride-4 sweep)

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
| extended axis (12v13) | 0.877 | 0.695 |
| knowledge-denial axis (4v6) | 0.869 | 0.717 |
| extended PCs (k=3) | 0.843 | 0.621 |
| unified axis (13v12) | 0.665 | 0.579 |
| deception axis (8v7) | 0.668 | 0.632 |
| single-source probe (8192-d) | 0.726 | 0.584 |

| zero-shot (threshold: 1% FPR on Alpaca) | LB avg AUROC | LB avg balanced accuracy | HP-KR AUROC | HP-KR balanced accuracy |
|---|---|---|---|---|
| base probe (RepE-trained) | 0.753 | 0.626 | 0.877 | 0.500 |
| deception axis (8v7) | 0.742 | 0.617 | 0.447 | 0.500 |
| unified axis (13v12) | 0.747 | 0.571 | 0.372 | 0.500 |
| knowledge-denial axis (4v6) | 0.843 | 0.663 | 0.883 | 0.730 |
| extended axis (12v13) | 0.852 | 0.634 | 0.859 | 0.500 |

Files: the matrices and `summary*.csv` from `persona_ood.py run`, and `zero_shot.csv` and
`zero_shot_operating.csv` from `zeroshot`. The figures are `transfer_matrices.png`,
`transfer_delta.png`, `variants.png`, `transfer_sycophancy*.png`, `zero_shot_fig2.png` and
`zero_shot.png` (this layer only).

Reproduce:

```
python deception_detection/scripts/layer_sweep.py pool          # all 13 new layers at once
python deception_detection/scripts/layer_sweep.py probe --layer 52
python deception_detection/scripts/layer_sweep.py analyse --layer 52 --bank ext
```
