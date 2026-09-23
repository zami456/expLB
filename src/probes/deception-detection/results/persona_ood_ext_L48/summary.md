# Layer 48 (stride-4 sweep)

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
| extended axis (12v13) | 0.888 | 0.714 |
| knowledge-denial axis (4v6) | 0.881 | 0.722 |
| extended PCs (k=3) | 0.873 | 0.618 |
| unified axis (13v12) | 0.702 | 0.591 |
| deception axis (8v7) | 0.686 | 0.653 |
| single-source probe (8192-d) | 0.716 | 0.585 |

| zero-shot (threshold: 1% FPR on Alpaca) | LB avg AUROC | LB avg balanced accuracy | HP-KR AUROC | HP-KR balanced accuracy |
|---|---|---|---|---|
| base probe (RepE-trained) | 0.746 | 0.626 | 0.837 | 0.500 |
| deception axis (8v7) | 0.771 | 0.640 | 0.512 | 0.500 |
| unified axis (13v12) | 0.803 | 0.584 | 0.516 | 0.500 |
| knowledge-denial axis (4v6) | 0.857 | 0.669 | 0.890 | 0.735 |
| extended axis (12v13) | 0.865 | 0.657 | 0.850 | 0.500 |

Files: the matrices and `summary*.csv` from `persona_ood.py run`, and `zero_shot.csv` and
`zero_shot_operating.csv` from `zeroshot`. The figures are `transfer_matrices.png`,
`transfer_delta.png`, `variants.png`, `transfer_sycophancy*.png`, `zero_shot_fig2.png` and
`zero_shot.png` (this layer only).

Reproduce:

```
python deception_detection/scripts/layer_sweep.py pool          # all 13 new layers at once
python deception_detection/scripts/layer_sweep.py probe --layer 48
python deception_detection/scripts/layer_sweep.py analyse --layer 48 --bank ext
```
