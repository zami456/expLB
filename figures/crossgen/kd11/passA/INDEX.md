# figures/crossgen — index

Every figure is drawn from the JSONs in `run-crossgen/`; none reads the activation store.

| folder | what it shows |
|---|---|
| `figure6_style/` | the paper's Figure 6 layout: raw-probe AUROC, then random subspace, dataset PCA and unified-axis PCs as changes from raw. `banks/<name>/` for each persona bank. |
| `bank_comparison/` | every persona bank's mean change vs raw, per feature, with dataset-bootstrap intervals (CSV beside each) |
| `curves/` | transfer and own-dataset AUROC against layer, shaded band = range over the 5 extra 80/20 splits |
| `matched/` | each persona feature against raw AND against a matched control at the same k (dataset PCA, random subspace) |
| `blocks/` | mean AUROC per construct block (intent/judge/belief, sycophancy row and column) and the inverted-cell counts |
| `per_source/`, `per_target/` | small multiples: each dataset as a training source, and as a test target, against layer |

Files: 20 PNG (each with a PDF twin).
