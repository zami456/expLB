# Paper sources

| Folder | Document | Generator |
|---|---|---|
| [`arxiv/`](arxiv/) | The arXiv paper | `make_arxiv_assets.py` |
| [`thesis/`](thesis/) | The thesis (the arXiv paper's numbers are the thesis's own `\V{key}`s) | `make_thesis_assets.py`, `make_xmodel_assets.py` (§5.6, App. D) |
| [`ieee/`](ieee/) | The IEEE short version | `make_ieee_assets.py` |
| [`draft/`](draft/) | The early draft, whose generated tables the thesis copies | `make_numbers.py`, `make_figures.py` |

Each generator reads `results/` and writes its document's `generated/` and `images/`. It also asserts the claims its text makes, so a changed result file stops the build instead of leaving a stale sentence. Run them from the repository root in dependency order:

```bash
python paper/draft/make_numbers.py && python paper/draft/make_figures.py
python paper/thesis/make_thesis_assets.py
python paper/thesis/make_xmodel_assets.py     # needs the cross-model reads (XMODEL_ROOT)
python paper/ieee/make_ieee_assets.py
python paper/arxiv/make_arxiv_assets.py       # also reads the thesis's generated files
cd paper/arxiv && tectonic main.tex
```

The generators read Liars' Bench from the Hugging Face cache, and with a warm cache `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1` keeps them offline. They also read the benchmark's own probe baselines from `third_party/liars-bench/results/probes/`.

**Status (2026-11-11).** `make_xmodel_assets.py` stops on 8 inputs of the in-progress Qwen user-turn re-run (`s3u`/`s23u` banks) that are not produced yet. The committed `thesis/generated_xmodel/` tables are the ones the papers use. [`arxiv/ARXIV_SUBMISSION.md`](arxiv/ARXIV_SUBMISSION.md) has the arXiv build and upload notes.
