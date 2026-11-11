# arXiv submission notes

Source folder: `paper/arxiv/`. Built with Tectonic locally; written to
compile under pdflatex on arXiv (TeX Live 2025 default, 2023 selectable). ICLR 2027 style, running header
"Preprint"; main text ends on page 9 (ICLR's limit), statements on page 10, references, then appendices A to L.

## Build

From the repository root:

```
python paper/arxiv/make_qwen_roleplay.py     # only when the judge scores on E: change; writes paper/arxiv/data/qwen_roleplay.csv
python paper/arxiv/make_arxiv_assets.py      # numbers, tables, figures (add --no-plots to skip the plots)
cd paper/arxiv
tectonic main.tex --keep-logs --keep-intermediates   # main.pdf, main.bbl
python check_keys.py --style                         # unknown \V keys, design labels, dashes, spellings
python make_bundle.py                                # arxiv_bundle.zip
```

The anonymous (double-blind) variant: `python anon_copy.py && tectonic main_anon.tex`. It hides the author
block, the acknowledgements and the repository URL, and the style prints its own "Under review" header.

The generator stops the build when a result it asserts changes. One guard matters now: the Qwen paragraph
of `sections/results_models.tex` is written for the thesis's Qwen numbers of 2026-10-15 (the first,
system-message bank). If the Qwen user_stay re-run regenerates `paper/thesis/generated_xmodel/numbers.tex`
with that bank as primary, the generator raises; rewrite the Qwen paragraph, the claims table caption (the
dagger) and the abstract's Qwen sentence, then update the guard in `extra_keys`.

## Metadata (to paste into the arXiv form)

- Title: The Geometry of Lies: Knowledge and Persona Directions for Detecting Deception in Large Language Models
- Authors (report order): Md. Abdullah Al Zami, Suraiya Shanaz, Mahdi Hasan, Shoaib Ahmed Dipu, Farig Yousuf
  Sadeque, Md. Sabbir Ahmed (BRAC University)
- Abstract: `sections/abstract.tex` (one paragraph; remove the LaTeX line breaks)
- Primary category: cs.CL (alternative: cs.LG); cross-lists: cs.LG, cs.AI
- Comments line (fill the counts from the final build): "28 pages. Code and result files at
  https://github.com/zami456/expLB"
- Licence: CC BY 4.0 (the choice is irrevocable)
- Processor: pdflatex, TeX Live 2025 (if cleveref fails there, select TeX Live 2023)

## Rules checked on 2026-10-18 (arxiv.org help pages)

- Source is required for a TeX-produced paper; a PDF alone is "typically rejected".
- A `.bib` is processed since late 2025; `main.bbl` is included as well and must keep the main file's name.
- Supported processors: pdflatex, xelatex (added November 2025); TeX Live 2025 default.
- Figures as PDF or PNG; relative paths only; no spaces in file names; no hidden files; no `.aux`, `.log`, `.pdf`.
- The submitter must view the compiled PDF before finishing; endorsement is per category for a first
  submission, faster with an institutional email (read the January 2026 endorsement post first).

## Before submitting (decisions for the authors)

1. **Steering labels.** The disclosure labels of the steering experiment (27 scenarios x 11 answers) were made
   by the AI coding assistant, reading blind against a hidden key (`results/steering_L40_notebook/summary.md`,
   "There was one reader, me"). The paper now says so (method, results, limitations, AI use statement).
   Option: have a human re-label `blind_reading_sheet.md` blind and replace the labels.
2. **AI use statement** (`sections/statements.tex`): confirm the wording. Open items, listed in a LaTeX
   comment there: who wrote the knowledge direction's 120 questions (`recall_items.py`,
   `recall_items_matched.py`) and who hand-read the two knowledge-direction steering runs; if the AI
   assistant did either, add it to the statement.
3. **Qwen.** The paper reports Qwen with the thesis's first bank (28% of rollouts fully in role) and adds the
   user_stay re-run (69%, from the judge scores via `make_qwen_roleplay.py`): the persona contrast still
   fails there, and on both banks the failure comes from HP-KR. When the re-run finishes and the thesis is
   regenerated, decide which bank is Qwen's primary in the paper; the generator's guard will stop the build
   until the Qwen paragraph is rewritten.
4. **Steering source.** The scenarios' source paper is not cited (the thesis named no source); add it if wanted.
5. **Qwen3.5-27B question-bank leg.** One sentence says it is still running; add its result when it lands.
6. Re-run the generator, the key check and the bundle; upload `arxiv_bundle.zip`.
