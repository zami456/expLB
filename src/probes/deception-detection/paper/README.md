# Lying in Knowledge-Denier Persona Coordinates

LaTeX source of the paper. Every number in the text is generated from the committed results in
`../results/`, and every figure is redrawn from them. Nothing in `sections/` is typed by hand.

Build from `src/probes/deception-detection`:

```
python paper/make_numbers.py    # generated/numbers.tex and generated/tab_*.tex
python paper/make_figures.py    # generated/fig_*.pdf
cd paper && tectonic main.tex   # main.pdf
```

The build needs the project environment and [Tectonic](https://tectonic-typesetting.github.io),
which fetches its LaTeX packages on the first run. Neither step calls a model: the numbers and
figures come from the result CSVs, the runs' `results_table.csv` files and the persona bank.

`make_numbers.py` writes one macro per value, read in the text as `\V{key}`. An unknown key stops
the build rather than printing a blank. The script also re-derives the transfer summaries from the
matrices they summarise and fails on any mismatch.

The author block is a placeholder, and so is the collaborator credit in the acknowledgements.
