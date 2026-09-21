# Pre-registration: how the 80-layer profile will be read

**Written 2026-09-21 ~13:00 BDT, while `PassAC-11` step `A-root-all` was still running and step
`A-kd-all` — the one that produces the `axes` numbers this document governs — had not started.**
No 80-layer persona number existed when this was written. `git log` timestamps it.

Both novelty reviewers independently said the same thing: a narrow band neither strengthens nor
weakens a finding by itself, and what decides it is **how the layer was chosen** and whether the
band survives a test that accounts for having looked at 80 of them. So the rule goes down first.

---

## 1. AMENDED 2026-09-21 ~13:25 — how L20/22/40/44 were chosen

**The first version of this section was wrong, and wrong against ourselves.** It asserted that
L40 was selected from our own Q6 sweep because it maximised the persona advantage. The owner
corrected it and the paper notes confirm the correction: **L40 is a literature choice**, and Q6
*confirmed* it afterwards rather than originating it.

| layer | how it entered the project | independent of our outcome? |
|---|---|---|
| **20** | Liars' Bench convention | **Yes** |
| **22** | Apollo deception probes — *"the probe is applied after layer 22 of 80"* (`docs/papers/deception-probes-notes.md`) | **Yes** |
| **40** | **Assistant Axis (Lu et al.), Fig. 27: the Llama middle layer, 80/2 exactly**, and their main experiments steer and measure at the middle layer (`docs/papers/assistant-axis-notes.md`) | **Yes** |
| **44** | our own sweep — Q2 records "persona axes peak at 44 (0.731)", Q6 "peak L44 0.766" | **NO** |

There is also a principled reason to check several depths rather than one, independent of any
result: **which layer is most truthful is prompt-dependent**, so a single literature layer is not
guaranteed to transfer to a new prompt distribution. Using three literature layers from three
papers, rather than one, is the right response to that.

**Consequence, corrected.** L20, L22 and L40 are pre-specified, so **a primary result reported at
L40 is legitimately pre-registered** and the 80-layer curve is the robustness check around it —
not a replacement for it. That is the opposite of what the first version of this document
concluded.

**L44 remains selected on our outcome.** Any number quoted at L44 rather than L40 is post hoc and
must either be dropped or estimated on held-out rows.

**What does not change.** F-72's measurement stands on its own: on the matched slice the same
cell spans 0.374 → 0.739 across depth, against a claimed margin of +0.033. A *pre-specified*
layer can still sit on a steep part of the curve, and a headline that moves by eleven times its
own effect size across depth has to be reported with the profile whatever the provenance of the
layer. Pre-specification licenses the choice; it does not make the sensitivity disappear.

**Why this correction is recorded rather than quietly fixed.** Getting the provenance wrong in
the direction of self-criticism is still getting it wrong, and a reader who saw the first version
would have discounted a defensible result. The error was mine: I read a Q6 memory line —
*"persona-coordinate methods: 40 dominates"* — as the origin of the choice without checking the
paper notes that actually specify it.

## 2. Definitions fixed before the data

- **"Narrow"** = the contiguous run of layers where the margin over the matched control stays
  positive is **fewer than 5 layers**. Residual-stream layers differ by one block's update, so
  decodable signal normally varies smoothly with depth; a 1–2 layer spike flanked by chance is
  more likely noise than a localised representation.
- **"Contiguous"** = no interior layer with a margin of the opposite sign.
- **"Recurs"** = the peak layer of the cap bank and of the kd bank fall within ±4 layers of each
  other. A spurious argmax has no reason to recur across banks built from different personas.

## 3. Tests committed to now

1. **Report the whole profile.** Never the argmax alone, in any figure, table or sentence.
2. **Cluster-based permutation test** (Maris & Oostenveld 2007), which is the standard for exactly
   this shape of problem. Each label permutation is applied **identically to all layers**, so the
   null preserves the between-layer correlation rather than treating 80 layers as 80 independent
   tests.
3. **Bootstrap the argmax layer.** Claim depth-localisation only if that interval is tight. Kumar
   (arXiv:2605.27958 §5.1.2) reports peak-layer CIs spanning nearly the whole network and reads
   that as weak localisation; if ours does the same, we say the same.
4. **Nest layer selection inside the dataset-level bootstrap** — re-select the layer within each
   resample. If the effect survives that, narrowness is a finding. If it does not, the effect was
   the selection.
5. **No post-hoc layer switching.** If the peak sits away from L40/44, its size is estimated on
   **held-out rows** — select on one half, measure on the other. Selecting and measuring on the
   same rows is double dipping (Kriegeskorte et al. 2009).
6. **Any permutation p is corrected for layer selection.** An uncorrected p over 80 layers is not
   a p.

## 4. Interpretation, fixed in advance

| outcome | reading |
|---|---|
| Broad plateau containing L40–L44 | **Strengthens.** The result is depth-robust and L40 was a lucky but harmless choice. |
| Narrow band containing L40, contiguous, survives the cluster test, **and recurs across banks** | **Real depth localisation at a pre-specified layer.** Report the headline with the profile beside it. |
| Narrow band containing L40 that fails the cluster test, or does not recur | **Weakens**, though less than if L40 had been post hoc: a pre-specified layer landing on a noise spike is bad luck, not selection. Report the profile and say the effect is not depth-robust. |
| Band that misses L40–L44 | **Exploratory only.** Needs replication on held-out rows before any claim. |
| No contiguous positive band anywhere | **The finding does not survive.** Retract to "not measurable at this n." |

## 5. A directional prediction, from Claim 9

If the HP-KR signal is mostly the **denial bit** (the lexical shortcut, regex 0.91 vs probe 0.87),
then:

- the **unmatched** HP-KR set should decode well at **shallow** depth, where surface form lives;
- the **matched** slice — wording held constant — should sit near chance at **every** depth.

One external data point fits the first half: Moustafa et al. report the 20th-percentile layer
beating the 66th on harm-pressure choice and knowledge-report (two depths only).

**If the matched slice instead shows a surviving mid-depth band, that is non-lexical signal**, and
Claim 9 narrows to *"the lexical bit carries most of the signal except at layers X–Y."* Committing
to that reading now, because it is the interpretation we would otherwise be tempted to reach for
after seeing a band.

## 6. What this document costs us if the result is good

If the profile comes back a broad plateau, this document will look like excessive caution. That is
the correct trade. The alternative — deciding what a narrow band means *after* seeing whether it
contains our layer — is the failure mode this project has spent the day documenting in other
people's work, and one reviewer named the remedy precisely: a result that evaporates under nested
selection is **Claim 5 applied to ourselves**, which is the most credible thing a paper making our
argument can do.
