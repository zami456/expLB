# Pre-registration: how the 80-layer profile will be read

**Written 2026-09-21 ~13:00 BDT, while `PassAC-11` step `A-root-all` was still running and step
`A-kd-all` — the one that produces the `axes` numbers this document governs — had not started.**
No 80-layer persona number existed when this was written. `git log` timestamps it.

Both novelty reviewers independently said the same thing: a narrow band neither strengthens nor
weakens a finding by itself, and what decides it is **how the layer was chosen** and whether the
band survives a test that accounts for having looked at 80 of them. So the rule goes down first.

---

## 1. The honest answer to "how were L20/40/44 chosen?"

This is the question that decides everything else, and our answer is bad for us.

| layer | how it entered the project | independent of the outcome? |
|---|---|---|
| **20** | Liars' Bench convention | **Yes** — literature, fixed before any of our numbers |
| **22** | Apollo convention | **Yes** — literature |
| **40** | **our own Q6 layer sweep, 2026-09-16** | **NO** |
| **44** | same sweep, recorded there as the persona-axes peak (0.766) | **NO** |

The Q6 note says it in as many words: *"Persona-coordinate methods: 40 dominates. Median +0.113
over best-of-{20,22}."* L40 was selected **because it maximised the persona advantage on this
corpus**, and F-03, F-66 and F-68 were then all reported at L40.

That is selection on the reported outcome. It is the exact flank both reviewers named, and
F-72 already shows how large it is: on the matched slice the same cell spans 0.374 → 0.739 across
depth, eleven times the margin F-03 claims.

**Consequence, accepted in advance:** the three-layer summary is **not** the primary analysis. The
**full 80-layer curve is the honest analysis**, and every headline number must be reported as a
point on it, never alone. We do not get to keep L40 as "pre-registered" because it was not.

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
| Narrow band containing L40–L44, contiguous, survives the cluster test, **and recurs across banks** | **Real depth localisation.** Report the headline as explicitly layer-contingent. |
| Narrow band containing L40–L44 that fails the cluster test, or does not recur | **Weakens.** We selected a layer and then reported the selection. Say so. |
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
