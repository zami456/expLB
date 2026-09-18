# Deceptive personas — layer 20 (shared-question run)

The 8 deceptive personas from our persona-coordinates replication, sliced at **layer 20**,
for dotting against Liars' Bench activations.

Source: `run-paper16-shared` — the run where **all 26 personas answered the same balanced 20
questions** (10 deception + the first 10 sycophancy), per the paper's Appendix C. This is the
right run to use: in the other run each group answered its own question set, which makes the
unified axis track *which questions were asked* rather than the behaviour.

## Files

- `deceptive_L20.pt` — torch: `torch.load(path, weights_only=False)` → dict
- `deceptive_L20.npz` — numpy: `np.load(path)`, no torch needed
- `deceptive_L20.json` — metadata alone

| Key | Shape | What |
|---|---|---|
| `personas` | `(8, 8192)` | one mean-pooled vector per deceptive persona |
| `deception_axis` | `(8192,)` | the deception direction at layer 20 |
| `persona_names` | 8 names | **row order of `personas`** |
| `persona_n_rollouts` | 8 ints | rollouts behind each vector |

## Provenance

- **Model: `meta-llama/Llama-3.3-70B-Instruct`**, layer 20 of 80, hidden 8192.
- Each vector = mean residual-stream activation over the **assistant response tokens**,
  averaged over that persona's kept rollouts (judge score 3).
- The 8 personas: handler, broker, impression_manager, coverup_operator, gatekeeper,
  spin_doctor, plausible_denier, narrative_manager.

## Three things to know before you interpret a number

1. **The axis is not a unit vector** — `|deception_axis| = 0.890`. Divide by it, or use
   cosine, before comparing magnitudes against anything else.

2. **`deception_axis` is `mean(deception) − mean(honesty)`.** The 7 honesty personas are not
   in this folder but they define the direction. If you want the honest end too, ask — it is
   the same one-line cut.

3. **Raw projections are uncentered, so their absolute sign means nothing.** Projecting these
   8 onto the normalised axis gives:

   ```
   +0.372  pc_coverup_operator     n=26
   +0.247  pc_plausible_denier     n=26
   +0.122  pc_spin_doctor          n=56
   -0.190  pc_impression_manager   n=49
   -0.198  pc_broker               n=72
   -0.490  pc_handler              n=64
   -0.596  pc_narrative_manager    n=49
   -0.661  pc_gatekeeper           n=82
   ```

   Five of the eight are negative. That is not a contradiction: persona vectors share a large
   common component whose projection shifts everything by a constant. **Use the ordering, or
   centre first** — subtract the bank mean, or the `default` persona vector, which is in the
   full bank at `../paper16_L20/shared_questions/`. Against the full 26-persona bank these
   same personas sit at the deceptive end, as expected.

## Sample size varies a lot

`n` ranges from **26** (coverup_operator, plausible_denier) to **82** (gatekeeper), out of 100
rollouts per persona (5 prompt variants × 20 questions). The difference is judge filtering: a
persona the model played less convincingly keeps fewer rollouts. The two personas at the top of
the projection list are also the two with the fewest rollouts, so treat their vectors as the
noisiest in the set.
