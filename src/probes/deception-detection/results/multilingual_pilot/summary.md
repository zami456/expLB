# Multilingual pilot: do the detectors carry over to another language?

An off-thesis pilot (2026-09-30). It uses one dataset (instructed deception), 20
statements per language, Llama-3.3-70B, and layer 40. German came first. Since German is close to
English, Hindi followed. The persona role-play part was dropped for both languages.

## Design

- **Items.** 20 statements from Liars' Bench instructed deception (azaria-mitchell), 10 true and 10
  false. The model knows each in English, and each has a valid honest and a valid lying English
  reply. Those 40 English rows are the benchmark's own.
- **Translation.** The statements and prompts are in `data/multilingual/{de,hi}.json`, translated by
  Claude and not checked by a native speaker.
- **Answers.** Llama answers each translated statement greedily on NDIF, three times: with no system
  prompt (its belief in that language), under the honest prompt, and under the lying prompt.
- **Labels.** A keyword rule labels each answer, then every answer is reviewed by hand
  (`id_labels.csv`, column `final`). An answer is a lie when it contradicts the belief in the same
  language.
- **Reads.** Layer 40, with the reader of `ask_and_detect.py`. The Alpaca re-read check (cosine at
  least 0.9998 with the stored rows) passed for both languages.
- **Detectors.** D8, D9, D5, the RepE base probe, K, and D9 + K.

## German

- The first lying prompt ("give a false answer") shared its verdict word *falsch* with the question.
  The model answered "Das ist falsch" to 18 of 20 statements. The second prompt asks for a
  deliberate lie.
- Kept: 18 honest answers and 17 lies. One item was dropped because the German belief was wrong
  (Alfred Nobel), one because its answers were invalid (krypton), and one lying-prompt answer told
  the truth.
- Every detector scores AUROC 1.000 in German. In English every detector also scores 1.000, except
  K at 0.995.
- Training check: probes were trained on 40 English rows, 40 German rows, or 20 + 20 mixed, and
  scored on the other four datasets. The three probes score the same within noise:

  | Test set | English-trained | German-trained | Mixed |
  |---|---|---|---|
  | HP-C | 0.704 | 0.670 | 0.693 |
  | HP-KR | 0.727 | 0.683 | 0.718 |
  | IT | 0.920 | 0.924 | 0.926 |
  | CG | 1.0 | 1.0 | 1.0 |

## Hindi

- **Belief and honest answers.** Both are right on all 20 items, including Alfred Nobel.
- **Llama seldom lies in Hindi.** Under each of two lying prompts it told only 3 clean lies, each
  time on different items:

  | Lying prompt | Clean lies | Truthful | Contradicts itself |
  |---|---|---|---|
  | v1: lie deliberately (झूठ बोलकर) | 3 | 15 | 2 |
  | v2: say the opposite of what is true | 3 | 14 | 3 |

  The self-contradicting answers call the statement false and then give the true reason. They are
  labelled invalid. For comparison, the second German prompt got 17 lies out of 20. The 20 English
  items were chosen to have valid English lies.
- **Scored rows.** Both prompts' lies are pooled: 6 lies on 6 different items, against 20 honest
  answers. The 29 lying-prompt answers that told the truth are kept as *declined*.

AUROC with 95% bootstrap intervals. The last two columns use the 6 declined answers that are word for
word the same as the honest answer to the same item:

| Detector | Lie v honest, en (20 v 20) | Lie v honest, hi (6 v 20) | Lie v declined, hi (6 v 29) | Declined v honest, hi (29 v 20) | Identical pairs: declined higher | Mean shift / lie gap |
|---|---|---|---|---|---|---|
| D8 knowledge-denial | 1.000 | 1.000 | 1.000 | 0.872 (0.74-0.98) | 6/6 | 0.22 |
| D9 extended | 1.000 | 1.000 | 1.000 | 0.886 (0.77-0.99) | 6/6 | 0.27 |
| D5 published | 1.000 | 1.000 | 0.994 (0.97-1.00) | 0.871 (0.75-0.96) | 6/6 | 0.32 |
| RepE base probe | 1.000 | 1.000 | 0.971 (0.90-1.00) | 1.000 | 6/6 | 0.25 |
| K | 0.995 (0.98-1.00) | 1.000 | 0.966 (0.89-1.00) | 0.726 (0.57-0.86) | 6/6 | 0.25 |
| D9 + K | 1.000 | 1.000 | 0.994 (0.97-1.00) | 0.821 (0.69-0.93) | 6/6 | 0.26 |

The Hindi training check is inconclusive, because the Hindi arm has only 6 lies:

| Probe | CG | HP-C | HP-KR | IT |
|---|---|---|---|---|
| Hindi-trained | 0.995 | 0.771 | 0.647 | 0.508 (0.28-0.87) |
| English-trained | 1.0 | 0.708 | 0.746 | 0.923 |

## What it says

- **Language.** The detectors are built from English only, yet they separate lies from honest
  answers in German and Hindi as well as in English. This dataset puts every detector at ceiling in
  all three languages, though. So it cannot rank the detectors or show a small loss.
- **Instruction.** In Hindi, the model answered truthfully 29 times after being told to lie. On every
  detector those truthful answers score higher than answers given under the honest prompt. The RepE
  probe separates the two perfectly, the persona directions reach 0.87 to 0.89, and K reaches 0.73.
  - On the 6 pairs with word-for-word identical text, all six detectors score the lying-prompt copy
    higher, by about a quarter of the lie gap.
  - So part of an instructed-lie AUROC comes from the instruction, not the lie. The RepE probe is
    the most instruction-sensitive detector here, which fits its training data: the same true fact
    after an "honest person" or a "dishonest person" instruction. K is the least sensitive.
- **The lie still counts for more.** With the lying prompt held fixed, lies score above truthful
  answers on every detector (0.97 to 1.00).

## RepE probe trained on Hindi, at layers 20 and 40

- **Question.** Does training the RepE probe on Hindi as well as English make it generalise better?
- **Data.** RepE's training data is a true fact written after "Pretend you're an honest person..."
  or after the same sentence with "dishonest". 100 of its 224 training facts were drawn at random
  (seed 0), leaving out the two that are also pilot statements. The facts and both prompts were
  translated in `data/multilingual/repe_hi.json`.
- **Reads.** The facts were read at layers 20 and 40 as the stored rows were. Two stored English
  rows were reproduced at each layer (cosine at least 0.99996). The pilot's rows were also read at
  layer 20, and the Alpaca check passed.
- **Probes.** Four were trained on the same 100 facts with `persona_ood.make_probe` on pooled
  means: English, Hindi, mixed (each fact in one language, half and half) and both (every fact in
  both languages). These are supervised probes on RepE's labelled pairs, the Mean Probe's training
  data, fitted on each response's mean rather than on every token. The bootstrap resamples the
  facts, pairs the arms draw by draw, and uses 200 draws.
- **Metrics.** AUROC, plus recall and balanced accuracy at each probe's own 1% false-positive
  threshold on the Alpaca control, which is the study's operating point. At both layers the
  deployed base probe reproduces its stored averages for all three metrics.
- **Code and outputs.** `deception_detection/scripts/multilingual_repe.py` writes
  `repe/score_L20.csv`, `repe/score_L40.csv`, `repe/compare.csv` and `repe/compare_minus_en.csv`.

The mean over the five datasets, with the difference from the English arm in brackets (95%
interval):

| Metric | Layer | English | Hindi | Mixed | Both | Deployed probe |
|---|---|---|---|---|---|---|
| AUROC | 20 | 0.745 | 0.699 [-0.046, -0.06 to -0.02] | 0.711 [-0.034, -0.06 to -0.01] | 0.704 [-0.041, -0.06 to -0.02] | 0.697 |
| AUROC | 40 | 0.766 | 0.800 [+0.034, -0.00 to +0.09] | 0.781 [+0.015, -0.02 to +0.08] | 0.785 [+0.019, -0.02 to +0.08] | 0.778 |
| Balanced accuracy | 20 | 0.648 | 0.665 [+0.017, -0.01 to +0.07] | 0.648 [-0.001, -0.02 to +0.02] | 0.647 [-0.002, -0.02 to +0.01] | 0.721 |
| Balanced accuracy | 40 | 0.642 | 0.657 [+0.015, -0.01 to +0.07] | 0.656 [+0.014, +0.01 to +0.09] | 0.666 [+0.024, +0.01 to +0.07] | 0.626 |
| Recall | 20 | 0.311 | 0.534 [+0.22] | 0.513 [+0.20] | 0.517 [+0.21] | 0.489 |
| Recall | 40 | 0.311 | 0.368 [+0.06] | 0.355 [+0.04] | 0.385 [+0.07] | 0.281 |

AUROC by dataset, layer 20 / layer 40:

| Test | English | Hindi | Mixed | Both |
|---|---|---|---|---|
| ID | 0.959 / 0.978 | 0.950 / 0.969 | 0.949 / 0.969 | 0.948 / 0.967 |
| CG | 0.916 / 0.932 | 0.888 / 0.708 | 0.895 / 0.740 | 0.868 / 0.713 |
| HP-C | 0.529 / 0.644 | 0.472 / 0.777 | 0.460 / 0.771 | 0.462 / 0.752 |
| HP-KR | 0.404 / 0.912 | 0.222 / 0.893 | 0.297 / 0.920 | 0.305 / 0.917 |
| IT | 0.916 / 0.364 | 0.962 / 0.655 | 0.954 / 0.506 | 0.938 / 0.574 |
| Hindi pilot: lie v declined | 0.759 / 0.937 | 0.822 / 0.977 | 0.822 / 0.994 | 0.805 / 0.988 |
| Hindi pilot: declined v honest | 0.886 / 0.993 | 0.931 / 0.984 | 0.903 / 0.993 | 0.900 / 0.984 |

Balanced accuracy and recall by dataset, layer 20 / layer 40:

| Test | English | Hindi | Mixed | Both |
|---|---|---|---|---|
| ID, balanced accuracy | 0.849 / 0.932 | 0.904 / 0.890 | 0.877 / 0.893 | 0.883 / 0.892 |
| ID, recall | 0.764 / 0.998 | 0.966 / 1.000 | 0.865 / 1.000 | 0.883 / 1.000 |
| CG, balanced accuracy | 0.820 / 0.778 | 0.853 / 0.830 | 0.850 / 0.850 | 0.850 / 0.840 |
| CG, recall | 0.641 / 0.557 | 0.707 / 0.701 | 0.701 / 0.701 | 0.701 / 0.701 |
| HP-C, balanced accuracy | 0.500 / 0.500 | 0.500 / 0.517 | 0.500 / 0.519 | 0.500 / 0.517 |
| HP-C, recall | 0.000 / 0.000 | 0.000 / 0.037 | 0.000 / 0.037 | 0.000 / 0.037 |
| HP-KR, balanced accuracy | 0.500 / 0.500 | 0.493 / 0.532 | 0.500 / 0.514 | 0.500 / 0.539 |
| HP-KR, recall | 0.000 / 0.000 | 0.000 / 0.065 | 0.000 / 0.029 | 0.000 / 0.079 |
| IT, balanced accuracy | 0.573 / 0.500 | 0.577 / 0.515 | 0.510 / 0.503 | 0.501 / 0.542 |
| IT, recall | 0.151 / 0.000 | 1.000 / 0.036 | 1.000 / 0.007 | 1.000 / 0.106 |

Every arm scores 1.000 AUROC on lie v honest in the English, German and Hindi pilots at layer 40.

What it says:

- **AUROC: no gain, and a loss at layer 20.**
  - At layer 20, adding Hindi lowers the mean of five by 0.03 to 0.05, and the intervals stay below
    zero. Most of the loss is on HP-KR (-0.10 to -0.18) and HP-C (-0.06).
  - At layer 40 the mean rises by 0.015 to 0.034, and every interval reaches zero. There Hindi
    costs about 0.2 on CG and gains on HP-C and IT.
- **Balanced accuracy barely moves.** The benchmark's headline metric changes by -0.002 to +0.024
  at either layer.
- **The recall gains are a threshold effect.** At layer 20 the probes trained with Hindi catch
  every IT lie, but they also flag 85 to 100% of IT's honest answers, and their balanced accuracy on
  IT is 0.50 to 0.58. Their scores sit higher against the Alpaca threshold. The lies do not
  separate any better.
- **An English threshold does not suit Hindi text once the probe has seen Hindi.** At layer 40:
  - the probes trained with Hindi flag every truthful Hindi answer given under the lying prompt.
    They also flag 50% (Hindi), 80% (mixed) or 95% (both) of the honest Hindi answers;
  - the English-trained probe flags 4 of the 6 Hindi lies and none of the 49 truthful Hindi
    answers.
- **Every arm still reads the instruction.** Declined against honest scores 0.89 to 0.95 at layer
  20 and 0.98 to 1.00 at layer 40 for every arm.
- **Hindi training helps a little inside Hindi.** Hindi lies separate from truthful answers under
  the same lying prompt better by 0.04 to 0.06 at both layers.
- **The layer matters far more than the language.**
  - At layer 20 every arm is inverted on HP-KR (0.22 to 0.40) and good on IT (0.92 to 0.96).
  - At layer 40 every arm is good on HP-KR (0.89 to 0.92) and weak or inverted on IT (0.36 to
    0.66).
  - Adding Hindi does not change this pattern.
- **Caveats.**
  - RepE leaves the last five words of each fact out of the read span, and Hindi puts the verb
    last. So the Hindi span usually stops before the verb, while the English one often includes
    it. "Hindi" therefore means the language plus a different span.
  - Training-sample noise is of the same size as the AUROC effects. English on 100 facts and on all
    224 facts differ by 0.01 to 0.03 in the mean.

## Limits

- One dataset (the easiest), 20 items per language and one model. Only 6 Hindi lies. The pilot's
  detectors were scored at layer 40 only, and the RepE probes at layers 20 and 40.
- The translations are Claude's and have not been checked by a native speaker.
- An item can have up to two declined answers, one per lying prompt, so the bootstrap intervals,
  which treat rows as independent, are too narrow.

## Files

- `de/` and `hi/` hold:
  - `id_generations.json`: the answers under the current prompts;
  - `id_generations_lying_v1.json`: the first lying prompt's answers;
  - `id_labels.csv` and `id_labels_v1.csv`: the hand-reviewed labels;
  - `id_fate.csv`: which answers were kept, and why;
  - `id_rows.csv` and `id_reads_L40.npz`: the scored rows and their reads;
  - `id_score.csv` and `id_training.csv`: the AUROCs and the training check.
- `hi/id_identical.csv` holds the identical-text pairs.
- `de/` and `hi/` also hold `id_reads_L20.npz`, the same rows read at layer 20.
- `repe/` holds:
  - `hi_reads_L20.npz` and `hi_reads_L40.npz`: the Hindi RepE reads;
  - `score_L20.csv` and `score_L40.csv`: every arm, test and metric, with the paired differences;
  - `compare.csv` and `compare_minus_en.csv`: the two layers side by side.
- The code is `deception_detection/scripts/multilingual_pilot.py` and
  `deception_detection/scripts/multilingual_repe.py`. Their docstrings give the commands.
