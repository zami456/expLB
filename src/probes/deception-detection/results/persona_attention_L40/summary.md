# Persona-anchored attention pooling at layer 40

Written 2026-09-28 from the CSVs of the 2026-09-27 run. Script:
`deception_detection/scripts/persona_attention.py` (`run` for the main folder, `multi` for `multi/`).

Every earlier persona result reads one axis v over an answer by a fixed rule (mean, max, top 10%).
Here the pooling is learned attention, but the readout stays v:

    logits_t = v . x_t / sqrt(d) + w . x_t        a = softmax over the answer's tokens
    score    = sum_t a_t (v . x_t)

So the probe can choose tokens but cannot build a new feature. The query correction w is fitted
with a ridge penalty in the training set's top 512 within-answer PCs. The penalty is chosen on a
held-out fifth of the training set.

Protocol: the transfer matrix. Train on one of CG, HP-KR, IT, ID, RepE (HP-C answers are one token,
so attention cannot act there), test on all six, and average the 25 off-diagonal cells. Fixed reads
need no training, and their off-diagonal mean averages the same cells.

## Learned pooling does not beat the max read

Mean off-diagonal AUROC:

| model | extended (12v13) | knowledge-denial (4v6) |
|---|---|---|
| mean read | 0.908 | 0.905 |
| identity start (w = 0) | 0.908 | 0.906 |
| max read | 0.916 | 0.927 |
| learned query, penalty selected | 0.908 | 0.910 |
| raw 8192-d LR probe, same pairs | 0.727 | 0.727 |

The identity start equals the mean read, as expected at these activation scales. Weaker penalties
do worse (0.896 and 0.882 at 1e-4), so the learned query overfits the training set rather than
finding a transferable token choice.

Per test set, averaged over the training sets that exclude it:

| axis | model | CG | HP-C | HP-KR | ID | IT | RepE |
|---|---|---|---|---|---|---|---|
| extended | mean | 0.912 | 0.712 | 0.912 | 0.978 | 0.980 | 1.000 |
| extended | max | 0.993 | 0.711 | 0.900 | 0.969 | 0.975 | 1.000 |
| extended | learned | 0.920 | 0.712 | 0.905 | 0.978 | 0.979 | 1.000 |
| knowledge-denial | mean | 0.902 | 0.755 | 0.919 | 0.980 | 0.913 | 1.000 |
| knowledge-denial | max | 0.960 | 0.755 | 0.954 | 0.969 | 0.971 | 0.999 |
| knowledge-denial | learned | 0.923 | 0.755 | 0.919 | 0.978 | 0.920 | 1.000 |

The learned query recovers part of CG's gap to the max read, and nothing else.

## Several fixed persona queries (`multi/`)

A second design attends with several heads, each querying with a fixed persona direction, and
reads v in every head. M1 mixes the heads uniformly (zero-shot), M2 learns the mix, M3 also learns
a key map. Q4 uses the four bank axes as queries, Q36 the 36 centred personas.

| model | extended | knowledge-denial |
|---|---|---|
| max read | 0.916 | 0.927 |
| one head querying with v itself, c = 1 | 0.918 | 0.927 |
| M1 Q4, c = 1 | 0.917 | 0.921 |
| M2 Q4 | 0.918 | 0.925 |
| M2 Q36 | 0.917 | 0.913 |
| M3 Q4 | 0.911 | 0.911 |
| M3 Q36 | 0.903 | 0.900 |

None beats the max read by more than 0.002. Paired bootstrap intervals show significant losses
against the max read for every M1, M2 and M3 variant on CG (extended axis) or on HP-KR and IT
(knowledge-denial axis), and no significant gain anywhere. More learned parameters transfer worse.

## HP-KR operating point

Threshold at 1% false positives on Alpaca. Knowledge-denial axis:

| model | AUROC | lies flagged | honest denials flagged | honest claims flagged |
|---|---|---|---|---|
| mean read | 0.919 | 0.885 | 0.948 | 0.001 |
| max read | 0.954 | 0.000 | 0.000 | 0.000 |
| learned query, CG-trained | 0.917 | 0.820 | 0.904 | 0.000 |
| learned query, IT-trained | 0.932 | 0.453 | 0.378 | 0.000 |
| learned query, ID-trained | 0.911 | 0.612 | 0.748 | 0.000 |

The max read ranks HP-KR better than the mean but flags nothing at the Alpaca threshold. Alpaca's
answers are long (median 217 tokens against 5 for HP-KR), which likely raises the max's threshold;
this was not tested. Every learned query flags fewer HP-KR lies than
the mean read. Under every model that flags lies, honest denials are flagged at about the same rate,
which is the wording confound already listed as a limitation. The extended axis flags no HP-KR
answer under any model.

## Conclusion

At layer 40, learned token selection on a fixed persona readout adds nothing over the max read for
ranking, and loses the mean read's HP-KR operating point. The follow-up
(`persona_attnprobe_L*`) lets the probe combine persona coordinates instead of reading one axis.
