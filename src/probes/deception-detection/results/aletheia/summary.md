# The thesis's persona detector on the Aletheia's Quest test sets

Leaderboard: https://aletheias-quest.github.io/winners.html (30 entries; each entry's 10-dataset mean reproduced to 1e-4).
Protocol: `deception_detection/scripts/aletheia/spec.py`, fixed before any test row was read.

## Headline

D9 + K (the thesis's selected design), read through each organism at the middle block, reasoning dropped: mean AUROC **0.657** [95% 0.633, 0.680] over 8 datasets (Echo, Harm, GS-F, GS-M, BotC, AB-AW, AB-HC, AB-SP).
Rank among the 30 entries over the same datasets: **16** (rank interval 13-20).
Not in the headline (no organism read yet): HG-N, HG-G.

## Leaderboard over 8 datasets

| # | Entry | Track | Mean AUROC |
|---|---|---|---|
| 1 | GMU-Veritaserum | white-box | 0.8680 |
| 2 | FlaiChi | white-box | 0.8326 |
| 3 | yy | white-box | 0.7861 |
| 4 | pinocchio | white-box | 0.7860 |
| 5 | dami | white-box | 0.7684 |
| 6 | SAIN Groningen | white-box | 0.7581 |
| 7 | FlaiChi | black-box | 0.7554 |
| 8 | SAIN Groningen | black-box | 0.7416 |
| 9 | Recall Lab | white-box | 0.7131 |
| 10 | Snake | white-box | 0.7122 |
| 11 | TS | white-box | 0.6954 |
| 12 | EleutherAI | white-box | 0.6820 |
| 13 | Tony | white-box | 0.6698 |
| 14 | mpi | black-box | 0.6665 |
| 15 | aa | white-box | 0.6612 |
| 16 | **ours: D9 + K (headline)** | white-box (zero-shot) | 0.6571 |
| 17 | EleutherAI | black-box | 0.6461 |
| 18 | vk | white-box | 0.6457 |
| 19 | Snake | black-box | 0.6413 |
| 20 | Simit | white-box | 0.6405 |
| 21 | Simit | black-box | 0.6210 |
| 22 | Recall Lab | black-box | 0.6205 |
| 23 | JSR | black-box | 0.6185 |
| 24 | ru | white-box | 0.6093 |
| 25 | Tony | black-box | 0.6049 |
| 26 | ru | black-box | 0.5980 |
| 27 | yy | black-box | 0.5494 |
| 28 | TaraResearch | white-box | 0.5484 |
| 29 | mpi | white-box | 0.5373 |
| 30 | JSR | white-box | 0.5271 |
| 31 | TaraResearch | black-box | 0.5213 |

## Per dataset

| Dataset | n (lies) | D9 + K [95%] | D9 | random 97.5% | GMU-Veritaserum | FlaiChi | yy | best white-box |
|---|---|---|---|---|---|---|---|---|
| HG-N | 200 (152) | not read | n/a | n/a | 0.9344 | 0.0000 | 0.0293 | 0.9344 |
| HG-G | 189 (139) | not read | n/a | n/a | 0.6327 | 0.0496 | 0.3881 | 0.8201 |
| Echo† | 400 (200) | 0.189 [0.149, 0.232] | 0.604 | 0.778 | 0.7152 | 0.5211 | 0.5427 | 0.8652 |
| Harm | 200 (100) | 0.885 [0.835, 0.929] | 0.900 | 0.800 | 0.9767 | 0.9443 | 0.8597 | 0.9767 |
| GS-F | 154 (74) | 0.771 [0.699, 0.841] | 0.774 | 0.852 | 0.9916 | 0.9914 | 0.9485 | 0.9916 |
| GS-M | 155 (75) | 0.798 [0.723, 0.866] | 0.714 | 0.813 | 0.9603 | 0.9863 | 0.9133 | 0.9863 |
| BotC | 400 (200) | 0.604 [0.548, 0.659] | 0.490 | 0.625 | 0.5734 | 0.5622 | 0.4722 | 0.6103 |
| AB-AW | 172 (26) | 0.786 [0.712, 0.849] | 0.616 | 0.727 | 0.8859 | 0.8488 | 0.8796 | 0.8859 |
| AB-HC | 176 (30) | 0.408 [0.311, 0.503] | 0.683 | 0.859 | 0.9454 | 0.8607 | 0.7884 | 0.9454 |
| AB-SP | 182 (34) | 0.818 [0.758, 0.873] | 0.856 | 0.817 | 0.8953 | 0.9461 | 0.8841 | 0.9461 |

† Echo: the pre-registered dev-row check failed (organism_Echo_dev_check.json): the dev rows tagged with this adapter are more likely under the base model; accepted on the organism's own test replies instead (text only, no labels), by the user's decision.

## Every pre-registered cell

Means over the headline's 8 datasets where the cell covers all of them; otherwise over the datasets it covers (listed).

| Cell | Role | Detector | Read | Render | Span | Block | Mean | Rank it would take | Datasets |
|---|---|---|---|---|---|---|---|---|---|
| headline | headline | D9+K | organism | no-reasoning | content | mid | 0.657 | 16 of 31 | 8 |
| S1-d9 | secondary | D9 | organism | no-reasoning | content | mid | 0.705 | 11 of 31 | 8 |
| S2-reasoning | secondary | D9+K | organism | reasoning | content | mid | 0.758 | 7 of 31 | 8 |
| S2-reasoning-d9 | secondary | D9 | organism | reasoning | content | mid | 0.799 | 3 of 31 | 8 |
| S3-base | secondary | D9+K | base | no-reasoning | content | mid | 0.644 | 18 of 31 | 8 |
| S3-base-d9 | secondary | D9 | base | no-reasoning | content | mid | 0.668 | 14 of 31 | 8 |
| S4-inside | secondary | D9+K | organism | no-reasoning | content | mid | 0.666 | 15 of 31 | 8 |
| S5-base-k | secondary | D9+K | organism | no-reasoning | content | mid | 0.677 | 13 of 31 | 8 |
| S6-d5 | secondary | D5 | organism | no-reasoning | content | mid | 0.592 | 26 of 31 | 8 |
| S6-d8 | secondary | D8 | organism | no-reasoning | content | mid | 0.673 | 13 of 31 | 8 |
| S6-k | secondary | K | organism | no-reasoning | content | mid | 0.591 | 26 of 31 | 8 |
| S7-dev-block | secondary | D9+K | organism | no-reasoning | content | dev-selected | 0.657 | 16 of 31 | 8 |
| S8-eot | secondary | D9+K | organism | no-reasoning | content+eot | mid | 0.659 | 16 of 31 | 8 |
| R1-repe | reference | D2 | organism | no-reasoning | content | mid | 0.784 | 5 of 31 | 8 |
| R2-dev-probe | reference | dev-probe | organism | no-reasoning | content | mid | 0.618 | 23 of 31 | 8 |

## The organism's read and the base model's

| Dataset | D9 + K, organism read | D9 + K, base read |
|---|---|---|
| HG-N | not read | not read |
| HG-G | not read | not read |
| Echo | 0.189 | 0.284 |
| Harm | 0.885 | 0.885 |
| GS-F | 0.771 | 0.756 |
| GS-M | 0.798 | 0.736 |
| BotC | 0.604 | 0.559 |
| AB-AW | 0.786 | 0.744 |
| AB-HC | 0.408 | 0.359 |
| AB-SP | 0.818 | 0.827 |

## Post hoc: D5 + K, rule K, D5 + rule K, D9 + rule K, D4

Not in the protocol: asked for after the results above were seen. Read through each organism at the middle block, reasoning dropped, as the headline; rule K (`posthoc.py`, the thesis's `k_final` recipe) is built inside each organism from its own reads; D4 is the thesis's probe in the bank's top 3 extended PCs, trained on the model's dev rows (as R2), never on test labels.

| Dataset | D5 + K | rule K | D5 + rule K | D9 + rule K | D4 |
|---|---|---|---|---|---|
| Echo | 0.206 | 0.222 | 0.231 | 0.214 | 0.647 |
| Harm | 0.696 | 0.766 | 0.871 | 0.876 | 0.515 |
| GS-F | 0.735 | 0.552 | 0.621 | 0.642 | 0.743 |
| GS-M | 0.782 | 0.594 | 0.646 | 0.658 | 0.676 |
| BotC | 0.573 | 0.599 | 0.560 | 0.577 | 0.486 |
| AB-AW | 0.794 | 0.750 | 0.768 | 0.776 | 0.590 |
| AB-HC | 0.311 | 0.250 | 0.305 | 0.389 | 0.697 |
| AB-SP | 0.804 | 0.589 | 0.748 | 0.765 | 0.847 |

Means: D5 + K 0.613 (rank 23 of 31, 8 datasets); rule K 0.540 (rank 28 of 31, 8 datasets); D5 + rule K 0.594 (rank 26 of 31, 8 datasets); D9 + rule K 0.612 (rank 23 of 31, 8 datasets); D4 0.650 (rank 16 of 31, 8 datasets).

## Caveats

- A base-model read of an organism's transcript is the thesis's control, not a result for that dataset: the base model has no fine-tuned belief or goal to conceal, so it reads only the text. A dataset whose organism could not be read (the Hidden-Goal adapters are gated) is reported as not run; its base read appears only beside the organism read, as the control. Harmful-Looking is different: plain Gemma wrote it, so the base model is its organism.
- The top entries are active methods (the organism judging its own transcript, follow-up questions, probes trained on the competition's dev data); this detector is a passive zero-shot read of one direction.
- The starter kit dropped the `reasoning` field and cut rows to their last 256 tokens by default; the headline drops the reasoning as the kit did; rows are cut only by the frozen rule, listed below.
- A July 22 starter-kit commit fixed a silent no-op in which Qwen3.5 adapter keys matched 0 of 512; whether the official white-box reads of the Qwen organisms had their adapters applied is not known, so the base-model read is shown beside the organism read.
- AB-AW has 26 lies: per-dataset intervals there are wide.

- Rows cut by the frozen rule (kept its last 4,096 tokens): base/test__BotC__no-reasoning: 163; lora-botc-qwen3.5-27b@2/test__BotC__no-reasoning: 148.
  - Of them cut untried: base/test__BotC__no-reasoning: 160; lora-botc-qwen3.5-27b@2/test__BotC__no-reasoning: 147. Each is longer than a row of the same read that ran out of memory alone on all four tries with no other read on the cluster, and memory grows with length (the user's decision, a deviation from the frozen rule's wording, which asks each row to fail itself). Every row shorter than that row was read whole.
  - BotC: a first pass through the adapter cut 151 rows after a row failed while other reads shared the cluster; those reads were set aside unscored (`E:\aletheia\acts\superseded\botc_cut_under_contention`) and the rows reread with the cluster idle. The organism and base reads cut at their own thresholds, so their BotC cut sets differ (148 and 163 rows).
- Rows left unread (too long for the server; its cells are not scored): base/test__HG-G__no-reasoning: 1.
Cells not yet scored: 30 (results/aletheia/missing.csv).

