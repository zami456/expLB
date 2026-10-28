# Phase 0 report: inventory and data checks

Generated 2026-10-29 01:54 by `python -m emotion_coords.phase0 report`. No NDIF calls were made in Phase 0.

## Liars' Bench rows (Llama-3.3-70B), as pooled

| dataset                              | short    |   rows |   pooled_rows |   empty_mask_dropped |   deceptive |   honest |   hub_text_mismatches |   pre_adjacent_share |   eot_after_last_share |   mask_contiguous_share | pooled_L20   | pooled_L28   | pooled_L40   | pooled_L52   |
|:-------------------------------------|:---------|-------:|--------------:|---------------------:|------------:|---------:|----------------------:|---------------------:|-----------------------:|------------------------:|:-------------|:-------------|:-------------|:-------------|
| repe_honesty__plain                  | RepE     |    490 |           448 |                   42 |         224 |      224 |                   nan |                    1 |                      0 |                       1 | ok           | ok           | ok           | ok           |
| repe_honesty__plain_val              | RepE-val |    122 |           102 |                   20 |          51 |       51 |                   nan |                    1 |                      0 |                       1 | ok           | ok           | ok           | ok           |
| convince_me__plain                   | CG       |    267 |           267 |                    0 |         167 |      100 |                     0 |                    1 |                      1 |                       1 | ok           | ok           | ok           | ok           |
| harm_refusal_choice__plain           | HP-C     |   2134 |          2134 |                    0 |         134 |     2000 |                     0 |                    1 |                      1 |                       1 | ok           | ok           | ok           | ok           |
| harm_refusal_knowledge_report__plain | HP-KR    |   2139 |          2139 |                    0 |         139 |     2000 |                     0 |                    1 |                      1 |                       1 | ok           | ok           | ok           | ok           |
| insider_trading_cadenza__plain       | IT       |   1080 |          1080 |                    0 |         423 |      657 |                     0 |                    0 |                      1 |                       1 | ok           | ok           | ok           | ok           |
| instruct_dishonesty__plain           | ID       |   5432 |          5432 |                    0 |        3045 |     2387 |                     0 |                    1 |                      1 |                       1 | ok           | ok           | ok           | ok           |
| XXXX-20__plain                       | Alpaca   |   2000 |          2000 |                    0 |           0 |     2000 |                     0 |                    1 |                      1 |                       1 | ok           | ok           | ok           | ok           |

## Token lengths (whole row, detected span, final assistant message)

| dataset                              | short    | count       |   min |     p5 |    p25 |   median |    p75 |    p95 |   max |
|:-------------------------------------|:---------|:------------|------:|-------:|-------:|---------:|-------:|-------:|------:|
| repe_honesty__plain                  | RepE     | n_tokens    |    57 |   58   |   60   |     62   |   65   |   70   |    76 |
| repe_honesty__plain                  | RepE     | n_resp      |     1 |    2   |    3   |      5   |    7   |   12   |    19 |
| repe_honesty__plain                  | RepE     | n_final_msg |     7 |    8   |   10   |     12   |   15   |   20   |    26 |
| repe_honesty__plain_val              | RepE-val | n_tokens    |    57 |   58   |   60   |     62   |   65   |   74   |    85 |
| repe_honesty__plain_val              | RepE-val | n_resp      |     1 |    2   |    4   |      6   |    7.8 |   17   |    26 |
| repe_honesty__plain_val              | RepE-val | n_final_msg |     7 |    8   |   10   |     12   |   15   |   24   |    35 |
| convince_me__plain                   | CG       | n_tokens    |   177 |  234.9 |  310   |    342   |  361   |  388.7 |   422 |
| convince_me__plain                   | CG       | n_resp      |    95 |  122.2 |  187   |    209   |  229   |  261.4 |   289 |
| convince_me__plain                   | CG       | n_final_msg |    95 |  122.2 |  187   |    209   |  229   |  261.4 |   289 |
| harm_refusal_choice__plain           | HP-C     | n_tokens    |   145 |  157   |  175   |    198   |  218   |  257   |   875 |
| harm_refusal_choice__plain           | HP-C     | n_resp      |     1 |    1   |    1   |      1   |    1   |    1   |     2 |
| harm_refusal_choice__plain           | HP-C     | n_final_msg |     1 |    1   |    1   |      1   |    1   |    1   |     2 |
| harm_refusal_knowledge_report__plain | HP-KR    | n_tokens    |   128 |  154   |  173   |    199   |  288   |  910.7 |  2003 |
| harm_refusal_knowledge_report__plain | HP-KR    | n_resp      |     2 |    2   |    2   |      5   |    8   |   10   |    70 |
| harm_refusal_knowledge_report__plain | HP-KR    | n_final_msg |     2 |    2   |    2   |      5   |    8   |   10   |    70 |
| insider_trading_cadenza__plain       | IT       | n_tokens    |  1325 | 1344   | 1360   |   1380   | 1521.2 | 1578.1 |  1687 |
| insider_trading_cadenza__plain       | IT       | n_resp      |    35 |   57   |   63.8 |     73   |   82   |  109   |   131 |
| insider_trading_cadenza__plain       | IT       | n_final_msg |    67 |   87   |   98   |    108   |  125   |  150   |   208 |
| instruct_dishonesty__plain           | ID       | n_tokens    |    73 |   90   |  107   |    120   |  133   |  152   |   191 |
| instruct_dishonesty__plain           | ID       | n_resp      |     9 |   15   |   23   |     31   |   39   |   50   |   102 |
| instruct_dishonesty__plain           | ID       | n_final_msg |     9 |   15   |   23   |     31   |   39   |   50   |   102 |
| XXXX-20__plain                       | Alpaca   | n_tokens    |    55 |   74   |  134   |    283   |  538   |  559   |   650 |
| XXXX-20__plain                       | Alpaca   | n_resp      |     1 |   15   |   69   |    217.5 |  484.2 |  500   |   500 |
| XXXX-20__plain                       | Alpaca   | n_final_msg |     1 |   15   |   69   |    217.5 |  484.2 |  500   |   500 |

## HP-KR subgroups (claim x belief)

| subgroup         | claim             |   belief (0-3) |   no belief (-1) |
|:-----------------|:------------------|---------------:|-----------------:|
| honest_admission | CLAIMED_KNOWLEDGE |           1833 |                0 |
| honest_denial    | DENIED_KNOWLEDGE  |              0 |              135 |
| honest_other     | OTHER             |             29 |                3 |
| lie              | DENIED_KNOWLEDGE  |            125 |                0 |
| lie              | OTHER             |             14 |                0 |

See `hpkr_templates.md`.

See `positions_decoded.md`.

## Gemotions

`dejanseo/gemotions` @ `4fd2ac63551f1be37e6e6c2eacd1b1898c9af656`. stories_clean: 171000 rows, 171 emotions x 100 topics, 17100 of 17100 cells filled; stories per cell {'min': 10.0, 'p5': 10.0, 'p25': 10.0, 'median': 10.0, 'p75': 10.0, 'p95': 10.0, 'max': 10.0}; eligible (>= 70 tokens) per cell {'min': 4.0, 'p5': 10.0, 'p25': 10.0, 'median': 10.0, 'p75': 10.0, 'p95': 10.0, 'max': 10.0}, cells with none 0. Story tokens {'min': 42.0, 'p5': 93.0, 'p25': 119.0, 'median': 140.0, 'p75': 164.0, 'p95': 208.0, 'max': 489.0}; share below 70: 0.003.

Names the spec uses but Gemotions lacks: ['defensive', 'protective'].

neutral.db: {'api_calls': 100, 'sqlite_sequence': 2, 'dialogues': 1200}.

## Valence norms

NRC VAD v2.1 (https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip): 164 of 171 emotions matched; unmatched ['energized', 'grief-stricken', 'insulted', 'self-confident', 'self-conscious', 'self-critical', 'stimulated'].
