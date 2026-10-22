# GV1: Verdict rows

- every agent-episode has a row or a recorded exclusion: yes
- own messages equal the channel transcript: yes
- verdict turns equal the recorded verdicts: yes
- Llama rows within budget: yes
- Qwen rows within budget: yes
- Llama, BARE and Qwen rows cover the same agent-episodes: yes
- AWQ check rows tokenised as recorded: yes

Rows: 1000 agent-episodes; excluded 0 (none).
Earlier episodes dropped for the Llama budget: {'0': 1000}; extra for Qwen: {'0': 1000}. Own messages per agent-episode: {'5': 762, '4': 193, '3': 40, '2': 4, '1': 1}. Rendering notes: {'dropped send_message': 2588, 'dropped submit_verdict': 13, 'dropped query_database': 2, 'assistant text beside tool calls': 8}.

Files: `llama_ctx` 1000 rows, 3,014,716 tokens, longest 6668; `llama_bare_comm` 1000 rows, 237,847 tokens, longest 404; `llama_bare_refl` 1000 rows, 162,364 tokens, longest 776; `qwen_verdict` 1000 rows, 4,667,236 tokens, longest 12236; `qwen_reflection` 1000 rows, 5,437,918 tokens, longest 12964; `qwen_awq_check` 20 rows, 52,515 tokens, longest 2802

# GV3: labels

- collusion 146 of 500: yes
- 33 of 50 runs: yes
- collusion = both_accept: yes
- trace verdicts = episodes table: yes
- Rose-flagged episodes: 51; valid relaxation labels: 1000

GV1 PASSED; GV3 PASSED.
