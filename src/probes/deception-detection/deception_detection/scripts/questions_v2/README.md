# questions_v2: the 150-question shared set for the persona bank

Provenance of `kd_questions_shared_v2.jsonl` (ids 0-149; `id` equals the line position, which is the
pipeline's `question_index`):

| ids | set | source |
|---|---|---|
| 0-29 | `orig` | the kd bank's 30 questions, byte-identical and in order (`kd_questions_shared.jsonl`) |
| 30-59 | `mono` | 30 control questions in the originals' form, types cycling quality, quantity, relation, manner, denial, ignorance |
| 60-149 | `div` | 90 questions, one per slot of `grid_slots.json` (type × who would deceive × domain × format) |

Files, in build order:

1. `grid_allocate.py` → `grid_slots.json`, `grid_slots.md` (closed-form allocation, seed 20261016).
2. `prompt_template.md` + `build_prompt.py` → `prompt.md` (the fixed authoring prompt; includes the slot table and the 30 originals as exemplars).
3. `raw_claude_output.md`: the generator's output, verbatim, one pass (Claude Fable 5.1, 2026-10-16). Inputs opened while writing: `prompt.md` and `grid_slots.json` only.
4. `postprocess.py` → `kd_questions_shared_v2.jsonl` (deterministic; no manual edits).
5. `check_questions.py`: the gates (ids, originals unchanged, duplicates, lengths, grid margins, n-gram overlap among new items and against the originals, format markers). Prints the SHA-256.
6. `PREREGISTRATION.md`, `FREEZE.sha256`, and `E:\qbank\logs\freeze.txt` (hashes + commit, before any generation).

Deliberate restrictions:

- The two-turn format is rendered inside one user message (`Earlier I asked: "…" You said: "…"` then the follow-up), so the pipeline's record schema, the judge template, the span rule and the Aletheia renderers are untouched. Real multi-turn transcripts are not tested.
- "Persona" items put the assistant in the situation with a second-person premise and never assign it a name or a new identity; the system-prompt persona stays in charge of who is speaking.
- Overlap checks run against the 30 originals only. No test set is read by any script in this folder.
- Targeted cells (grid cells that coincide with a known test behaviour) are declared in `grid_allocate.targeted` and marked in the jsonl (`targeted`).
