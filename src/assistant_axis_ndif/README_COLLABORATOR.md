# Assistant Axis run — quick guide

Thanks for running this. It extracts **persona vectors** from Llama-3.3-70B for 100
character roles, then builds the "Assistant Axis" from them. **All the heavy compute runs
remotely on NDIF** — your machine only sends prompts and stores small result files, so no
GPU is needed. It just needs to stay on for a couple of days.

Claude Code does the actual work. You only need to do the four setup steps below, then tell
Claude to run it.

## What you need (one-time, ~10 minutes)

1. **An NDIF API key** — free, from <https://login.ndif.us>.
2. **A Hugging Face token** (read scope) from <https://huggingface.co/settings/tokens>, and
   with that same account click **"Agree and access"** on both:
   - <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>
   - <https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct>
3. **Put the keys in a `.env` file**: copy `src/assistant_axis_ndif/env.aa.example` to
   `src/assistant_axis_ndif/.env` and paste your `NDIF_API_KEY` and `HF_TOKEN` into it.
   (This file stays on your machine — it is never committed or shared.)
4. **`uv`** installed — if you don't have it: `winget install astral-sh.uv`.

## Then just tell Claude

Open this repo in Claude Code and say:

> Run the assistant axis pipeline per `src/assistant_axis_ndif/RUNBOOK.md`.

Claude will set up the environment, run a ~15-minute smoke test to confirm everything works,
then start the full run. It manages everything — you don't need to babysit commands.

## What to expect

- **~1.5–2 days** of wall-clock, mostly generation. It runs unattended.
- **Interrupting is safe.** If the machine sleeps, reboots, or you Ctrl-C, just tell Claude
  to run it again — it resumes exactly where it left off and never redoes finished work.
- Keep the machine **awake and online** (disable sleep). NDIF is a shared service, so speed
  varies with load.

## What to send back

When it finishes, send me this folder (~260 MB):

```
results/assistant_axis/meta-llama_Llama-3.3-70B-Instruct/
    vectors/*.pt      the per-role persona vectors
    axis.pt           the Assistant Axis
    axes.pt           axis + deception/sycophancy contrasts + PCA
```

Zip it and share however is easiest. That's the whole deliverable — thank you!

*(If Claude reports the "b'P'" socket error, the RUNBOOK tells it how to fix it. Anything
else odd, send me the message.)*
