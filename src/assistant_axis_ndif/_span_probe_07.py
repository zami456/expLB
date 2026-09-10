#!/usr/bin/env python3
"""Step 0a (part 2): exercise the REAL 2_activations_ndif.py session path on a small
HOT model, so the shipped code -- not a proxy -- is what gets validated.

    .venv-aa/bin/python assistant_axis/pipeline/_span_probe_07.py [model]

Asserts: layer .output is a (B,T,H) tensor (not a tuple), the masked span-mean
reduces to (K,H), and a multi-batch session accumulates server-side correctly.
"""
import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PIPELINE_DIR = Path(__file__).resolve().parent
load_dotenv(PIPELINE_DIR / ".env")
load_dotenv(PIPELINE_DIR.parents[1] / ".env")
os.environ.setdefault("NDIF_HOST", "https://api.ndif.us")
for _v in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
    if os.getenv(_v):
        os.environ["HF_TOKEN"] = os.environ["HF_HUB_TOKEN"] = os.environ[_v].strip()
        break

import torch  # noqa: E402
from nnsight import CONFIG, LanguageModel  # noqa: E402

CONFIG.API.HOST = os.environ["NDIF_HOST"]
CONFIG.API.APIKEY = (os.getenv("NDIF_API_KEY") or "").strip()
CONFIG.APP.REMOTE_LOGGING = False

# load 2_activations_ndif.py by path (module name starts with a digit)
_spec = importlib.util.spec_from_file_location("act_ndif", PIPELINE_DIR / "2_activations_ndif.py")
act = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(act)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B"
llm = LanguageModel(MODEL)
NL, H = int(llm.config.num_hidden_layers), int(llm.config.hidden_size)
pad_id = llm.tokenizer.pad_token_id or llm.tokenizer.eos_token_id
print(f"model={MODEL}  layers={NL}  hidden={H}  pad_id={pad_id}")

# --- resolve_layers ------------------------------------------------------- #
assert act.resolve_layers(["target"], NL, NL // 2) == [NL // 2]
assert act.resolve_layers(["all"], NL, NL // 2) == list(range(NL))
assert act.resolve_layers(["3", "1", "1"], NL, 0) == [1, 3]
try:
    act.resolve_layers([str(NL)], NL, 0); raise SystemExit("resolve_layers should reject out-of-range")
except ValueError:
    pass
print("resolve_layers: OK")

# --- build 4 items, span = the second half of each sequence ---------------- #
texts = ["The capital of France is Paris, a city famous for its museums.",
         "Photosynthesis converts light energy into chemical energy in plants.",
         "The mitochondrion is often called the powerhouse of the cell.",
         "Ada Lovelace wrote the first algorithm intended for a machine."]
items = []
for t in texts:
    ids = llm.tokenizer(t)["input_ids"]
    items.append((ids, len(ids) // 2, len(ids)))   # (ids, a0, a1)

# --- pad_batch shape/mask sanity (pure client-side) ------------------------ #
ii, am, sp = act.pad_batch(items[:2], pad_id)
assert ii.shape == am.shape == sp.shape, (ii.shape, am.shape, sp.shape)
for b, (ids, a0, a1) in enumerate(items[:2]):
    assert am[b].sum().item() == len(ids), "attention_mask must cover exactly the real tokens"
    assert sp[b].sum().item() == a1 - a0, "span mask must cover exactly the assistant span"
    assert sp[b, :a0].sum().item() == 0 and sp[b, a1:].sum().item() == 0
print(f"pad_batch: OK  padded={tuple(ii.shape)}")

LAYERS = [NL // 2]

# --- single batch through the real remote path ---------------------------- #
got = act._run_batch_client(llm, LAYERS, ii, am, sp, retries=2)
assert got is not None, "client-mode batch failed"
one, n1 = got
print(f"_run_batch_client: OK  shape={tuple(one.shape)} n={n1}  (want ({len(LAYERS)}, {H}), 2.0)")
assert tuple(one.shape) == (len(LAYERS), H), "layer .output was probably a tuple, not (B,T,H)"
assert torch.isfinite(one).all()

# --- same two batches via a session; must equal the sum of the parts ------- #
b1 = act.pad_batch(items[:2], pad_id)
b2 = act.pad_batch(items[2:], pad_id)
got = act._run_session(llm, LAYERS, H, [b1, b2], retries=2)
assert got is not None, "session-mode failed"
sess, nS = got
c1 = act._run_batch_client(llm, LAYERS, *b1, retries=2)[0]
c2 = act._run_batch_client(llm, LAYERS, *b2, retries=2)[0]
ref = c1 + c2
diff = (sess - ref).abs().max().item()
rel = diff / ref.abs().max().item()
print(f"_run_session: shape={tuple(sess.shape)} n={nS} (want 4.0)")
print(f"  session-sum vs client-sum: max abs diff={diff:.3e}  rel={rel:.3e}")
assert nS == 4.0, f"session count wrong: {nS}"
assert rel < 1e-3, "server-side accumulation disagrees with per-batch download"
print("\nALL CHECKS PASSED -- session accumulation is correct; --accum session is safe.")
