#!/usr/bin/env python3
"""Throwaway: verify NDIF creds + nnsight 0.7 API shape before the real smoke test.

    .venv-aa/bin/python src/assistant_axis_ndif/_ndif_probe.py [model]

Checks: remote trace works, what `layer.output` looks like, layer count, tokenizer.
"""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import torch  # noqa: E402
import nnsight  # noqa: E402
from nnsight import CONFIG, LanguageModel  # noqa: E402

print("nnsight", nnsight.__version__, "torch", torch.__version__)

api_key = (os.getenv("NDIF_API_KEY") or "").strip()
hf = (os.getenv("HF_TOKEN") or "").strip()
os.environ["HF_TOKEN"] = hf
os.environ["HF_HUB_TOKEN"] = hf
print("default CONFIG.API.HOST:", getattr(CONFIG.API, "HOST", "?"))
host = os.getenv("NDIF_HOST", "").strip() or "https://api.ndif.us"
if not host.startswith("http"):
    host = "https://" + host
CONFIG.API.HOST = host
for cand in ("WS_HOST", "WSHOST", "WEBSOCKET_HOST"):
    if hasattr(CONFIG.API, cand):
        print("  CONFIG.API.%s =" % cand, getattr(CONFIG.API, cand))
CONFIG.API.APIKEY = api_key
print("NDIF host:", CONFIG.API.HOST, "| key set:", bool(api_key), "| hf set:", bool(hf))

model_name = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B"
print("\n=== instantiating", model_name, "===")
t = time.time()
llm = LanguageModel(model_name)
print("  LanguageModel() in", round(time.time() - t, 1), "s")
print("  num_hidden_layers:", llm.config.num_hidden_layers, "| hidden:", llm.config.hidden_size)
print("  tokenizer pad:", llm.tokenizer.pad_token_id, "eos:", llm.tokenizer.eos_token_id)

toks = llm.tokenizer("The capital of France is", return_tensors="pt")
print("  input_ids shape:", tuple(toks["input_ids"].shape))

print("\n=== remote trace ===")
t = time.time()
try:
    with llm.trace({"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}, remote=True):
        out_direct = llm.model.layers[0].output
        shp = out_direct.shape.save()                                  # tuple? tensor?
        h = out_direct                                                 # tf5: tensor (B,T,H)
        h_mean = h.detach().to(torch.float32).mean(dim=1).cpu().save()  # want (B, H)
        hs_stack = torch.stack([llm.model.layers[L].output for L in range(min(3, llm.config.num_hidden_layers))])
        hs_shape = hs_stack.shape.save()
        last_logits = llm.output.logits[:, -1, :].argmax(dim=-1).cpu().save()
    hm = getattr(h_mean, "value", h_mean)
    ll = getattr(last_logits, "value", last_logits)
    print("  OK in", round(time.time() - t, 1), "s")
    print("  layers[0].output .shape ->", getattr(shp, "value", shp))
    print("  torch.stack([layers[L].output for L in 0..2]).shape ->", getattr(hs_shape, "value", hs_shape))
    print("  layer0 out mean shape:", tuple(hm.shape), "dtype:", hm.dtype, "(want (1, hidden))")
    print("  greedy next token:", repr(llm.tokenizer.decode(ll)))
except Exception as e:
    print("  REMOTE TRACE FAILED in", round(time.time() - t, 1), "s")
    print("  ", type(e).__name__, str(e)[:800])
    sys.exit(2)
