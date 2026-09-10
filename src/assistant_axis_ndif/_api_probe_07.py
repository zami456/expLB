#!/usr/bin/env python3
"""Step 0a: settle the four nnsight-0.7.0 API unknowns against the live NDIF service.

The docs vendored at repo `docs/` describe nnsight 0.8; `.venv-aa` runs 0.7.0. This
script decides, empirically, which idiom the pipeline must use. Each check is
independent -- a failure prints a verdict and moves on.

    .venv-aa/bin/python src/assistant_axis_ndif/_api_probe_07.py [model]

Default model is Llama-3.1-8B (HOT + pinned on NDIF) so the checks are fast; the
answers are API-shape facts that carry to the 70B.
"""
import os
import sys
import time
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
import nnsight  # noqa: E402
from nnsight import CONFIG, LanguageModel  # noqa: E402

host = (os.getenv("NDIF_HOST", "").strip() or "https://api.ndif.us")
if not host.startswith("http"):
    host = "https://" + host
CONFIG.API.HOST = host
CONFIG.API.APIKEY = (os.getenv("NDIF_API_KEY") or "").strip()
CONFIG.APP.REMOTE_LOGGING = False  # keep the spinner out of the transcript

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B"
print(f"nnsight {nnsight.__version__} | torch {torch.__version__} | host {CONFIG.API.HOST}")
print(f"model: {MODEL}\n")

llm = LanguageModel(MODEL)
NL, H = int(llm.config.num_hidden_layers), int(llm.config.hidden_size)
print(f"config: {NL} layers, hidden={H}")

tok = llm.tokenizer("The capital of France is", return_tensors="pt")
IDS, ATTN = tok["input_ids"], tok["attention_mask"]

verdict = {}


def check(name):
    def deco(fn):
        print(f"\n=== {name} ===")
        t = time.time()
        try:
            res = fn()
            verdict[name] = res
            print(f"  -> {res}   [{time.time() - t:.1f}s]")
        except Exception as e:  # noqa: BLE001
            verdict[name] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
            print(f"  -> FAILED {type(e).__name__}: {str(e)[:300]}   [{time.time() - t:.1f}s]")
        return fn
    return deco


# --- 4. layer .output shape: tensor (B,T,H) or a tuple? ---------------------- #
@check("4_layer_output_shape")
def _():
    with llm.trace({"input_ids": IDS, "attention_mask": ATTN}, remote=True):
        o = llm.model.layers[0].output
        is_tuple = isinstance(o, tuple)
        shp = (o[0] if is_tuple else o).shape.save()
    s = tuple(getattr(shp, "value", shp))
    return f"tuple={is_tuple}  shape={s}  (want tuple=False, (1,T,{H}))"


# --- 1. remote generate: tracer.result (0.8 docs) vs llm.generator.output (0.7) --- #
@check("1a_generate_tracer_result")
def _():
    with llm.generate(IDS, attention_mask=ATTN, max_new_tokens=3,
                      do_sample=False, remote=True) as tracer:
        if not hasattr(tracer, "result"):
            raise AttributeError("tracer has no attribute 'result'")
        r = tracer.result.save()
    r = getattr(r, "value", r)
    return f"WORKS  shape={tuple(r.shape)}  {llm.tokenizer.decode(r[0])!r}"


@check("1b_generate_generator_output")
def _():
    with llm.generate(IDS, attention_mask=ATTN, max_new_tokens=3,
                      do_sample=False, remote=True):
        r = llm.generator.output.save()
    r = getattr(r, "value", r)
    return f"WORKS  shape={tuple(r.shape)}  {llm.tokenizer.decode(r[0])!r}"


# --- 2. does generate() nest inside session(remote=True)? -------------------- #
@check("2_generate_inside_session")
def _():
    with llm.session(remote=True):
        outs = nnsight.save([])
        for _i in range(2):
            with llm.generate(IDS, attention_mask=ATTN, max_new_tokens=3, do_sample=False):
                outs.append(llm.generator.output)
    o = getattr(outs, "value", outs)
    return f"WORKS  n={len(o)}  shapes={[tuple(x.shape) for x in o]}"


# --- 3. in-place accumulator across traces inside one session ---------------- #
@check("3a_accum_tensor_save_iadd")
def _():
    with llm.session(remote=True):
        acc = torch.zeros(4, dtype=torch.float32).save()
        cnt = torch.zeros(1, dtype=torch.float32).save()
        for _i in range(3):
            with llm.trace({"input_ids": IDS, "attention_mask": ATTN}):
                h = llm.model.layers[0].output
                acc += h[0, 0, :4].detach().to(torch.float32).cpu()
                cnt += 1.0
    a, c = getattr(acc, "value", acc), getattr(cnt, "value", cnt)
    ok = float(c[0]) == 3.0
    return f"cnt={float(c[0])} (want 3.0) acc={[round(float(x),3) for x in a]} -> {'WORKS' if ok else 'BROKEN'}"


@check("3b_accum_nnsight_save_iadd")
def _():
    with llm.session(remote=True):
        acc = nnsight.save(torch.zeros(1, dtype=torch.float32))
        for _i in range(3):
            with llm.trace({"input_ids": IDS, "attention_mask": ATTN}):
                acc += 1.0
    a = getattr(acc, "value", acc)
    ok = float(a[0]) == 3.0
    return f"acc={float(a[0])} (want 3.0) -> {'WORKS' if ok else 'BROKEN'}"


@check("3c_fallback_saved_list_of_sums")
def _():
    """Fallback if 3a/3b are broken: collect per-trace (K,H) sums in a saved list."""
    with llm.session(remote=True):
        sums = nnsight.save([])
        for _i in range(3):
            with llm.trace({"input_ids": IDS, "attention_mask": ATTN}):
                h = llm.model.layers[0].output
                sums.append(h[0, :, :4].detach().to(torch.float32).sum(dim=0).cpu())
    s = getattr(sums, "value", sums)
    return f"n={len(s)} (want 3)  shapes={[tuple(x.shape) for x in s]}"


print("\n" + "=" * 72)
print("VERDICTS")
print("=" * 72)
for k, v in verdict.items():
    print(f"  {k:34s} {v}")
