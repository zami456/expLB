#!/usr/bin/env python
"""Resilience helpers shared by every script that talks to NDIF.

These were all learned the hard way during Pass B and then existed in exactly one script,
`8_coords_ndif.py`, while `1_generate_ndif.py`, `2_activations_ndif.py`, `3_judge_ndif.py`
and `10_sycophancy_ndif.py` had none of them. That gap cost a 40-minute silent hang during
the persona-bank rebuild: the NDIF status counter froze at "RUNNING (32.0s)" and stayed
there, and because a hang raises nothing, `--retries` never fired and not one vector was
written.

Nothing here is a trace body, so importing it is safe. A function that appears *inside*
`trace`/`generate`/`session` must be defined in the script that runs, because nnsight
serialises the block from its source module and NDIF rejects one that belongs to an
importlib-loaded module with "Module importlib is not whitelisted".

`8_coords_ndif.py` keeps its own copy of the deadline rather than importing this. That
duplication is deliberate: it is the script that produced Pass B and will produce Passes A
and C, and swapping a proven import path mid-run buys nothing.
"""

from __future__ import annotations

import io
import logging
import time

logger = logging.getLogger(__name__)


class SlowDownload(RuntimeError):
    """An NDIF result download collapsed and is not worth waiting for."""


class SessionTimeout(RuntimeError):
    """An NDIF job gave no result within NDIF_SESSION_TIMEOUT seconds and was abandoned."""


def session_deadline(seconds: float | None = None):
    """Context manager: abandon a remote session that gives no result in `seconds` (2026-10-14).

    A job can sit at "RUNNING" for hours and never finish (a qwen_roleplay job: 6 h 10 m). That raises nothing, so
    `--retries` never fires and a chain waiting on the shard stalls for good. A timer interrupts the main thread --
    which is polling NDIF in Python -- and the interrupt is re-raised here as SessionTimeout, an ordinary Exception,
    so each script's existing retry loop retries the session. `seconds` defaults to the NDIF_SESSION_TIMEOUT
    environment variable; unset or 0 = no deadline (every earlier run's behaviour). Wrap the session:
    `with _nc.session_deadline(), llm.session(remote=True): ...` -- this function is not a trace body."""
    import _thread
    import contextlib
    import os
    import threading

    if seconds is None:
        seconds = float(os.environ.get("NDIF_SESSION_TIMEOUT") or 0)

    @contextlib.contextmanager
    def _cm():
        if not seconds or seconds <= 0:
            yield
            return
        fired = threading.Event()
        timer = threading.Timer(seconds, lambda: (fired.set(), _thread.interrupt_main()))
        timer.daemon = True
        timer.start()
        try:
            yield
        except KeyboardInterrupt:
            if fired.is_set():
                raise SessionTimeout(f"no result in {seconds:.0f}s; abandoned (NDIF_SESSION_TIMEOUT)") from None
            raise
        finally:
            timer.cancel()

    return _cm()


def exc_tail(exc: BaseException, limit: int = 400) -> str:
    """The informative last line of an exception, not the first 400 characters of it.

    nnsight wraps a remote failure in a client-side traceback, so `str(exc)[:300]` spends
    its whole budget on "File remote.py, line 567, in __call__" and stops immediately before
    the RemoteException description. 160 sessions failed that way during Pass B and not one
    log line said why; switching to the last line revealed the cause in a single run.
    """
    lines = [ln for ln in str(exc).strip().splitlines() if ln.strip()]
    return (lines[-1].strip() if lines else repr(exc))[:limit]


def install_download_deadline(min_kbps: float = 300.0, grace: float = 12.0,
                              attempts: int = 4) -> bool:
    """Abort a collapsed NDIF result download and re-GET it on a fresh connection.

    Measured on the lab PC 2026-09-15: result downloads from prod-ndif-results.s3 are
    bimodal. A healthy 34 MB session arrives in 3-35s; a sick one trickles at ~35 KB/s for
    eleven minutes while the worker sits at 0% CPU and NDIF reports RUNNING=15s. Not
    bandwidth -- 91 MB/s to Cloudflare on the same host at the same moment -- not the socket
    pins, and not client reuse, since nnsight opens a fresh httpx.Client per download.
    One sick TCP flow; a new connection clears it.

    httpx's read timeout cannot catch this, because bytes keep arriving, just very few. So
    the check is on throughput after a grace period, and a violation retries the GET against
    the same presigned URL rather than raising straight out: the result is already in S3, so
    re-dispatching the whole session would pay a recompute to fetch a file that exists. Only
    when every attempt draws a sick flow does this raise, and then the caller's own retry
    takes over.

    Returns False if nnsight's internals have moved, in which case the caller runs unpatched.
    """
    try:
        from nnsight.intervention.backends.remote import RemoteBackend
    except Exception:  # noqa: BLE001
        return False
    if not (hasattr(RemoteBackend, "get_result")
            and hasattr(RemoteBackend, "_decompress_and_load")):
        return False

    import httpx
    from tqdm import tqdm

    def get_result(self, url: str, content_length=None):
        timeout = httpx.Timeout(self.CONNECT_TIMEOUT, read=self.READ_TIMEOUT)
        last = None
        for attempt in range(1, attempts + 1):
            buf = io.BytesIO()
            t0, got = time.time(), 0
            try:
                with httpx.Client(timeout=timeout) as client:
                    with client.stream("GET", url) as stream:
                        total = content_length or float(
                            stream.headers.get("Content-length", 0))
                        with tqdm(total=total or None, unit="B", unit_scale=True,
                                  desc="downloading", leave=False) as bar:
                            for data in stream.iter_bytes(chunk_size=128 * 1024):
                                got += len(data)
                                bar.update(len(data))
                                buf.write(data)
                                elapsed = time.time() - t0
                                if elapsed > grace and (got / 1024.0 / elapsed) < min_kbps:
                                    raise SlowDownload(
                                        "%.1f MB in %.0fs = %.0f KB/s (floor %.0f)"
                                        % (got / 1e6, elapsed,
                                           got / 1024.0 / elapsed, min_kbps))
                return self._decompress_and_load(buf)
            except SlowDownload as exc:
                last = exc
                buf.close()
                logger.warning("  download attempt %d/%d: %s -- new connection",
                               attempt, attempts, exc)
        raise SlowDownload("%s after %d attempts" % (last, attempts))

    RemoteBackend.get_result = get_result
    return True


def tokenizer_kwargs(repo: str) -> dict:
    """Extra tokenizer arguments a checkpoint needs (xmodel 2026-10-07).

    Mistral's tekken tokenizers (Small-24B-2501, Small-3.1-24B-2503) ship a Llama-3-style
    pre-tokenizer regex; transformers 5 loads it as is and warns "This will lead to incorrect
    tokenization" unless asked to replace it with fix_mistral_regex=True. Measured on 2501: the
    pre-tokenizers differ, yet 0 of 1,210 of our texts (kd questions, persona prompts, 1,000
    replies) tokenise differently -- so it is set for correctness, not because it moved anything.
    """
    return {"fix_mistral_regex": True} if "mistral" in repo.lower() else {}


# Every Liars' Bench Mistral row carries Mistral Small 3's default system prompt dated this day
# (results/subsets/harm-pressure-knowledge-report.csv: 2,358 of 2,358 rows).
TEMPLATE_DATE = "2025-09-01"


def pin_template_date(tok, date: str = TEMPLATE_DATE):
    """Take the clock out of the chat template (xmodel 2026-10-07).

    Mistral-Small's template builds its default system prompt -- used whenever a conversation has no
    system message, as the default role's first variant ("") has in every bank -- from
    strftime_now("%Y-%m-%d"): the client's clock. A reply generated one day and read the next would
    be read under a different system prompt, and no rerun could reproduce either. The date is fixed to
    the one in Liars' Bench's own Mistral prompts. Llama-3.3 (a static "26 Jul 2024") and Qwen2.5 (no
    date) never read the clock, so this is a no-op for them; any other strftime_now raises.
    """
    t = getattr(tok, "chat_template", None)
    if isinstance(t, str) and "strftime_now" in t:
        t = t.replace('strftime_now("%Y-%m-%d")', f'"{date}"')
        if "strftime_now" in t:
            raise RuntimeError("chat template reads the clock in a form pin_template_date does not know")
        tok.chat_template = t
    return tok


def load_tokenizer(repo: str):
    """The tokenizer every xmodel script uses for `repo`: Mistral's regex fix, the processor's chat
    template when the tokenizer ships none (Mistral-Small-3.1), and the template date pinned."""
    import json as _json
    import os

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo, token=os.getenv("HF_TOKEN"), **tokenizer_kwargs(repo))
    if not tok.chat_template:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(repo, "chat_template.json", token=os.getenv("HF_TOKEN"))
        tok.chat_template = _json.loads(open(p, encoding="utf-8").read())["chat_template"]
    return pin_template_date(tok)


def load_language_model(repo: str):
    """The nnsight wrapper for `repo`, with the tokenizer fixes above (xmodel 2026-10-07).

    Text checkpoints (Llama, Qwen, Mistral-Small-2501): LanguageModel, as before, plus
    tokenizer_kwargs and the pinned template date.

    Vision-language checkpoints (Mistral-Small-3.1-2503: Mistral3ForConditionalGeneration):
    nnsight 0.7's LanguageModel refuses them and names VisionLanguageModel. That class loads an
    AutoProcessor, whose image half needs Pillow/torchvision -- absent here, and installing
    torchvision would move the pinned torch. Text-only inputs never touch the processor
    (VisionLanguageModel._prepare_input falls back to LanguageModel's tokenizer path), so it is
    given a stand-in holding only the tokenizer. The tokenizer has no chat template of its own;
    the processor's chat_template.json supplies it. On NDIF the model key is
    VisionLanguageModel:{repo_id, revision}, so the server builds the real model. The decoder is
    at model.language_model.layers (2_activations / 8_coords pass vl=True).

    Gemma-3 (Gemma3ForConditionalGeneration) goes the same way (2026-10-08). It had been excluded on the
    guess that LanguageModel loads it, since transformers maps gemma3 under causal-LM too; it does not:
    nnsight 0.7's _check_is_text_only refuses any model_type in the image-text-to-text mapping
    (measured on transformers 5.8 and 5.17: "Use VisionLanguageModel instead").
    """
    import json as _json
    import os
    from types import SimpleNamespace

    from transformers import AutoConfig, AutoTokenizer

    tok_kw = tokenizer_kwargs(repo)
    cfg = AutoConfig.from_pretrained(repo, token=os.getenv("HF_TOKEN"))
    arch = (getattr(cfg, "architectures", None) or [""])[0]
    if arch.endswith("ForConditionalGeneration"):
        from huggingface_hub import hf_hub_download
        from nnsight.modeling.vlm import VisionLanguageModel

        tok = load_tokenizer(repo)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return VisionLanguageModel(repo, processor=SimpleNamespace(tokenizer=tok), tokenizer=tok)
    from nnsight import LanguageModel

    llm = LanguageModel(repo, tokenizer_kwargs=tok_kw)
    pin_template_date(llm.tokenizer)
    return llm


def harden(min_kbps: float = 300.0, grace: float = 12.0, attempts: int = 4) -> None:
    """Install every mitigation and say so, so a log records which run was protected.

    NDIF_MIN_DOWNLOAD_KBPS / NDIF_DOWNLOAD_GRACE override the floor (2026-10-07, laptop): three shards
    downloading at once split the laptop's link to ~70-95 KB/s each, under the lab-tuned 300 KB/s, and
    healthy downloads were cut and their NDIF sessions re-run. Unset, the defaults are unchanged."""
    import os
    min_kbps = float(os.environ.get("NDIF_MIN_DOWNLOAD_KBPS") or min_kbps)
    grace = float(os.environ.get("NDIF_DOWNLOAD_GRACE") or grace)
    ok = install_download_deadline(min_kbps, grace, attempts)
    logger.info("NDIF download floor %.0f KB/s after %.0fs: %s",
                min_kbps, grace, "installed" if ok else "NOT installed (internals moved)")


def trim_chatml_header_newline(tok, ids, a0: int, a1: int) -> int:
    """Start a ChatML assistant span after the newline that closes its role header.

    Measured 2026-10-07: the Assistant-Axis ConversationEncoder (its Qwen branch was written for
    Qwen-3) starts a Qwen-2.5 assistant span AT the '\n' of '<|im_start|>assistant\n', so every
    persona and evaluation mean folded in one header token ('\nThat is not something I have.'),
    while Llama's span starts at the first word ('That is not something I have.'). Returns a0 + 1
    when the span's first token is exactly that newline, right after the 'assistant' role tokens
    that follow '<|im_start|>'; otherwise a0 unchanged -- a no-op for non-ChatML templates (Llama).
    """
    start = tok.convert_tokens_to_ids("<|im_start|>")
    if not isinstance(start, int) or start == getattr(tok, "unk_token_id", None):
        return a0
    nl = tok.encode("\n", add_special_tokens=False)
    role = tok.encode("assistant", add_special_tokens=False)
    k = len(role)
    if len(nl) != 1 or a1 - a0 < 2 or a0 - k - 1 < 0:
        return a0
    if ids[a0] == nl[0] and list(ids[a0 - k:a0]) == role and ids[a0 - k - 1] == start:
        return a0 + 1
    return a0
