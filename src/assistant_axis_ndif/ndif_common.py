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


def harden(min_kbps: float = 300.0, grace: float = 12.0, attempts: int = 4) -> None:
    """Install every mitigation and say so, so a log records which run was protected."""
    ok = install_download_deadline(min_kbps, grace, attempts)
    logger.info("NDIF download floor %.0f KB/s after %.0fs: %s",
                min_kbps, grace, "installed" if ok else "NOT installed (internals moved)")
