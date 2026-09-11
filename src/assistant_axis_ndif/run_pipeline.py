#!/usr/bin/env python
"""Cross-platform driver for the NDIF Assistant-Axis pipeline.

The bash driver (run_pipeline_ndif.sh) is the reference; this is the Windows-friendly
equivalent so a fresh Claude can run one command on any OS. It runs the stages in order:

    1_generate_ndif.py    role responses            (Llama-3.3-70B on NDIF)
    3_judge_ndif.py       0-3 role-adherence        (Llama-3.1-70B on NDIF)
    2_activations_ndif.py per-role persona vectors   (Llama-3.3-70B on NDIF)
    5_axis.py             assistant axis             (local CPU, upstream, unmodified)
    6_axes.py             axis + class contrasts + PCA + validation   (local CPU)

Every stage is independently restartable -- safe to Ctrl-C and re-run. The deliverable
is <OUT_DIR>/vectors/*.pt + axis.pt + axes.pt.

Config comes from env vars (same names as the bash driver), so it reads identically on
Windows and Linux:

    OUT_DIR         (required) output directory
    MODEL           default meta-llama/Llama-3.3-70B-Instruct
    JUDGE_MODEL     default meta-llama/Llama-3.1-70B-Instruct
    LAYERS          "all" (default; every layer, ~2.6 MB/role) | "target" (middle
                    layer 40 only, ~32 KB/role) | explicit list "38 40 42"
    ROLES           explicit role names (overrides ROLES_FILE)
    ROLES_FILE      role-set JSON with group labels (default role_groups.json)
    GROUPS          subset of ROLES_FILE groups, e.g. "deception honesty"
    SHARD           "i/n" -- run only shard i of n; launch n processes to keep several
                    NDIF jobs in flight (each stage is per-role and resumable, so shards
                    never collide)
    PROMPT_INDICES  e.g. "0 1"   (default: all 5)
    QUESTION_COUNT  default 240
    GEN_BATCH / JUDGE_BATCH / ACT_BATCH                default 24 / 48 / 8
    GEN_SESSION_BATCHES / JUDGE_SESSION_BATCHES / ACT_SESSION_BATCHES   default 8 / 8 / 20
    MIN_COUNT       min score-3 responses/role for a vector (default 50; lower for smoke)
    AA_PY           python to run stages with (default: the repo's .venv-aa)
    ONLY            comma-sep stages to run, e.g. "1,3,2" or "6" (default: all)

Requires src/assistant_axis_ndif/.env (see env.aa.example).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

# Windows (uv standalone Python): give child stages a UTF-8 console (cp1252 can't encode
# nnsight's status glyphs) and a working CA bundle (its bundled store is stale, so the
# websocket TLS handshake fails). The stages also self-apply these, but setting them in the
# environment here means the child processes start already correct. Harmless on Linux.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:  # noqa: BLE001
    pass
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

PIPE = Path(__file__).resolve().parent
REPO = PIPE.parents[1]
AA_PIPE = REPO / "assistant_axis" / "pipeline"       # upstream 5_axis.py (unmodified)
ROLES_DIR = REPO / "assistant_axis" / "data" / "roles" / "instructions"


def find_python() -> str:
    """The venv python to run stages with: AA_PY, else the repo .venv-aa for this OS."""
    if os.environ.get("AA_PY"):
        return os.environ["AA_PY"]
    win = REPO / ".venv-aa" / "Scripts" / "python.exe"
    nix = REPO / ".venv-aa" / "bin" / "python"
    if win.exists():
        return str(win)
    if nix.exists():
        return str(nix)
    raise SystemExit(
        f"no venv python found ({win} or {nix}); run setup_venv_aa"
        f"{'.ps1' if os.name == 'nt' else '.sh'} first, or set AA_PY"
    )


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def split_args(value: str | None) -> list[str]:
    """A space-separated env value -> list of tokens (e.g. PROMPT_INDICES='0 1')."""
    return value.split() if value else []


def role_args() -> list[str]:
    roles = env("ROLES")
    if roles:
        return ["--roles", *split_args(roles)]
    roles_file = env("ROLES_FILE", str(PIPE / "role_groups.json"))
    args: list[str] = []
    if roles_file and Path(roles_file).exists():
        args = ["--roles_file", roles_file]
        if env("GROUPS"):
            args += ["--groups", *split_args(env("GROUPS"))]
    if env("SHARD"):
        args += ["--shard", env("SHARD")]
    return args


def run(py: str, script: Path, args: list[str]) -> None:
    cmd = [py, str(script), *args]
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"stage {script.name} exited {r.returncode}")


def main() -> None:
    py = find_python()
    if not (PIPE / ".env").exists():
        raise SystemExit(
            f"missing {PIPE / '.env'} -- copy {PIPE / 'env.aa.example'} to .env and fill "
            "NDIF_API_KEY + HF_TOKEN"
        )

    out = env("OUT_DIR")
    if not out:
        raise SystemExit("set OUT_DIR to an output directory (needs ~1 GB free)")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = env("MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    judge = env("JUDGE_MODEL", "meta-llama/Llama-3.1-70B-Instruct")
    layers = env("LAYERS", "all")
    qc = env("QUESTION_COUNT", "240")
    gen_batch = env("GEN_BATCH", "24")
    judge_batch = env("JUDGE_BATCH", "48")
    act_batch = env("ACT_BATCH", "8")
    gen_sess = env("GEN_SESSION_BATCHES", "8")
    judge_sess = env("JUDGE_SESSION_BATCHES", "8")
    act_sess = env("ACT_SESSION_BATCHES", "20")
    min_count = env("MIN_COUNT", "50")
    only = {s.strip() for s in env("ONLY", "1,3,2,5,6").split(",") if s.strip()}

    roles = role_args()
    pi = ["--prompt_indices", *split_args(env("PROMPT_INDICES"))] if env("PROMPT_INDICES") else []

    print("=== Assistant Axis / NDIF ===")
    print(f"python={py}")
    print(f"model={model}  judge={judge}")
    print(f"out={out_dir}  layers={layers}  question_count={qc}  shard={env('SHARD', 'none')}")
    print(f"roles={env('ROLES') or (env('GROUPS') or Path(env('ROLES_FILE', 'role_groups.json')).name)}")

    if "1" in only:
        print("\n=== step 1/5: generate (NDIF) ===")
        run(py, PIPE / "1_generate_ndif.py",
            ["--model", model, "--output_dir", str(out_dir / "responses"),
             "--question_count", qc, "--gen_batch", gen_batch,
             "--session_batches", gen_sess, *pi, *roles])

    if "3" in only:
        print("\n=== step 2/5: judge (NDIF) ===")
        run(py, PIPE / "3_judge_ndif.py",
            ["--responses_dir", str(out_dir / "responses"), "--roles_dir", str(ROLES_DIR),
             "--output_dir", str(out_dir / "scores"), "--judge_model", judge,
             "--judge_batch", judge_batch, "--session_batches", judge_sess, *roles])

    if "2" in only:
        print("\n=== step 3/5: per-role persona vectors (NDIF) ===")
        run(py, PIPE / "2_activations_ndif.py",
            ["--model", model, "--responses_dir", str(out_dir / "responses"),
             "--scores_dir", str(out_dir / "scores"), "--output_dir", str(out_dir / "vectors"),
             "--batch_size", act_batch, "--min_count", min_count,
             "--layers", *split_args(layers), "--session_batches", act_sess, *roles])

    if "5" in only:
        print("\n=== step 4/5: assistant axis (local, upstream) ===")
        run(py, AA_PIPE / "5_axis.py",
            ["--vectors_dir", str(out_dir / "vectors"), "--output", str(out_dir / "axis.pt")])

    if "6" in only:
        print("\n=== step 5/5: axes + PCA + validation (local) ===")
        args = ["--vectors_dir", str(out_dir / "vectors"), "--output", str(out_dir / "axes.pt")]
        if env("ROLES_FILE", str(PIPE / "role_groups.json")):
            args += ["--roles_file", env("ROLES_FILE", str(PIPE / "role_groups.json"))]
        run(py, PIPE / "6_axes.py", args)

    print("\n=== done ===")
    print(f"persona vectors : {out_dir / 'vectors'}/*.pt")
    print(f"assistant axis  : {out_dir / 'axis.pt'}")
    print(f"axes + PCA      : {out_dir / 'axes.pt'}")


if __name__ == "__main__":
    sys.exit(main())
