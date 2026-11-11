"""Run a trajectory command under a Python audit hook that logs the first open of every file, flagging
confirmation data (spec.FORBIDDEN: AISI held-out, Gender-Secret, Aletheia test, NARCBench Transfer,
Stego and Verdict). Same mechanism as `questions_v2/audit_run.py`, with this study's flags.

    python deception_detection/scripts/trajectory/audit.py core.py gates
    python deception_detection/scripts/trajectory/audit.py lb.py fit

The log is `E:/trajectory/logs/files_opened.tsv` (appended, one header per command). The exit status
is the command's; a flagged open also makes it exit 3 after the command finishes.
"""

from __future__ import annotations

import datetime
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the repo root, so `deception_detection` imports
from deception_detection.scripts.trajectory import spec as T  # noqa: E402


def flagged(path: str) -> bool:
    low = os.path.normcase(path).replace("\\", "/")
    return any(os.path.normcase(f).replace("\\", "/") in low for f in T.FORBIDDEN)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    script, rest = HERE / sys.argv[1], sys.argv[2:]
    T.LOGS.mkdir(parents=True, exist_ok=True)
    fh = open(T.LOGS / "files_opened.tsv", "a", encoding="utf-8")
    fh.write(f"# {datetime.datetime.now().isoformat(timespec='seconds')} start: {script.name} {' '.join(rest)}\n")
    fh.write("time\tflagged\tpath\n")
    fh.flush()
    skip = tuple(os.path.normcase(p) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix} if p)
    seen: set[str] = set()
    counts = {"files": 0, "flagged": 0}
    busy = [False]

    def hook(event: str, args: tuple) -> None:
        if event != "open" or busy[0]:
            return
        busy[0] = True
        try:
            p = args[0] if args else None
            if isinstance(p, (str, bytes, os.PathLike)):
                p = os.path.abspath(os.fsdecode(p))
                key = os.path.normcase(p)
                if key not in seen and not key.startswith(skip):
                    seen.add(key)
                    flag = flagged(p)
                    counts["files"] += 1
                    counts["flagged"] += int(flag)
                    fh.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\t{'yes' if flag else ''}\t{p}\n")
                    fh.flush()
        finally:
            busy[0] = False

    sys.addaudithook(hook)
    sys.argv = [str(script)] + rest
    status = 0
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        status = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:  # noqa: BLE001
        import traceback

        busy[0] = True
        traceback.print_exc()
        busy[0] = False
        status = 1
    busy[0] = True
    fh.write(f"# {datetime.datetime.now().isoformat(timespec='seconds')} end (status {status}): "
             f"{counts['files']} files opened, {counts['flagged']} flagged\n")
    fh.close()
    if counts["flagged"]:
        print(f"AUDIT: {counts['flagged']} confirmation-data files opened (see files_opened.tsv)", file=sys.stderr)
        return status or 3
    return status


if __name__ == "__main__":
    sys.exit(main())
