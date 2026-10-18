"""Run a driver command under a Python audit hook that records every file the process opens (first open of each
path, ISO time), flagging confirmation-set files, so the report can show what development and the single
confirmation reads touched.

    python deception_detection/scripts/questions_v2/audit_run.py <log.tsv> <script.py> [script args...]

e.g.  audit_run.py results/question_bank/qwen35/dev_files_opened.tsv \
          deception_detection/scripts/aletheia/question_bank_aletheia.py dev

The log is appended to (one header per command). Flagged ("yes") paths: Llama confirmation sets
(activations/aisi_llama, activations/gender_secret) and Aletheia test data (data/labels/test_*, data/test/*,
acts/<model>/<read>/test__*). Python's own library files are skipped. The exit status is the script's.
"""

from __future__ import annotations

import datetime
import os
import runpy
import sys

FLAGS = ("/activations/aisi_llama/", "/activations/gender_secret/", "/aletheia/data/labels/test_",
         "/aletheia/data/test/")


def _flag(path: str) -> bool:
    low = os.path.normcase(path).replace("\\", "/")
    return any(f in low for f in FLAGS) or ("/aletheia/acts/" in low and "/test__" in low)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    log_path, script, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    fh = open(log_path, "a", encoding="utf-8")  # opened before the hook is installed
    fh.write(f"# {datetime.datetime.now().isoformat(timespec='seconds')} start: {script} {' '.join(rest)}\n")
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
                    flag = _flag(p)
                    counts["files"] += 1
                    counts["flagged"] += int(flag)
                    fh.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\t{'yes' if flag else ''}\t{p}\n")
                    fh.flush()
        finally:
            busy[0] = False

    sys.addaudithook(hook)
    sys.argv = [script] + rest
    status = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        status = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:  # noqa: BLE001 -- record the end line, then report the failure
        import traceback

        busy[0] = True
        traceback.print_exc()
        busy[0] = False
        status = 1
    busy[0] = True
    fh.write(f"# {datetime.datetime.now().isoformat(timespec='seconds')} end (status {status}): "
             f"{counts['files']} files opened, {counts['flagged']} flagged\n")
    fh.close()
    print(f"audit: {counts['files']} files opened, {counts['flagged']} flagged -> {log_path}", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
