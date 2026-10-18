"""Run `question_bank.confirm` (the single Llama confirmation read) under a Python audit hook that records every
file the process opens, so the report can list each file read under the confirmation sets with its time.

    python deception_detection/scripts/questions_v2/confirm_audited.py --bank <bank chosen by verdict.md>

Writes results/question_bank/llama/confirm_files_opened.tsv (first open of each path, ISO time; Python's own
library files are skipped) next to confirm.csv. `confirm` itself refuses to run twice; so does this wrapper.
"""

from __future__ import annotations

import datetime
import os
import sys

import fire


def main(bank: str) -> None:
    from deception_detection.scripts import question_bank as QB

    if (QB.OUT / "confirm.csv").exists():
        raise SystemExit("confirm.csv exists: the confirmation read has been done; it is not repeated")
    log_path = QB.OUT / "confirm_files_opened.tsv"
    fh = open(log_path, "w", encoding="utf-8")  # opened before the hook is installed
    fh.write(f"# confirm --bank {bank}; started {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    fh.write("time\tconfirmation_set\tpath\n")
    skip = tuple(os.path.normcase(p) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix} if p)
    seen: set[str] = set()
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
                    low = key.replace("\\", "/")
                    flag = "yes" if ("/aisi_llama/" in low or "/gender_secret/" in low) else ""
                    fh.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\t{flag}\t{p}\n")
                    fh.flush()
        finally:
            busy[0] = False

    sys.addaudithook(hook)
    QB.confirm(bank)
    busy[0] = True
    fh.write(f"# finished {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    fh.close()


if __name__ == "__main__":
    fire.Fire(main)
