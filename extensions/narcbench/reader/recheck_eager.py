"""verdict_read.py's recheck, unchanged, with each saved .npz read once (2026-10-23).

cmd_recheck takes z["points"][i] row by row, and an NpzFile reads the whole member again on every access; each saved
row keeps its own copy of the file's array alive, so a batched file of n rows costs n copies (qwen_lb_five: 228-637
rows per file; out of memory 7 times). Here np.load hands it each file's arrays read once. The reader file, the rows it
reads again and the comparison are unchanged.

    python -I recheck_eager.py recheck --model qwen --name qwen_lb_five
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict_read as VR  # noqa: E402

_load = np.load


class _Loaded(dict):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _load_once(file, *args, **kwargs):
    z = _load(file, *args, **kwargs)
    if not isinstance(z, np.lib.npyio.NpzFile):
        return z
    with z:
        return _Loaded({k: z[k] for k in z.files})


if __name__ == "__main__":
    np.load = _load_once
    sys.argv = [VR.__file__, *sys.argv[1:]]
    sys.exit(VR.main())
