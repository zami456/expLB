"""Insert-only check (P17): every line of SNAPSHOT must still be present in FILE, in order.

Usage: python audit/scripts/p17_check_insert_only.py SNAPSHOT FILE
Prints "OK <file> (<n> snapshot lines, <m> file lines, <m-n> inserted)" or the first snapshot line
that cannot be matched in order. Lines are compared byte-for-byte, line endings included.
"""
import sys


def main() -> int:
    snap_path, file_path = sys.argv[1], sys.argv[2]
    snap = open(snap_path, "rb").read().splitlines(keepends=True)
    cur = open(file_path, "rb").read().splitlines(keepends=True)
    j = 0
    for i, line in enumerate(snap, 1):
        while j < len(cur) and cur[j] != line:
            j += 1
        if j == len(cur):
            print(f"FAIL {file_path}: snapshot line {i} not found in order: "
                  f"{line.decode('utf-8', 'replace').rstrip()[:100]!r}")
            return 1
        j += 1
    print(f"OK {file_path} ({len(snap)} snapshot lines, {len(cur)} file lines, "
          f"{len(cur) - len(snap)} inserted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
