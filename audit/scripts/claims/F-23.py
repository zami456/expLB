#!/usr/bin/env python3
"""AUDIT, READ-ONLY. F-23: what the replicated paper actually evaluates on.

Opens papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf during the audit
(BRIEF rule 8: do not summarise a paper you have not opened) and reports, from its own text:
  * every dataset name mentioned, with counts and page numbers
  * the contents of Table 1
  * every figure caption, so "Figure 7 ... [HPC excluded]" can be checked verbatim
  * whether harm-pressure-knowledge-report / knowledge-report appears at all
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pypdf import PdfReader

PDF = Path("papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf")

NAMES = [
    "HarmPressureChoice", "HarmPressureKnowledgeReport", "harm-pressure-choice",
    "harm-pressure-knowledge-report", "knowledge-report", "KnowledgeReport",
    "HarmPressure", "MASK", "Mask", "AILiar", "AI Liar", "ai_liar", "Sandbagging",
    "RolePlaying", "Roleplaying", "InsiderTrading", "Insider Trading",
    "ConvincingGame", "Convincing Game", "InstructedDeception", "Instructed Deception",
    "Alpaca", "Liars' Bench", "Liars Bench", "LiarsBench", "kr-matched", "matched",
]


def main():
    r = PdfReader(str(PDF))
    pages = [p.extract_text() or "" for p in r.pages]
    print(f"PDF: {PDF}  pages={len(pages)}  bytes={PDF.stat().st_size}")
    print(f"title line: {pages[0].splitlines()[0][:120] if pages[0] else '(no text)'}")
    print()

    print("=== dataset-name mentions (count, pages) ===")
    for n in NAMES:
        hits = [i + 1 for i, t in enumerate(pages) if n in t]
        if hits:
            c = sum(t.count(n) for t in pages)
            print(f"  {n:<32} count={c:<4} pages={hits}")
    print()

    print("=== every line containing 'Table 1' or starting a table ===")
    for i, t in enumerate(pages):
        for line in t.splitlines():
            if re.search(r"Table\s*1\b", line):
                print(f"  p{i+1}: {line.strip()[:200]}")
    print()

    print("=== figure captions ===")
    for i, t in enumerate(pages):
        lines = t.splitlines()
        for j, line in enumerate(lines):
            if re.match(r"\s*Figure\s+\d+", line):
                cap = " ".join(lines[j:j + 4]).strip()
                print(f"  p{i+1}: {cap[:400]}")
    print()

    print("=== lines containing 'excluded' or 'HPC' ===")
    for i, t in enumerate(pages):
        for line in t.splitlines():
            if re.search(r"excluded|HPC|exclud", line, re.I):
                print(f"  p{i+1}: {line.strip()[:250]}")
    print()

    print("=== page holding Table 1 (raw text dump) ===")
    for i, t in enumerate(pages):
        if re.search(r"Table\s*1[:.]", t):
            print(f"--- page {i+1} ---")
            print(t[:3500])
            break


if __name__ == "__main__":
    main()
