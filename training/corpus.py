"""Build Steelbound's own training corpus from this repo only.

Sources (all authored for this project, clean-room):
  - training/curriculum.txt — one sequence per line, by design
  - README.md, CHANGELOG.md, WORK_LEDGER.md, docs/*.md — prose only:
    code fences, URLs, hashes, tables and markdown scaffolding stripped

Nothing external is fetched or copied. The corpus hash is recorded in
every checkpoint manifest/training report, so what a generation was
raised on is always auditable.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from steelbound.model import tokenize  # noqa: E402

_DOC_FILES = ["README.md", "CHANGELOG.md", "WORK_LEDGER.md"]


def _clean_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b[0-9a-f]{12,}\b", " ", text)          # commit/ledger hashes
    text = re.sub(r"^\s*\|.*$", " ", text, flags=re.M)       # tables
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[*_>\[\]()]", " ", text)
    return text


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def doc_sequences() -> list[list[str]]:
    texts = []
    for name in _DOC_FILES:
        texts.append((ROOT / name).read_text(encoding="utf-8"))
    for path in sorted((ROOT / "docs").glob("*.md")):
        texts.append(path.read_text(encoding="utf-8"))
    seqs = []
    for text in texts:
        for sent in _sentences(_clean_markdown(text)):
            toks = tokenize(sent)
            if 3 <= len(toks) <= 60:
                seqs.append(toks)
    return seqs


def curriculum_sequences() -> list[list[str]]:
    seqs = []
    for line in (ROOT / "training" / "curriculum.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = tokenize(line)
        if len(toks) >= 3:
            seqs.append(toks)
    return seqs


def unique_sequences() -> list[list[str]]:
    seen, out = set(), []
    for seq in doc_sequences() + curriculum_sequences():
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            out.append(seq)
    return out


def stable_split(seqs: list[list[str]], val_mod: int = 20):
    """Content-addressed train/val split: a sequence is validation
    iff sha256(its text) mod val_mod == 0 (~5%). The split depends
    only on the sequence itself — never on corpus order or size — so
    every generation of the bloodline is judged on the same ground
    even as the corpus grows. (An earlier positional split silently
    reshuffled whenever a doc changed; Gen 0 was retrained on this
    split and it is frozen from here on.)"""
    import hashlib
    train, val = [], []
    for seq in seqs:
        h = int(hashlib.sha256(" ".join(seq).encode("utf-8")).hexdigest(), 16)
        (val if h % val_mod == 0 else train).append(seq)
    return train, val
