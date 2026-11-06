"""Evolve Gen 1 from Gen 0 — the bloodline step.

The PARENT produces the child: Gen 1 starts from Gen 0's own weights
(copy, never a third-party model), with one recorded variation —
Gen 0's vocabulary grows by the new words in teacher lesson pack v1,
and the child trains on Steelbound's own corpus plus that pack.

Teacher pack provenance: training/lessons_teacher_v1.txt was authored
by Muse (the teacher) at the owner's direction, as training DATA.
Its SHA-256 is recorded in the child's manifest and in
training/gen1_report.json. Steelbound was taught these lessons; it
did not originate them, and its own curriculum says so.

Judging: head-to-head against Gen 0 on the same held-out own-corpus
split (seed 7, as train_gen0.py) and the same core probes, under the
no-regression rule — plus a separate lesson-probe score for parent
and child, so what the teaching actually added is measured, not
assumed. Promotion is a separate explicit step (owner path).

Usage: PYTHONPATH=src:training python3 training/evolve_gen1.py [--epochs N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from corpus import curriculum_sequences, stable_split, unique_sequences  # noqa: E402
from steelbound.evolution import evolve_generation  # noqa: E402
from steelbound.model import SteelboundModel, Vocab, tokenize  # noqa: E402

LESSON_PROBES = [
    ("what is two plus two :", "four"),
    ("what is seven plus eight :", "fifteen"),
    ("what is three times three :", "nine"),
    ("what is a fact :", "checked"),
    ("is a guess a fact :", "no"),
    ("what is profit :", "revenue"),
    ("how do you check work :", "compare"),
    ("who taught these lessons :", "teacher"),
    ("did steelbound originate these :", "no"),
    ("what is a customer :", "buys"),
]


def lesson_sequences() -> list[list[str]]:
    seqs = []
    for line in (ROOT / "training" / "lessons_teacher_v1.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = tokenize(line)
        if len(toks) >= 3:
            seqs.append(toks)
    return seqs


def lesson_score(model) -> dict:
    passed, details = 0, []
    for prompt, expected in LESSON_PROBES:
        out = model.generate(prompt, max_tokens=24)
        ok = expected.lower() in out.lower()
        passed += int(ok)
        details.append({"prompt": prompt, "expected": expected, "output": out, "passed": ok})
    return {"passed": passed, "total": len(LESSON_PROBES), "details": details}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--lesson-repeat", type=int, default=6)
    ap.add_argument("--own-replay", type=int, default=2,
                    help="replay factor for the parent's own corpus (anti-forgetting)")
    ap.add_argument("--hidden-growth", type=int, default=64,
                    help="new hidden units grown for the child (capacity growth)")
    ap.add_argument("--parent", default=str(ROOT / "src" / "steelbound" / "model_data" / "gen0"))
    ap.add_argument("--out", default=str(ROOT / "src" / "steelbound" / "model_data" / "gen1"))
    args = ap.parse_args()

    pack_path = ROOT / "training" / "lessons_teacher_v1.txt"
    pack_sha = hashlib.sha256(pack_path.read_bytes()).hexdigest()

    unique = unique_sequences()
    curriculum = curriculum_sequences()
    lessons = lesson_sequences()
    # Frozen, content-addressed split — the same ground Gen 0 was
    # trained and judged on, for every generation (see corpus.py).
    train_unique, val_unique = stable_split(unique)

    parent = SteelboundModel.load_checkpoint(args.parent)
    new_tokens = sorted({t for seq in lessons for t in seq} - set(parent.vocab.itos))
    child_vocab = Vocab(parent.vocab.itos + new_tokens)
    encode_child = lambda seqs: [child_vocab.encode(s) for s in seqs]  # noqa: E731
    encode_parent = lambda seqs: [parent.vocab.encode(s) for s in seqs]  # noqa: E731

    own_seqs = train_unique + curriculum * 5
    train_seqs = own_seqs * args.own_replay + lessons * args.lesson_repeat
    variation = {
        "kind": "vocab_and_capacity_growth_plus_teacher_lessons",
        "note": "gen1: parent weights + vocabulary grows by teacher-pack words + "
                "hidden layer grows (capacity for new knowledge without evicting "
                "the parent's); trained on own corpus (replayed, anti-forgetting) "
                "plus teacher lesson pack v1; epoch selection and judging include "
                "the lesson probes as the generation's declared target axis",
        "hidden_growth": args.hidden_growth,
        "teacher_pack": "training/lessons_teacher_v1.txt",
        "teacher_pack_sha256": pack_sha,
        "teacher_authored": True,
        "lesson_sequences": len(lessons),
        "lesson_repeat": args.lesson_repeat,
        "own_replay": args.own_replay,
    }
    report = evolve_generation(
        args.parent, encode_child(train_seqs), encode_parent(val_unique), args.out,
        variation=variation, epochs=args.epochs, lr=args.lr, seed=11,
        new_tokens=new_tokens, target_probes=LESSON_PROBES,
        hidden_growth=args.hidden_growth)

    child = SteelboundModel.load_checkpoint(args.out)
    slim = {k: report[k] for k in ("generation", "variation", "verdict",
                                   "child_weights_sha256", "parent_weights_sha256",
                                   "vocab_added_tokens", "hidden_growth",
                                   "parent_target_probe_score", "child_target_probe_score")}
    for side in ("parent_metrics", "child_metrics"):
        slim[side] = {k: report[side][k] for k in
                      ("val_loss", "val_perplexity", "probe_passed", "probe_total", "probe_score")}
    slim["lesson_probes_parent"] = lesson_score(parent)
    slim["lesson_probes_child"] = lesson_score(child)
    (ROOT / "training" / "gen1_report.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(json.dumps(slim, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
