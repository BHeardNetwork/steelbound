"""Run the multiverse evolution experiment — branching paths, Gen 1
through Gen 10, raced under the no-regression gate.

Owner direction 2026-11-04: "Run experiments, and add multiverse
path/action thru 10x generations."

Design (see src/steelbound/multiverse.py for the full rationale):
  * Root: Gen 0, the live model. Every candidate is produced by its
    parent — weights inherited, one recorded variation per branch
    recipe. No third-party weights or data anywhere: training data is
    Steelbound's own corpus (frozen stable_split ground) plus the
    existing teacher curriculum (training/teacher_lessons.jsonl,
    teacher-authored DATA, provenance recorded below).
  * Three paths race from Gen 0:
      a1_diff      — A1 two-stage epoch gate + differential learning
                     rates (inherited x0.25, grown units full rate)
      a2_diff      — A2 fraction-term epoch score + the SAME
                     differential-LR recipe as a1_diff, so the two
                     selection strategies race on identical footing
      a1_emphasis  — A1 gate + lesson-pack emphasis (teacher share
                     3x the diff paths') + larger capacity growth,
                     uniform rates
    Mix/rate calibration (from a 2026-11-04 smoke run at generation
    1, preserved in the report): at lr 0.002 with the teacher
    curriculum at ~40% of training sequences, both diff candidates
    regressed hard (perplexity 54.2 -> 90.9 / 104.5; emphasis path
    131.8) despite large lesson gains, and the per-epoch records show
    epoch 1 ALONE already cost +0.41 nats of val loss — Gen 0 sits at
    a sharp early-stopped optimum (its own training history rises
    monotonically after epoch 3), so no epoch-selection strategy can
    rescue a too-large step. The recipes below therefore (a) hold
    the teacher share near the best historical attempt's (~15-18% of
    examples: full own train split + curriculum x5, teacher x1) for
    the diff paths and 3x that for the emphasis path, and (b) train
    gently — lr 0.0005 over 6 epochs for the diff paths — so the
    trajectory has selection points fine-grained enough for A1/A2
    to find a within-band epoch if one exists. The gate itself is
    NOT loosened: advancement still uses the standing 2% band.
  * A candidate advances its path only on no-regression vs its own
    parent (perplexity within the 2% band, probes not worse).
    Refusals are preserved and end the path. At most 3 paths live.
  * NOTHING is promoted: model_data/current.json is never written.
    A final tip that strictly beats Gen 0 head-to-head is recorded
    in the manifest as a promotion CANDIDATE for the owner's say.

Artefacts: src/steelbound/model_data/multiverse/ (checkpoints +
manifest.json). Runtime-ledger entries (MODEL_EVOLVED per candidate,
MULTIVERSE_RUN at the end) go to a scratch Core data dir under /tmp;
the manifest + WORK_LEDGER are the durable record.

Usage: PYTHONPATH=src:training python3 training/run_multiverse.py
       [--generations N] [--epochs N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from build_teacher_curriculum import load_lessons, provenance, render_sequences, teacher_probes  # noqa: E402
from corpus import curriculum_sequences, stable_split, unique_sequences  # noqa: E402
from steelbound.config import Core, Settings  # noqa: E402
from steelbound.model import sha256_text  # noqa: E402
from steelbound.multiverse import (  # noqa: E402
    STRATEGY_A1, STRATEGY_A2, PathConfig, run_multiverse)

PATHS = [
    PathConfig(name="a1_diff", strategy=STRATEGY_A1, lr=0.0005,
               inherited_lr_scale=0.25, teacher_repeat=1, curriculum_repeat=5,
               hidden_growth_first=64, hidden_growth_later=16,
               epochs=6, seed_base=101),
    PathConfig(name="a2_diff", strategy=STRATEGY_A2, lr=0.0005,
               inherited_lr_scale=0.25, teacher_repeat=1, curriculum_repeat=5,
               hidden_growth_first=64, hidden_growth_later=16,
               epochs=6, seed_base=202),
    PathConfig(name="a1_emphasis", strategy=STRATEGY_A1, lr=0.001,
               inherited_lr_scale=None, teacher_repeat=3, curriculum_repeat=5,
               hidden_growth_first=96, hidden_growth_later=24,
               epochs=6, seed_base=303),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the per-candidate epoch cap for every path")
    ap.add_argument("--out", default=str(ROOT / "src" / "steelbound" / "model_data" / "multiverse"))
    ap.add_argument("--data-dir", default="/tmp/sb-mv-ledger")
    args = ap.parse_args()

    configs = PATHS
    if args.epochs is not None:
        for cfg in configs:
            cfg.epochs = args.epochs

    lessons = load_lessons()
    t_seqs = render_sequences(lessons)
    t_probes = teacher_probes(lessons)
    t_prov = provenance(lessons, t_seqs)

    unique = unique_sequences()
    curriculum = curriculum_sequences()
    train_seqs, val_seqs = stable_split(unique)
    ground = {
        "split": "training/corpus.py stable_split (frozen, content-addressed)",
        "unique_sequences": len(unique),
        "train_sequences": len(train_seqs),
        "val_sequences": len(val_seqs),
        "own_corpus_sha256": sha256_text("\n".join(" ".join(s) for s in unique)),
        "teacher_provenance": t_prov,
        "teacher_probes": len(t_probes),
    }
    print(json.dumps({"ground": ground}, indent=2), flush=True)

    core = Core(Settings(data_dir=args.data_dir))
    try:
        manifest = run_multiverse(
            root_checkpoint=str(ROOT / "src" / "steelbound" / "model_data" / "gen0"),
            out_root=args.out,
            own_train_seqs=train_seqs, curriculum_seqs=curriculum,
            teacher_seqs=t_seqs, val_seqs=val_seqs, target_probes=t_probes,
            path_configs=configs, max_generations=args.generations,
            ground_info=ground, ledger=core.ledger,
            log=lambda msg: print(msg, flush=True))
    finally:
        core.close()
    print(json.dumps({"final": manifest["final"],
                      "paths": manifest["paths"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
