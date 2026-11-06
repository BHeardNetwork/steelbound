"""Run the Gen 1 bloodline step — teacher curriculum v1.

The PARENT produces the child: Gen 1 starts from Gen 0's own weights
(copy, never a third-party model) with exactly one recorded variation:

    {"kind": "teacher_curriculum_v1",
     "note": "first teacher-authored structured curriculum;
              teacher provenance recorded"}

Training data: Steelbound's own corpus (frozen stable_split protocol,
same held-out ground as Gen 0) PLUS the teacher curriculum rendered by
training/build_teacher_curriculum.py from training/teacher_lessons.jsonl
(teacher-authored DATA, provenance below). New words in the lessons
grow the child's vocabulary (parent itos + new tokens appended, the
evolve_generation new_tokens path); the val sequences stay
parent-encoded so parent and child are judged on identical ground.

Judging: head-to-head on held-out perplexity, the core DEFAULT_PROBES,
and the teacher-probe set (the lessons' probe fields, reported
separately), with bits-per-token (Synthesis S1) for both — all under
the existing no-regression compare() rule with the teacher score as
the declared target axis. Promotion happens ONLY on a
promotion_worthy verdict (owner ordered this run): it calls
evolution.promote_generation and both the evolution and the promotion
are ledgered (MODEL_EVOLVED / MODEL_PROMOTED) in the core's ledger.
Any other verdict keeps Gen 1 as an unpromoted candidate checkpoint.

Usage: PYTHONPATH=src:training python3 training/run_gen1_teacher.py [--epochs N]
Outputs: src/steelbound/model_data/gen1/ and training/gen1_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from build_teacher_curriculum import (  # noqa: E402
    load_lessons, provenance, render_sequences, teacher_probes)
from corpus import curriculum_sequences, stable_split, unique_sequences  # noqa: E402
from steelbound.config import Core, Settings  # noqa: E402
from steelbound.evolution import evolve_generation, promote_generation  # noqa: E402
from steelbound.model import SteelboundModel, Vocab, sha256_text  # noqa: E402

VARIATION = {"kind": "teacher_curriculum_v1",
             "note": "first teacher-authored structured curriculum; "
                     "teacher provenance recorded"}


def probe_details(model, probes) -> dict:
    passed, details = 0, []
    for prompt, expected in probes:
        out = model.generate(prompt, max_tokens=24)
        ok = expected.lower() in out.lower()
        passed += int(ok)
        details.append({"prompt": prompt, "expected": expected,
                        "output": out, "passed": ok})
    return {"passed": passed, "total": len(probes), "details": details}


def slim_metrics(m: dict) -> dict:
    return {k: m[k] for k in ("val_loss", "val_perplexity", "bits_per_token",
                              "probe_passed", "probe_total", "probe_score")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--teacher-repeat", type=int, default=6)
    ap.add_argument("--own-replay", type=int, default=2)
    ap.add_argument("--parent", default=str(ROOT / "src" / "steelbound" / "model_data" / "gen0"))
    ap.add_argument("--out", default=str(ROOT / "src" / "steelbound" / "model_data" / "gen1"))
    ap.add_argument("--data-dir", default=str(ROOT / ".steelbound"))
    args = ap.parse_args()

    lessons = load_lessons()
    t_seqs = render_sequences(lessons)
    t_probes = teacher_probes(lessons)
    t_prov = provenance(lessons, t_seqs)

    unique = unique_sequences()
    curriculum = curriculum_sequences()
    own_hash = sha256_text("\n".join(" ".join(s) for s in unique))
    train_unique, val_seqs = stable_split(unique)

    parent = SteelboundModel.load_checkpoint(args.parent)
    new_tokens = sorted({t for seq in t_seqs for t in seq} - set(parent.vocab.itos))
    child_vocab = Vocab(parent.vocab.itos + new_tokens)

    own_seqs = train_unique + curriculum * 5
    train_mix = own_seqs * args.own_replay + t_seqs * args.teacher_repeat
    encode_child = lambda seqs: [child_vocab.encode(s) for s in seqs]  # noqa: E731
    encode_parent = lambda seqs: [parent.vocab.encode(s) for s in seqs]  # noqa: E731

    core = Core(Settings(data_dir=args.data_dir))
    try:
        report = evolve_generation(
            args.parent, encode_child(train_mix), encode_parent(val_seqs),
            args.out, variation=VARIATION, epochs=args.epochs, lr=args.lr,
            seed=11, ledger=core.ledger, new_tokens=new_tokens,
            target_probes=t_probes)
        child = SteelboundModel.load_checkpoint(args.out)
        out = {
            "generation": report["generation"],
            "variation": report["variation"],
            "verdict": report["verdict"],
            "promoted": False,
            "parent_weights_sha256": report["parent_weights_sha256"],
            "child_weights_sha256": report["child_weights_sha256"],
            "vocab_added_tokens": report["vocab_added_tokens"],
            "own_corpus_sha256": own_hash,
            "teacher_provenance": t_prov,
            "parent_metrics": slim_metrics(report["parent_metrics"]),
            "child_metrics": slim_metrics(report["child_metrics"]),
            "teacher_probes_parent": probe_details(parent, t_probes),
            "teacher_probes_child": probe_details(child, t_probes),
        }
        if report["verdict"] == "promotion_worthy":
            pointer = promote_generation(args.out, ledger=core.ledger)
            out["promoted"] = True
            out["promotion_pointer"] = pointer
    finally:
        core.close()
    (ROOT / "training" / "gen1_report.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("teacher_probes_parent", "teacher_probes_child")},
                     indent=2))
    print("teacher probes: parent "
          f"{out['teacher_probes_parent']['passed']}/{out['teacher_probes_parent']['total']}"
          " -> child "
          f"{out['teacher_probes_child']['passed']}/{out['teacher_probes_child']['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
