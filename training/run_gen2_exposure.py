"""Run the Gen 2 experiment — full-coverage teacher exposure.

Design: GEN2_DESIGN.md (goal workspace, prepared 2026-11-06).
Diagnosis under test: Gen 1's teacher recall stalled at 12/78 from
EXPOSURE STARVATION — its ~5% teacher share came from a strided
subset of 93 of the 471 rendered teacher sequences (only 20/78
probe Q&A pairings ever trained; 171 of 327 added teacher tokens
got zero gradient). Dose-response across all recorded runs is
monotonic in teacher share; silent growth already removed the
growth shock that originally justified rationing the share.

Recipe (one candidate, one variation — teacher exposure):
  * Parent: Gen 1 (silent growth), the LIVE model at
    model_data/gen1_silent, weights SHA-256 verified against the
    design record before anything trains; the run aborts if the
    hash does not match.
  * Silent growth, same step as Gen 1: hidden 224 -> 288 (+64),
    zero-init. Vocabulary growth +0 BY DESIGN — Gen 1 already
    admitted every teacher token; the runner verifies the
    teacher-only token set is empty and aborts if it is not.
  * THE VARIATION: the full 471-sequence rendered teacher set,
    repeated to ~25% of training examples
    (steelbound.gen2.build_teacher_exposure), replacing Gen 1's
    5% strided subset. Own-corpus replay unchanged (frozen
    stable_split train split + curriculum.txt x5).
  * Everything else at Gen 1 values: lr 0.0005, inherited-LR
    scale x0.25, batch 128, 6 epochs, 24 sub-epoch snapshots
    (4/epoch), A2 selection (multiverse.select_epoch_a2), seed 321.

Acceptance criteria (stated in the design BEFORE the run; ALL
four required for a PASS):
  G1 own-corpus val perplexity within the standing 2% band of the
     parent re-measured in-run on the same frozen ground;
  G2 core probes 10/10 (equal to parent);
  G3 teacher recall >= 30/78;
  G4 >= 10/46 combined on the teacher-only topics at zero today
     (arithmetic 30 + logic 8 + reasoning steps 8).
The standing multiverse.advancement() rule is also computed and
recorded; the verdict is PASS only if G1-G4 all pass.

Artefacts: src/steelbound/model_data/gen2_exposure/ — candidate
checkpoint under gen2/ plus manifest.json (recipe, ground hashes,
per-snapshot records, per-topic probe tables, criteria, verdict).
Runtime-ledger entries go to a scratch Core data dir under /tmp.

HARD: model_data/current.json is NEVER written; gen0 / gen1 /
gen1_silent / multiverse / silent_growth are never modified.
Gen 1 stays live whatever happens; a PASS makes the child a
promotion CANDIDATE for the owner's decision only.

Usage: PYTHONPATH=src:training python3 training/run_gen2_exposure.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from build_teacher_curriculum import load_lessons, provenance, render_sequences, teacher_probes  # noqa: E402
from corpus import curriculum_sequences, stable_split, unique_sequences  # noqa: E402
from steelbound.config import Core, Settings  # noqa: E402
from steelbound.evolution import evaluate_model  # noqa: E402
from steelbound.gen2 import TEACHER_TARGET_SHARE, build_teacher_exposure, example_count  # noqa: E402
from steelbound.model import SteelboundModel, Vocab, make_examples, sha256_text  # noqa: E402
from steelbound.multiverse import (  # noqa: E402
    STRATEGY_A2, _probe_fraction, advancement, build_lr_scales, select_epoch_a2)
from steelbound.silentgrowth import grow_silent, train_subepoch  # noqa: E402

SEED = 321
LR = 0.0005
EPOCHS = 6
SNAPSHOTS_PER_EPOCH = 4
HIDDEN_GROWTH = 64
INHERITED_LR_SCALE = 0.25
CURRICULUM_REPEAT = 5
TOLERANCE = 0.02

PARENT_DIR = ROOT / "src" / "steelbound" / "model_data" / "gen1_silent"
EXPECTED_PARENT_SHA256 = "20d038f52cc300559729cff021717884993392470fbb358b554478da5c759b3d"

# G4 substance guard: the teacher-only topics at 0 today for both
# generations (design §3): arithmetic 30 + logic 8 + reasoning 8.
G4_TOPICS = ("arithmetic sums", "arithmetic products",
             "arithmetic subtraction", "arithmetic comparisons",
             "logic", "reasoning steps")
G3_TARGET = 30
G4_TARGET = 10


def _slim(metrics: dict) -> dict:
    return {k: metrics[k] for k in ("val_loss", "bits_per_token", "val_perplexity",
                                    "probe_passed", "probe_total", "probe_score")}


def probe_report(model, lessons) -> dict:
    """Per-topic teacher probe results, judged exactly as
    multiverse._probe_fraction judges them (greedy generation,
    expected substring, <=24 tokens) — one generation per probe,
    grouped by the lesson's topic."""
    per_topic: dict[str, dict] = {}
    passed_total = 0
    total = 0
    for lesson in lessons:
        probe = lesson["probe"]
        out = model.generate(probe["prompt"], max_tokens=24)
        ok = probe["expected"].lower() in out.lower()
        slot = per_topic.setdefault(lesson["topic"], {"passed": 0, "total": 0})
        slot["total"] += 1
        slot["passed"] += int(ok)
        passed_total += int(ok)
        total += 1
    return {"passed": passed_total, "total": total, "per_topic": per_topic}


def g4_count(report: dict) -> tuple[int, int]:
    passed = sum(report["per_topic"].get(t, {"passed": 0})["passed"]
                 for t in G4_TOPICS)
    total = sum(report["per_topic"].get(t, {"total": 0})["total"]
                for t in G4_TOPICS)
    return passed, total


def save_and_place(child, cand_dir: Path, generation: int, training: dict) -> dict:
    """Save to a /tmp staging dir first, then copy into place with
    retries and a hash-verified load check (mirrors the Gen 1
    runner — a transient write failure must not lose a trained
    candidate)."""
    staging = Path("/tmp/sb-gen2-staging/gen2")
    shutil.rmtree(staging.parent, ignore_errors=True)
    manifest = child.save_checkpoint(str(staging), generation=generation,
                                     training=training)
    last_exc = None
    for attempt in range(10):
        try:
            if cand_dir.exists():
                shutil.rmtree(cand_dir)
            shutil.copytree(staging, cand_dir)
            SteelboundModel.load_checkpoint(str(cand_dir))  # hash-verifies
            return manifest
        except Exception as exc:  # transient write/placement failure
            last_exc = exc
            print(f"checkpoint placement attempt {attempt + 1} failed: "
                  f"{exc!r}; retrying", flush=True)
            time.sleep(3)
    raise RuntimeError(f"could not place checkpoint at {cand_dir}") from last_exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--out", default=str(
        ROOT / "src" / "steelbound" / "model_data" / "gen2_exposure"))
    ap.add_argument("--data-dir", default="/tmp/sb-gen2-ledger")
    args = ap.parse_args()

    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    out_root = Path(args.out)
    cand_dir = out_root / "gen2"
    os.makedirs(out_root, exist_ok=True)

    # ---- Parent: verify before anything else -------------------------
    file_sha = hashlib.sha256(
        (PARENT_DIR / "weights.npz").read_bytes()).hexdigest()
    parent = SteelboundModel.load_checkpoint(str(PARENT_DIR))
    manifest_sha = parent.manifest["weights_sha256"]
    if file_sha != EXPECTED_PARENT_SHA256 or manifest_sha != EXPECTED_PARENT_SHA256:
        print(f"ABORT: parent weights hash mismatch — file {file_sha}, "
              f"manifest {manifest_sha}, expected {EXPECTED_PARENT_SHA256}",
              flush=True)
        return 2
    print(f"parent verified: gen1_silent weights sha256 {file_sha}", flush=True)

    # ---- Ground ------------------------------------------------------
    lessons = load_lessons()
    t_seqs_full = render_sequences(lessons)
    t_probes = teacher_probes(lessons)
    t_prov = provenance(lessons, t_seqs_full)

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

    parent_vocab_size = len(parent.vocab)
    parent_hidden = parent.hidden_dim
    X_val, y_val = make_examples([parent.vocab.encode(s) for s in val_seqs],
                                 parent.context_len)
    parent_metrics = evaluate_model(parent, X_val, y_val)
    parent_target = _probe_fraction(parent, t_probes)
    parent_report = probe_report(parent, lessons)
    parent_g4, g4_total = g4_count(parent_report)
    # Secondary readout (design §3): teacher-set perplexity — parent
    # and child share one vocabulary in Gen 2, so the encoding is
    # identical for both (the like-for-like control).
    X_t, y_t = make_examples([parent.vocab.encode(s) for s in t_seqs_full],
                             parent.context_len)
    parent_teacher_ppl = float(parent.perplexity(X_t, y_t))
    print(f"Gen 1 re-measured: perplexity {parent_metrics['val_perplexity']:.2f}, "
          f"bits/token {parent_metrics['bits_per_token']:.3f}, "
          f"core {parent_metrics['probe_passed']}/{parent_metrics['probe_total']}, "
          f"teacher {parent_report['passed']}/{parent_report['total']}, "
          f"G4 topics {parent_g4}/{g4_total}, "
          f"teacher-set perplexity {parent_teacher_ppl:.2f}", flush=True)

    # ---- Design premise: vocabulary growth +0 -------------------------
    new_tokens = sorted({t for seq in t_seqs_full for t in seq}
                        - set(parent.vocab.itos))
    if new_tokens:
        print(f"ABORT: design premise violated — {len(new_tokens)} teacher "
              f"tokens are not in the Gen 1 vocabulary: {new_tokens[:10]}",
              flush=True)
        return 3
    print("vocabulary premise verified: 0 teacher tokens missing "
          "from the Gen 1 vocabulary (growth +0)", flush=True)

    # ---- Mix: own replay + full-coverage teacher at ~25% --------------
    base_examples = (example_count(train_seqs)
                     + CURRICULUM_REPEAT * example_count(curriculum))
    teacher_block, exposure = build_teacher_exposure(
        t_seqs_full, base_examples, TEACHER_TARGET_SHARE)
    exposure["teacher_share_sequences"] = (
        exposure["teacher_sequences_with_repeats"]
        / (len(train_seqs) + CURRICULUM_REPEAT * len(curriculum)
           + exposure["teacher_sequences_with_repeats"]))
    total_examples = base_examples + exposure["teacher_examples"]
    print(f"teacher mix: full set x{exposure['full_repeats']} + strided "
          f"top-up {exposure['topup_count']} = "
          f"{exposure['teacher_sequences_with_repeats']} sequences "
          f"(coverage {exposure['coverage_fraction']:.0%} of "
          f"{len(t_seqs_full)}), share by examples "
          f"{exposure['teacher_share_examples']:.4f} "
          f"(target {TEACHER_TARGET_SHARE}), by sequences "
          f"{exposure['teacher_share_sequences']:.4f}, "
          f"total train examples {total_examples}", flush=True)

    mix = train_seqs + curriculum * CURRICULUM_REPEAT + teacher_block

    variation = {
        "kind": "teacher_exposure_full_coverage_25pct",
        "strategy": STRATEGY_A2,
        "growth": "silent_zero_init",
        "hidden_growth": HIDDEN_GROWTH,
        "vocab_growth": 0,
        "teacher_full_count": len(t_seqs_full),
        "teacher_full_repeats": exposure["full_repeats"],
        "teacher_topup_count": exposure["topup_count"],
        "teacher_target_share_examples": TEACHER_TARGET_SHARE,
        "teacher_share_examples": exposure["teacher_share_examples"],
        "teacher_share_sequences": exposure["teacher_share_sequences"],
        "curriculum_repeat": CURRICULUM_REPEAT,
        "inherited_lr_scale": INHERITED_LR_SCALE,
        "lr": args.lr,
        "teacher_authored": True,
        "design": "GEN2_DESIGN.md (goal workspace files)",
    }

    # ---- Child: parent weights + silent growth (hidden only) ----------
    child_vocab = Vocab(parent.vocab.itos)  # grow_silent replaces it
    child = SteelboundModel(child_vocab, context_len=parent.context_len,
                            embed_dim=parent.embed_dim,
                            hidden_dim=parent.hidden_dim, seed=args.seed)
    for name in ("E", "W1", "b1", "W2", "b2"):
        setattr(child, name, getattr(parent, name).copy())
    child._adam = None
    added_tokens, added_units = grow_silent(child, [], HIDDEN_GROWTH,
                                            seed=args.seed)
    print(f"grown silently: +{len(added_tokens)} tokens, +{added_units} "
          f"hidden units, params {child.param_count()}", flush=True)

    # Step-0 check: with vocab +0 and zero-init W2 rows for the new
    # hidden units, the child's logits are exactly the parent's —
    # raw metrics should be bit-for-bit the parent's (stronger than
    # the renormalized invariant Gen 1 needed).
    step0_metrics = evaluate_model(child, X_val, y_val)
    step0_target = _probe_fraction(child, t_probes)
    print(f"step 0 after silent growth (raw): perplexity "
          f"{step0_metrics['val_perplexity']:.4f} vs parent "
          f"{parent_metrics['val_perplexity']:.4f}; core "
          f"{step0_metrics['probe_passed']}/{step0_metrics['probe_total']}, "
          f"teacher {step0_target}", flush=True)

    lr_scales = build_lr_scales(child, parent_vocab_size, parent_hidden,
                                INHERITED_LR_SCALE)
    X_train, y_train = make_examples([child.vocab.encode(s) for s in mix],
                                     parent.context_len)
    print(f"training examples encoded: {len(X_train)} "
          f"(teacher share by encoded examples "
          f"{exposure['teacher_examples'] / len(X_train):.4f})", flush=True)

    core = Core(Settings(data_dir=args.data_dir))
    try:
        def _log(rec):
            print(f"  snapshot epoch {rec['epoch']:.2f}: val_loss "
                  f"{rec['val_loss']:.4f} core {rec['core_score']} "
                  f"target {rec['target_score']}", flush=True)

        records, snapshots = train_subepoch(
            child, X_train, y_train, X_val, y_val,
            epochs=args.epochs, lr=args.lr, seed=args.seed, batch=128,
            lr_scales=lr_scales, target_probes=t_probes,
            snapshots_per_epoch=SNAPSHOTS_PER_EPOCH, log=_log)

        chosen, selection_info = select_epoch_a2(records)
        for name, arr in snapshots[chosen]["params"].items():
            setattr(child, name, arr)
        child._adam = None
        print(f"selected snapshot: index {chosen}, epoch "
              f"{records[chosen]['epoch']}, selection {selection_info}", flush=True)

        generation = int(parent.manifest.get("generation", 0)) + 1
        ckpt_manifest = save_and_place(child, cand_dir, generation,
            training={
                "method": "gen2_exposure_candidate_from_parent",
                "variation": variation,
                "strategy": STRATEGY_A2,
                "selection": {**selection_info,
                              "selected_epoch": records[chosen]["epoch"],
                              "selected_index": chosen},
                "vocab_added_tokens": added_tokens,
                "parent_dir": os.path.basename(os.path.normpath(str(PARENT_DIR))),
                "parent_weights_sha256": parent.manifest["weights_sha256"],
                "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
                "hidden_growth": int(HIDDEN_GROWTH),
                "inherited_lr_scale": INHERITED_LR_SCALE,
                "snapshots_per_epoch": SNAPSHOTS_PER_EPOCH,
                "train_examples": int(len(X_train)),
                "teacher_share_examples": exposure["teacher_share_examples"],
                "teacher_share_sequences": exposure["teacher_share_sequences"],
                "snapshot_records": records,
            })
        child_metrics = evaluate_model(child, X_val, y_val)
        child_target = _probe_fraction(child, t_probes)
        child_report = probe_report(child, lessons)
        child_g4, _ = g4_count(child_report)
        child_teacher_ppl = float(child.perplexity(X_t, y_t))
        advanced, reasons = advancement(parent_metrics, child_metrics,
                                        parent_target, child_target, TOLERANCE)

        # ---- The design's criteria, scored exactly as predefined ------
        ppl_band = parent_metrics["val_perplexity"] * (1.0 + TOLERANCE)
        criteria = {
            "G1_no_regression": {
                "threshold": f"child perplexity <= parent x {1.0 + TOLERANCE:.2f} "
                             f"({ppl_band:.2f})",
                "parent": parent_metrics["val_perplexity"],
                "child": child_metrics["val_perplexity"],
                "pass": child_metrics["val_perplexity"] <= ppl_band,
            },
            "G2_core_probes": {
                "threshold": "10/10, equal to parent",
                "parent": f"{parent_metrics['probe_passed']}/{parent_metrics['probe_total']}",
                "child": f"{child_metrics['probe_passed']}/{child_metrics['probe_total']}",
                "pass": (child_metrics["probe_passed"] == child_metrics["probe_total"]
                         and child_metrics["probe_passed"]
                         >= parent_metrics["probe_passed"]),
            },
            "G3_teacher_recall": {
                "threshold": f">= {G3_TARGET}/{len(t_probes)}",
                "parent": parent_report["passed"],
                "child": child_report["passed"],
                "pass": child_report["passed"] >= G3_TARGET,
            },
            "G4_substance_guard": {
                "threshold": f">= {G4_TARGET}/{g4_total} on arithmetic+logic+reasoning",
                "parent": parent_g4,
                "child": child_g4,
                "pass": child_g4 >= G4_TARGET,
            },
        }
        passed_all = all(c["pass"] for c in criteria.values())
        verdict = "PASS" if passed_all else "REFUSED"
        failed = [k for k, c in criteria.items() if not c["pass"]]
        seconds = round(time.time() - started, 1)
        print(f"candidate: perplexity {child_metrics['val_perplexity']:.2f} "
              f"(parent {parent_metrics['val_perplexity']:.2f}, band {ppl_band:.2f}), "
              f"core {child_metrics['probe_passed']}/{child_metrics['probe_total']}, "
              f"teacher {child_report['passed']}/{child_report['total']} "
              f"(parent {parent_report['passed']}), G4 {child_g4}/{g4_total} "
              f"(parent {parent_g4}), teacher-set perplexity "
              f"{child_teacher_ppl:.2f} (parent {parent_teacher_ppl:.2f}) "
              f"— {verdict} (failed: {failed or 'none'}; standing-rule "
              f"advancement: {'ADVANCES' if advanced else 'REFUSED'} {reasons})",
              flush=True)

        manifest = {
            "experiment": "gen2_exposure",
            "design": "GEN2_DESIGN.md (goal workspace files)",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "seconds": seconds,
            "recipe": {
                **variation,
                "epochs": args.epochs,
                "snapshots_per_epoch": SNAPSHOTS_PER_EPOCH,
                "selection_points": len(records),
                "seed": args.seed,
                "batch": 128,
                "tolerance": TOLERANCE,
                "train_sequences_total": len(mix),
                "train_examples_total": int(len(X_train)),
                "base_examples_own_plus_curriculum": base_examples,
                "teacher_examples_included": exposure["teacher_examples"],
                "teacher_exposure": exposure,
            },
            "ground": ground,
            "parent": {
                "checkpoint": str(PARENT_DIR),
                "generation": int(parent.manifest.get("generation", 0)),
                "weights_sha256": parent.manifest["weights_sha256"],
                "weights_sha256_verified_before_training": True,
                "metrics_on_run_ground": _slim(parent_metrics),
                "target_score_on_run_ground": parent_target,
                "teacher_passed": parent_report["passed"],
                "teacher_per_topic": parent_report["per_topic"],
                "g4_passed": parent_g4,
                "g4_total": g4_total,
                "teacher_set_perplexity": parent_teacher_ppl,
            },
            "step0_after_silent_growth": {
                "metrics_raw": _slim(step0_metrics),
                "target_score_raw": step0_target,
                "note": ("Vocab growth is +0 and the new hidden units' "
                         "W2 rows are zero, so step-0 logits are exactly "
                         "the parent's (not merely renormalized-equal)."),
            },
            "candidate": {
                "id": "gen2_exposure-g2",
                "generation": generation,
                "checkpoint": os.path.relpath(str(cand_dir), str(out_root)),
                "selected_snapshot_index": chosen,
                "selected_epoch": records[chosen]["epoch"],
                "selection": selection_info,
                "snapshot_records": records,
                "parent_metrics": _slim(parent_metrics),
                "child_metrics": _slim(child_metrics),
                "parent_target_score": parent_target,
                "child_target_score": child_target,
                "teacher_per_topic": child_report["per_topic"],
                "teacher_set_perplexity_parent": parent_teacher_ppl,
                "teacher_set_perplexity_child": child_teacher_ppl,
                "perplexity_delta_pct": round(
                    100.0 * (child_metrics["val_perplexity"]
                             - parent_metrics["val_perplexity"])
                    / parent_metrics["val_perplexity"], 2),
                "criteria": criteria,
                "standing_rule_advancement": {
                    "advanced": bool(advanced), "reasons": reasons},
                "verdict": verdict,
                "promotion_candidate": bool(passed_all),
                "failed_criteria": failed,
                "parent_weights_sha256": parent.manifest["weights_sha256"],
                "child_weights_sha256": ckpt_manifest["weights_sha256"],
                "vocab_added_tokens": added_tokens,
                "vocab_added_count": len(added_tokens),
                "hidden_growth": int(HIDDEN_GROWTH),
                "param_count": ckpt_manifest["architecture"]["param_count"],
            },
            "note": ("Promotion is the owner's decision. This experiment "
                     "never writes model_data/current.json; Gen 1 remains "
                     "live; a PASS records the child as a promotion "
                     "CANDIDATE only, a REFUSED child is preserved as-is."),
        }
        manifest_path = out_root / "manifest.json"
        tmp = str(manifest_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        os.replace(tmp, manifest_path)

        core.ledger.append("evolution", "MODEL_EVOLVED", {
            "generation": generation, "experiment": "gen2_exposure",
            "strategy": STRATEGY_A2, "verdict": verdict.lower(),
            "variation": variation,
            "parent_perplexity": parent_metrics["val_perplexity"],
            "child_perplexity": child_metrics["val_perplexity"],
            "parent_probe_score": parent_metrics["probe_score"],
            "child_probe_score": child_metrics["probe_score"],
            "parent_teacher_passed": parent_report["passed"],
            "child_teacher_passed": child_report["passed"],
            "child_weights_sha256": ckpt_manifest["weights_sha256"],
        })
        core.ledger.append("evolution", "GEN2_EXPOSURE_RUN", {
            "verdict": verdict,
            "promotion_candidate": bool(passed_all),
            "failed_criteria": failed,
            "selected_epoch": records[chosen]["epoch"],
            "child_perplexity": child_metrics["val_perplexity"],
            "child_teacher_passed": child_report["passed"],
            "child_g4_passed": child_g4,
        })
    finally:
        core.close()
    print(json.dumps({"verdict": manifest["candidate"]["verdict"],
                      "promotion_candidate": manifest["candidate"]["promotion_candidate"],
                      "criteria": {k: c["pass"] for k, c in criteria.items()},
                      "child_metrics": manifest["candidate"]["child_metrics"],
                      "selected_epoch": manifest["candidate"]["selected_epoch"]},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
