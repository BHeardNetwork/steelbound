"""Run the silent-growth Gen 1 retry — one candidate, zero-init growth.

Owner-approved experiment following the multiverse null result
(2026-11-04, WORK_LEDGER 0039). Diagnosis under test: GROWTH SHOCK.
Multiverse children added new vocabulary / hidden capacity at
random init and started epoch 1 already ~+0.35 nats over the
parent, ~17x the width of the standing 2% no-regression band, so no
epoch-selection strategy ever had a within-band epoch to pick.

Fix under test (src/steelbound/silentgrowth.py): SILENT growth.
New capacity enters at zero — new embedding rows 0, new-token
output columns / biases 0, new hidden units' output rows 0 — so at
step 0 the child's logits over the parent's vocabulary are exactly
the parent's, and the renormalized next-token distribution over old
tokens is unchanged (proved in tests/test_silentgrowth.py). The new
capacity then earns its way in during training.

Recipe (single candidate — this is a Gen 1 retry, one child):
  * Parent: Gen 0 (the live model, untouched).
  * Silent growth: vocabulary += the teacher curriculum's new
    tokens (the 327 tokens in the rendered teacher sequences that
    Gen 0 does not know — verified and recorded in the manifest),
    hidden += 64.
  * Selection: A2 fraction-term rule (multiverse.select_epoch_a2 —
    the default the multiverse race recommended) over SUB-EPOCH
    snapshot records: quarter-epoch snapshots over 6 epochs give
    24 selection points.
  * Differential learning rates: inherited weights x0.25, grown
    capacity at the full rate, base lr 0.0005
    (multiverse.build_lr_scales).
  * Teacher share ~5%: full own train split + curriculum x5
    (the multiverse recipes' own-corpus replay), plus a
    deterministically strided subset of the 471 rendered teacher
    sequences sized so teacher EXAMPLES are ~5% of total training
    examples. The exact subset size and the actual shares (by
    examples and by sequences) are computed and recorded.

Ground: Steelbound's own corpus via training/corpus.py
unique_sequences + stable_split — the same frozen, content-addressed
ground the multiverse run used. The corpus grows as the project's
own docs / ledger grow, so the counts and SHA recorded in the
manifest are whatever this run actually gets, and Gen 0 is
re-measured in-run on that ground; every comparison is head-to-head
on the same ground.

Advancement: multiverse.advancement() semantics — the standing
no-regression rule, never loosened: own-corpus val perplexity
within the 2% band of Gen 0 (re-measured), core probes not worse,
teacher probes not worse.

Artefacts: src/steelbound/model_data/silent_growth/ — the candidate
checkpoint under gen1/ plus manifest.json (recipe, ground hashes,
per-snapshot records, selected snapshot, metrics, verdict).
Runtime-ledger entries go to a scratch Core data dir under /tmp.

HARD: model_data/current.json is NEVER written; gen0 / gen1 /
multiverse artefact dirs are never modified. A passing child is a
promotion CANDIDATE for the owner's decision only — this runner
does not promote.

Usage: PYTHONPATH=src:training python3 training/run_silent_growth.py
"""
from __future__ import annotations

import argparse
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
TEACHER_TARGET_SHARE = 0.05
TOLERANCE = 0.02


def _slim(metrics: dict) -> dict:
    return {k: metrics[k] for k in ("val_loss", "bits_per_token", "val_perplexity",
                                    "probe_passed", "probe_total", "probe_score")}


def _example_count(seqs) -> int:
    """make_examples yields len(seq)+1 examples per sequence."""
    return sum(len(s) + 1 for s in seqs)


def _strided(seqs, n: int):
    if n >= len(seqs):
        return list(seqs)
    step = len(seqs) / n
    return [seqs[int(i * step)] for i in range(n)]


def save_and_place(child, cand_dir: Path, generation: int, training: dict) -> dict:
    """Save the checkpoint to a /tmp staging dir first, then copy it
    into place with retries and a hash-verified load check. (A first
    run of this experiment trained cleanly but hit a transient
    PermissionError writing weights.npz directly into the workspace;
    staging + verified copy means a transient write failure can no
    longer lose a trained candidate.)"""
    staging = Path("/tmp/sb-sg-staging/gen1")
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


def choose_teacher_subset(teacher_seqs, base_examples: int,
                          target_share: float = TEACHER_TARGET_SHARE):
    """Strided subset of the teacher sequences whose EXAMPLE share
    of (base + subset) is closest to ``target_share``. Returns
    (subset, share_by_examples)."""
    best = None
    for n in range(1, len(teacher_seqs) + 1):
        sub = _strided(teacher_seqs, n)
        ex = _example_count(sub)
        share = ex / (base_examples + ex)
        if best is None or abs(share - target_share) < abs(best[1] - target_share):
            best = (sub, share)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--out", default=str(
        ROOT / "src" / "steelbound" / "model_data" / "silent_growth"))
    ap.add_argument("--data-dir", default="/tmp/sb-sg-ledger")
    args = ap.parse_args()

    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    out_root = Path(args.out)
    cand_dir = out_root / "gen1"
    os.makedirs(out_root, exist_ok=True)

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

    root_checkpoint = str(ROOT / "src" / "steelbound" / "model_data" / "gen0")
    parent = SteelboundModel.load_checkpoint(root_checkpoint)
    parent_vocab_size = len(parent.vocab)
    parent_hidden = parent.hidden_dim
    X_val, y_val = make_examples([parent.vocab.encode(s) for s in val_seqs],
                                 parent.context_len)
    parent_metrics = evaluate_model(parent, X_val, y_val)
    parent_target = _probe_fraction(parent, t_probes)
    print(f"Gen 0 re-measured: perplexity {parent_metrics['val_perplexity']:.2f}, "
          f"bits/token {parent_metrics['bits_per_token']:.3f}, "
          f"core {parent_metrics['probe_passed']}/{parent_metrics['probe_total']}, "
          f"teacher {parent_target} ({int(round((parent_target or 0) * len(t_probes)))}/{len(t_probes)})",
          flush=True)

    # ---- Mix: own replay + ~5% teacher --------------------------------
    base_seqs_count = len(train_seqs) + CURRICULUM_REPEAT * len(curriculum)
    base_examples = (_example_count(train_seqs)
                     + CURRICULUM_REPEAT * _example_count(curriculum))
    teacher_subset, teacher_share_ex = choose_teacher_subset(
        t_seqs_full, base_examples, TEACHER_TARGET_SHARE)
    teacher_share_seq = len(teacher_subset) / (base_seqs_count + len(teacher_subset))
    total_examples = base_examples + _example_count(teacher_subset)
    print(f"teacher mix: subset {len(teacher_subset)}/{len(t_seqs_full)} sequences "
          f"(strided), share by examples {teacher_share_ex:.4f}, "
          f"by sequences {teacher_share_seq:.4f}, total train examples {total_examples}",
          flush=True)

    mix = (train_seqs + curriculum * CURRICULUM_REPEAT + teacher_subset)

    # Vocabulary growth is the teacher curriculum's new tokens only
    # (teacher-only set; the multiverse narrative's 327 — verified here).
    new_tokens = sorted({t for seq in t_seqs_full for t in seq}
                        - set(parent.vocab.itos))
    print(f"teacher new tokens: {len(new_tokens)}", flush=True)

    variation = {
        "kind": "silent_zero_init_growth_plus_teacher_curriculum_5pct",
        "strategy": STRATEGY_A2,
        "growth": "silent_zero_init",
        "teacher_repeat_equivalent": round(len(teacher_subset) / len(t_seqs_full), 4),
        "teacher_subset_count": len(teacher_subset),
        "teacher_full_count": len(t_seqs_full),
        "curriculum_repeat": CURRICULUM_REPEAT,
        "hidden_growth": HIDDEN_GROWTH,
        "inherited_lr_scale": INHERITED_LR_SCALE,
        "lr": args.lr,
        "teacher_authored": True,
    }

    # ---- Child: parent weights + silent growth -------------------------
    child_vocab = Vocab(parent.vocab.itos)  # grow_silent replaces it
    child = SteelboundModel(child_vocab, context_len=parent.context_len,
                            embed_dim=parent.embed_dim,
                            hidden_dim=parent.hidden_dim, seed=args.seed)
    for name in ("E", "W1", "b1", "W2", "b2"):
        setattr(child, name, getattr(parent, name).copy())
    child._adam = None
    added_tokens, added_units = grow_silent(child, new_tokens, HIDDEN_GROWTH,
                                            seed=args.seed)
    print(f"grown silently: +{len(added_tokens)} tokens, +{added_units} hidden units, "
          f"params {child.param_count()}", flush=True)

    # Step-0 check on the frozen ground (raw metrics; the renormalized
    # invariance is proved in the unit tests — raw loss here includes
    # the new tokens' exp(0) softmax mass, recorded honestly).
    step0_metrics = evaluate_model(child, X_val, y_val)
    step0_target = _probe_fraction(child, t_probes)
    print(f"step 0 after silent growth (raw): perplexity "
          f"{step0_metrics['val_perplexity']:.2f} vs parent "
          f"{parent_metrics['val_perplexity']:.2f}; core "
          f"{step0_metrics['probe_passed']}/{step0_metrics['probe_total']}, "
          f"teacher {step0_target}", flush=True)

    lr_scales = build_lr_scales(child, parent_vocab_size, parent_hidden,
                                INHERITED_LR_SCALE)
    X_train, y_train = make_examples([child.vocab.encode(s) for s in mix],
                                     parent.context_len)
    print(f"training examples encoded: {len(X_train)} "
          f"(teacher share by encoded examples "
          f"{_example_count(teacher_subset) / len(X_train):.4f})", flush=True)

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
                "method": "silent_growth_candidate_from_parent",
                "variation": variation,
                "strategy": STRATEGY_A2,
                "selection": {**selection_info,
                              "selected_epoch": records[chosen]["epoch"],
                              "selected_index": chosen},
                "vocab_added_tokens": added_tokens,
                "parent_dir": os.path.basename(os.path.normpath(root_checkpoint)),
                "parent_weights_sha256": parent.manifest["weights_sha256"],
                "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
                "hidden_growth": int(HIDDEN_GROWTH),
                "inherited_lr_scale": INHERITED_LR_SCALE,
                "snapshots_per_epoch": SNAPSHOTS_PER_EPOCH,
                "train_examples": int(len(X_train)),
                "teacher_share_examples": teacher_share_ex,
                "teacher_share_sequences": teacher_share_seq,
                "snapshot_records": records,
            })
        child_metrics = evaluate_model(child, X_val, y_val)
        child_target = _probe_fraction(child, t_probes)
        advanced, reasons = advancement(parent_metrics, child_metrics,
                                        parent_target, child_target, TOLERANCE)
        verdict = "ADVANCES" if advanced else "REFUSED"
        seconds = round(time.time() - started, 1)
        print(f"candidate: perplexity {child_metrics['val_perplexity']:.2f} "
              f"(parent {parent_metrics['val_perplexity']:.2f}), "
              f"core {child_metrics['probe_passed']}/{child_metrics['probe_total']}, "
              f"teacher {child_target} — {verdict} {reasons}", flush=True)

        manifest = {
            "experiment": "silent_growth",
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
                "teacher_target_share_examples": TEACHER_TARGET_SHARE,
                "teacher_share_examples_actual": teacher_share_ex,
                "teacher_share_sequences_actual": teacher_share_seq,
                "train_sequences_total": len(mix),
                "train_examples_total": int(len(X_train)),
                "base_examples_own_plus_curriculum": base_examples,
                "teacher_examples_included": _example_count(teacher_subset),
            },
            "ground": ground,
            "root": {
                "checkpoint": root_checkpoint,
                "generation": int(parent.manifest.get("generation", 0)),
                "weights_sha256": parent.manifest["weights_sha256"],
                "metrics_on_run_ground": _slim(parent_metrics),
                "target_score_on_run_ground": parent_target,
            },
            "step0_after_silent_growth": {
                "metrics_raw": _slim(step0_metrics),
                "target_score_raw": step0_target,
                "note": ("Raw metrics include the new tokens' exp(0) "
                         "softmax mass; renormalized over the parent "
                         "vocabulary the step-0 distribution and "
                         "perplexity are exactly the parent's "
                         "(tests/test_silentgrowth.py)."),
            },
            "candidate": {
                "id": "silent_growth-g1",
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
                "perplexity_delta_pct": round(
                    100.0 * (child_metrics["val_perplexity"]
                             - parent_metrics["val_perplexity"])
                    / parent_metrics["val_perplexity"], 2),
                "verdict": verdict,
                "promotion_candidate": bool(advanced),
                "refusal_reasons": reasons,
                "parent_weights_sha256": parent.manifest["weights_sha256"],
                "child_weights_sha256": ckpt_manifest["weights_sha256"],
                "vocab_added_tokens": added_tokens,
                "vocab_added_count": len(added_tokens),
                "hidden_growth": int(HIDDEN_GROWTH),
                "param_count": ckpt_manifest["architecture"]["param_count"],
            },
            "note": ("Promotion is the owner's decision. This experiment "
                     "never writes model_data/current.json; a passing "
                     "child is recorded as a promotion CANDIDATE only."),
        }
        manifest_path = out_root / "manifest.json"
        tmp = str(manifest_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        os.replace(tmp, manifest_path)

        core.ledger.append("evolution", "MODEL_EVOLVED", {
            "generation": generation, "experiment": "silent_growth",
            "strategy": STRATEGY_A2, "verdict": verdict.lower(),
            "variation": variation,
            "parent_perplexity": parent_metrics["val_perplexity"],
            "child_perplexity": child_metrics["val_perplexity"],
            "parent_probe_score": parent_metrics["probe_score"],
            "child_probe_score": child_metrics["probe_score"],
            "child_weights_sha256": ckpt_manifest["weights_sha256"],
        })
        core.ledger.append("evolution", "SILENT_GROWTH_RUN", {
            "verdict": verdict,
            "promotion_candidate": bool(advanced),
            "selected_epoch": records[chosen]["epoch"],
            "child_perplexity": child_metrics["val_perplexity"],
        })
    finally:
        core.close()
    print(json.dumps({"verdict": manifest["candidate"]["verdict"],
                      "promotion_candidate": manifest["candidate"]["promotion_candidate"],
                      "child_metrics": manifest["candidate"]["child_metrics"],
                      "selected_epoch": manifest["candidate"]["selected_epoch"]},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
