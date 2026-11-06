"""Multiverse evolution — branching bloodlines raced under one gate.

The single-line bloodline in ``evolution.py`` grows one child per
generation. The multiverse grows a *tree*: each generation, every
living path (branch) produces one candidate child from its own tip —
parent weights plus one recorded variation — and the candidate
advances its path only if it passes the same no-regression rule the
bloodline has always used, judged head-to-head against its own
parent on the frozen held-out split (``training/corpus.py``
``stable_split`` — the caller supplies sequences already split by
it, so every candidate in every generation stands on the same
ground).

Why branches at all: the two Gen 1 children of 2026-11-02 learned
their teacher lessons but regressed on the own corpus and were
refused (WORK_LEDGER 0026/0027). The verified root cause was the
epoch-selection score inherited from Gen 0 training —

    score = val_loss + 0.3 * missed_probes (+ 0.3 * missed_targets)

— where a *count* of missed probes can swamp any plausible val-loss
difference (78 teacher probes x 0.3 = up to 23.4 nats against a
~1 nat corpus gap), so selection picked late, lesson-heavy, overfit
epochs and threw away the parent's corpus fit. Two fixes were
designed but never chosen between; this module races them as branch
strategies so the choice is empirical:

  A1  two-stage gate — stage 1 admits only epochs with perfect core
      probes and a target (lesson) score no worse than the parent's;
      stage 2 picks the lowest val_loss among the admitted. If no
      epoch is admissible, a recorded fallback protects the core
      first (fewest core misses, then best target, then val_loss).
  A2  fraction term — probe misses enter as *fractions* with fixed
      weights, so the total probe penalty is bounded (<= 1.0 nat at
      the default weights) no matter how many probes a generation
      declares: score = val_loss + 0.5*(1 - core_score)
      + 0.5*(1 - target_score).

Branch variations raced alongside the strategies: differential
learning rates (inherited weights train at a fraction of the rate;
newly grown vocabulary rows / hidden units train at the full rate —
the fix entry 0026 recorded as next), capacity growth, and
lesson-pack emphasis (teacher-sequence repeat factor).

Rules that do not bend here:
  * A candidate is produced BY ITS PARENT — weights copied from the
    parent checkpoint, never from a third-party model, never from
    scratch (Gen 0's random init is the only root).
  * Training data is whatever the caller supplies; the experiment
    runner supplies Steelbound's own corpus plus the existing
    provenance-tracked teacher packs only (clean-room).
  * Failing candidates are preserved as checkpoints and recorded as
    refused; a path whose candidate fails is extinct — recorded,
    never forced onward.
  * At most ``MAX_LIVING_PATHS`` paths stay alive per generation.
  * NOTHING here promotes. The live pointer (model_data/current.json)
    is never written by this module; the best final candidate is
    reported as a promotion *candidate* for the owner's decision.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .evolution import DEFAULT_PROBES, compare, evaluate_model
from .model import SteelboundModel, Vocab, make_examples

STRATEGY_A1 = "a1_two_stage_gate"
STRATEGY_A2 = "a2_fraction_term"
STRATEGY_LEGACY = "legacy_count_term"

# A2's probe-penalty budget: each axis contributes at most its weight
# in nats, so recall can never outweigh corpus fit by more than the
# sum of the weights, however many probes a generation declares.
A2_CORE_WEIGHT = 0.5
A2_TARGET_WEIGHT = 0.5

# The bloodline's standing no-regression band (evolution.compare
# uses the same 2%): a child may be up to this much worse on
# own-corpus perplexity and still count as no-regression; probes may
# not drop at all.
DEFAULT_TOLERANCE = 0.02
MAX_LIVING_PATHS = 3


# ---------------------------------------------------------------------------
# Epoch-selection strategies (pure functions over per-epoch records)
# ---------------------------------------------------------------------------

def score_epoch_legacy(record: dict, core_total: int,
                       target_total: int | None = None) -> float:
    """The pre-multiverse selection score (Gen 0 / Gen 1 protocol):
    val_loss + 0.3 per missed probe, counts unbounded. Kept so the
    diagnosis in WORK_LEDGER 0026/0027 is reproducible and testable."""
    score = record["val_loss"] + 0.3 * round((1.0 - record["core_score"]) * core_total)
    if target_total:
        missed_t = round((1.0 - (record.get("target_score") or 0.0)) * target_total)
        score += 0.3 * missed_t
    return score


def score_epoch_a2(record: dict, core_weight: float = A2_CORE_WEIGHT,
                   target_weight: float = A2_TARGET_WEIGHT) -> float:
    """A2 — fraction term: misses as fractions, penalty bounded by
    core_weight + target_weight no matter the probe counts."""
    score = record["val_loss"] + core_weight * (1.0 - record["core_score"])
    if record.get("target_score") is not None:
        score += target_weight * (1.0 - record["target_score"])
    return score


def select_epoch_a1(records: list[dict],
                    parent_target_score: float | None = None) -> tuple[int, dict]:
    """A1 — two-stage gate. Stage 1 admits epochs whose core probes
    are perfect and whose target score is no worse than the parent's;
    stage 2 takes the lowest val_loss among the admitted. Fallback
    (no admissible epoch): protect the core first, then the target,
    then val_loss — and say so in the returned info."""
    def admissible(rec: dict) -> bool:
        if rec["core_score"] < 1.0:
            return False
        if (parent_target_score is not None
                and rec.get("target_score") is not None
                and rec["target_score"] < parent_target_score):
            return False
        return True

    admitted = [i for i, rec in enumerate(records) if admissible(rec)]
    if admitted:
        best = min(admitted, key=lambda i: records[i]["val_loss"])
        return best, {"strategy": STRATEGY_A1, "fallback": False,
                      "admissible_epochs": [records[i]["epoch"] for i in admitted]}
    best = min(range(len(records)),
               key=lambda i: (-records[i]["core_score"],
                              -(records[i].get("target_score") or 0.0),
                              records[i]["val_loss"]))
    return best, {"strategy": STRATEGY_A1, "fallback": True, "admissible_epochs": []}


def select_epoch_a2(records: list[dict]) -> tuple[int, dict]:
    best = min(range(len(records)), key=lambda i: score_epoch_a2(records[i]))
    return best, {"strategy": STRATEGY_A2, "fallback": False,
                  "scores": [round(score_epoch_a2(r), 6) for r in records]}


def select_epoch(strategy: str, records: list[dict],
                 parent_target_score: float | None = None) -> tuple[int, dict]:
    if strategy == STRATEGY_A1:
        return select_epoch_a1(records, parent_target_score)
    if strategy == STRATEGY_A2:
        return select_epoch_a2(records)
    raise ValueError(f"unknown epoch-selection strategy: {strategy}")


# ---------------------------------------------------------------------------
# Advancement — the no-regression rule, applied per path
# ---------------------------------------------------------------------------

def advancement(parent_metrics: dict, child_metrics: dict,
                parent_target: float | None, child_target: float | None,
                tolerance: float = DEFAULT_TOLERANCE) -> tuple[bool, list[str]]:
    """A candidate advances its path iff it does not regress against
    its parent: own-corpus perplexity no worse than the parent's
    beyond ``tolerance``, core probes not worse, target (lesson)
    probes not worse. Returns (advanced, reasons_refused)."""
    reasons = []
    band = parent_metrics["val_perplexity"] * (1.0 + tolerance)
    if child_metrics["val_perplexity"] > band:
        reasons.append(
            f"perplexity regressed: {child_metrics['val_perplexity']:.2f} "
            f"> parent {parent_metrics['val_perplexity']:.2f} x {1.0 + tolerance:.2f}")
    if child_metrics["probe_score"] < parent_metrics["probe_score"]:
        reasons.append(
            f"core probes regressed: {child_metrics['probe_score']:.3f} "
            f"< parent {parent_metrics['probe_score']:.3f}")
    if (parent_target is not None and child_target is not None
            and child_target < parent_target):
        reasons.append(
            f"target probes regressed: {child_target:.3f} < parent {parent_target:.3f}")
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Differential learning rates
# ---------------------------------------------------------------------------

def build_lr_scales(child: SteelboundModel, parent_vocab_size: int,
                    parent_hidden: int, inherited_scale: float) -> dict:
    """Per-element Adam step scales for differential learning rates:
    entries that existed in the parent (the inherited block) are
    scaled by ``inherited_scale``; entries created by vocabulary or
    hidden growth are exactly 1.0 (full rate). Shapes follow the
    child's current parameters, so call AFTER expansion."""
    s = float(inherited_scale)
    pv, ph = int(parent_vocab_size), int(parent_hidden)
    scales = {}
    E = np.full(child.E.shape, 1.0); E[:pv] = s; scales["E"] = E
    W1 = np.full(child.W1.shape, 1.0); W1[:, :ph] = s; scales["W1"] = W1
    b1 = np.full(child.b1.shape, 1.0); b1[:ph] = s; scales["b1"] = b1
    W2 = np.full(child.W2.shape, 1.0); W2[:ph, :pv] = s; scales["W2"] = W2
    b2 = np.full(child.b2.shape, 1.0); b2[:pv] = s; scales["b2"] = b2
    return scales


def _probe_fraction(model: SteelboundModel, probes) -> float | None:
    if not probes:
        return None
    hits = sum(1 for prompt, expected in probes
               if expected.lower() in model.generate(prompt, max_tokens=24).lower())
    return hits / len(probes)


def _slim(metrics: dict) -> dict:
    return {k: metrics[k] for k in ("val_loss", "bits_per_token", "val_perplexity",
                                    "probe_passed", "probe_total", "probe_score")}


# ---------------------------------------------------------------------------
# Candidate production — one parent, one child, one recorded variation
# ---------------------------------------------------------------------------

def produce_candidate(parent_dir: str, train_token_seqs: list[list[str]],
                      val_token_seqs: list[list[str]], out_dir: str, *,
                      strategy: str, variation: dict,
                      epochs: int = 4, lr: float = 0.002, seed: int = 11,
                      hidden_growth: int = 0,
                      inherited_lr_scale: float | None = None,
                      target_probes=None, tolerance: float = DEFAULT_TOLERANCE,
                      ledger=None) -> dict:
    """Train one candidate child from ``parent_dir``'s weights and
    judge it against that parent. Mirrors evolution.evolve_generation's
    protocol (child-encoded train, parent-encoded val, best-epoch
    selection, hash-recorded checkpoint) with two changes: the epoch
    that gets kept is chosen by ``strategy`` (A1 or A2), and Adam can
    run at differential rates when ``inherited_lr_scale`` is set.

    The checkpoint is ALWAYS saved (refused children are preserved,
    as Gen 1 was). Returns the candidate record for the manifest."""
    started = time.time()
    parent = SteelboundModel.load_checkpoint(parent_dir)
    parent_vocab_size = len(parent.vocab)
    parent_hidden = parent.hidden_dim

    new_tokens = sorted({t for seq in train_token_seqs for t in seq}
                        - set(parent.vocab.itos))
    child_vocab = Vocab(parent.vocab.itos + new_tokens)

    child = SteelboundModel(parent.vocab, context_len=parent.context_len,
                            embed_dim=parent.embed_dim,
                            hidden_dim=parent.hidden_dim, seed=seed)
    for name in ("E", "W1", "b1", "W2", "b2"):
        setattr(child, name, getattr(parent, name).copy())
    child._adam = None
    added_tokens: list[str] = []
    if new_tokens:
        added_tokens = child.expand_vocab(new_tokens, seed=seed)
    if hidden_growth:
        child.expand_hidden(int(hidden_growth), seed=seed)

    lr_scales = None
    if inherited_lr_scale is not None:
        lr_scales = build_lr_scales(child, parent_vocab_size, parent_hidden,
                                    inherited_lr_scale)

    X_train, y_train = make_examples([child_vocab.encode(s) for s in train_token_seqs],
                                     parent.context_len)
    X_val, y_val = make_examples([parent.vocab.encode(s) for s in val_token_seqs],
                                 parent.context_len)
    parent_metrics = evaluate_model(parent, X_val, y_val)
    parent_target = _probe_fraction(parent, target_probes)

    snapshots: list[dict] = []

    def snapshot(m, row):
        snapshots.append({
            "epoch": row["epoch"],
            "val_loss": row["val_loss"],
            "core_score": _probe_fraction(m, DEFAULT_PROBES),
            "target_score": _probe_fraction(m, target_probes),
            "params": {k: p.copy() for k, p in m._params().items()},
        })

    child.train(X_train, y_train, epochs=epochs, lr=lr, seed=seed,
                X_val=X_val, y_val=y_val, epoch_hook=snapshot,
                lr_scales=lr_scales)
    records = [{k: r[k] for k in ("epoch", "val_loss", "core_score", "target_score")}
               for r in snapshots]
    chosen, selection_info = select_epoch(strategy, records, parent_target)
    for name, arr in snapshots[chosen]["params"].items():
        setattr(child, name, arr)
    child._adam = None

    generation = int(parent.manifest.get("generation", 0)) + 1
    manifest = child.save_checkpoint(out_dir, generation=generation, training={
        "method": "multiverse_candidate_from_parent",
        "variation": variation,
        "strategy": strategy,
        "selection": {**selection_info, "selected_epoch": records[chosen]["epoch"]},
        "vocab_added_tokens": added_tokens,
        "parent_dir": os.path.basename(os.path.normpath(parent_dir)),
        "parent_weights_sha256": parent.manifest["weights_sha256"],
        "epochs": epochs, "lr": lr, "seed": seed,
        "hidden_growth": int(hidden_growth),
        "inherited_lr_scale": inherited_lr_scale,
        "train_examples": int(len(X_train)),
        "epoch_records": records,
    })
    child_metrics = evaluate_model(child, X_val, y_val)
    child_target = _probe_fraction(child, target_probes)
    advanced, reasons = advancement(parent_metrics, child_metrics,
                                    parent_target, child_target, tolerance)
    record = {
        "generation": generation,
        "strategy": strategy,
        "variation": variation,
        "checkpoint": out_dir,
        "selected_epoch": records[chosen]["epoch"],
        "selection": selection_info,
        "epoch_records": records,
        "parent_metrics": _slim(parent_metrics),
        "child_metrics": _slim(child_metrics),
        "parent_target_score": parent_target,
        "child_target_score": child_target,
        "verdict": "advanced" if advanced else "refused",
        "refusal_reasons": reasons,
        "parent_weights_sha256": parent.manifest["weights_sha256"],
        "child_weights_sha256": manifest["weights_sha256"],
        "vocab_added_tokens": added_tokens,
        "hidden_growth": int(hidden_growth),
        "inherited_lr_scale": inherited_lr_scale,
        "epochs": epochs, "lr": lr, "seed": seed,
        "param_count": manifest["architecture"]["param_count"],
        "seconds": round(time.time() - started, 1),
    }
    if ledger is not None:
        ledger.append("evolution", "MODEL_EVOLVED", {
            "generation": generation, "experiment": "multiverse",
            "strategy": strategy, "verdict": record["verdict"],
            "variation": variation,
            "parent_perplexity": parent_metrics["val_perplexity"],
            "child_perplexity": child_metrics["val_perplexity"],
            "parent_probe_score": parent_metrics["probe_score"],
            "child_probe_score": child_metrics["probe_score"],
            "child_weights_sha256": manifest["weights_sha256"],
        })
    return record


# ---------------------------------------------------------------------------
# The multiverse run — paths, generations, pruning, extinction, manifest
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    """One branch's recipe: a selection strategy plus the training
    variation it applies every generation (growth is larger in
    generation 1, where the vocabulary also grows)."""
    name: str
    strategy: str
    lr: float = 0.002
    inherited_lr_scale: float | None = None
    teacher_repeat: int = 3
    curriculum_repeat: int = 3
    hidden_growth_first: int = 64
    hidden_growth_later: int = 16
    epochs: int = 4
    seed_base: int = 11


@dataclass
class PathState:
    config: PathConfig
    tip_dir: str
    tip_id: str
    alive: bool = True
    extinct_at: int | None = None
    extinction_reason: str | None = None
    history: list[str] = field(default_factory=list)


def prune_paths(states: list[PathState], tip_metrics: dict[str, dict],
                max_living: int = MAX_LIVING_PATHS) -> list[PathState]:
    """Keep at most ``max_living`` living paths, ranked by their
    tip's last child metrics (lowest own-corpus perplexity, then
    highest target score). Paths beyond the cap are marked extinct
    by pruning. Returns the states pruned this call."""
    living = [s for s in states if s.alive]
    if len(living) <= max_living:
        return []
    def rank(state: PathState):
        m = tip_metrics[state.config.name]
        return (m["child_metrics"]["val_perplexity"],
                -(m["child_target_score"] or 0.0))
    living.sort(key=rank)
    pruned = living[max_living:]
    for state in pruned:
        state.alive = False
        state.extinction_reason = "pruned: path cap"
    return pruned


def _write_manifest(path: str, manifest: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def run_multiverse(*, root_checkpoint: str, out_root: str,
                   own_train_seqs: list[list[str]],
                   curriculum_seqs: list[list[str]],
                   teacher_seqs: list[list[str]],
                   val_seqs: list[list[str]],
                   target_probes, path_configs: list[PathConfig],
                   max_generations: int = 10,
                   ground_info: dict | None = None,
                   tolerance: float = DEFAULT_TOLERANCE,
                   ledger=None, log=None) -> dict:
    """Run the branching experiment from ``root_checkpoint`` (Gen 0)
    through ``max_generations``. All artefacts live under
    ``out_root``: one checkpoint dir per candidate plus manifest.json
    (rewritten atomically after every candidate, so a crash still
    leaves a truthful tree). The live pointer is never touched."""
    log = log or (lambda *a: None)
    os.makedirs(out_root, exist_ok=True)
    root = SteelboundModel.load_checkpoint(root_checkpoint)
    X_val_root, y_val_root = make_examples(
        [root.vocab.encode(s) for s in val_seqs], root.context_len)
    root_metrics = evaluate_model(root, X_val_root, y_val_root)
    root_target = _probe_fraction(root, target_probes)

    manifest: dict = {
        "experiment": "multiverse",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "root": {
            "checkpoint": root_checkpoint,
            "generation": int(root.manifest.get("generation", 0)),
            "weights_sha256": root.manifest["weights_sha256"],
            "metrics_on_run_ground": _slim(root_metrics),
            "target_score_on_run_ground": root_target,
        },
        "ground": ground_info or {},
        "tolerance": tolerance,
        "max_living_paths": MAX_LIVING_PATHS,
        "path_configs": {c.name: {k: v for k, v in vars(c).items()}
                         for c in path_configs},
        "candidates": [],
        "paths": {},
        "final": None,
    }
    manifest_path = os.path.join(out_root, "manifest.json")

    states = [PathState(config=c, tip_dir=root_checkpoint, tip_id="gen0")
              for c in path_configs]
    last_records: dict[str, dict] = {}
    _write_manifest(manifest_path, manifest)

    for generation in range(1, max_generations + 1):
        living = [s for s in states if s.alive]
        if not living:
            log(f"generation {generation}: no living paths — experiment over")
            break
        for state in living:
            cfg = state.config
            growth = (cfg.hidden_growth_first if generation == 1
                      else cfg.hidden_growth_later)
            mix = (own_train_seqs + curriculum_seqs * cfg.curriculum_repeat
                   + teacher_seqs * cfg.teacher_repeat)
            variation = {
                "kind": ("vocab_and_capacity_growth_plus_teacher_curriculum"
                         if generation == 1 else
                         "continued_training_path_recipe"),
                "path": cfg.name,
                "strategy": cfg.strategy,
                "teacher_repeat": cfg.teacher_repeat,
                "curriculum_repeat": cfg.curriculum_repeat,
                "hidden_growth": growth,
                "inherited_lr_scale": cfg.inherited_lr_scale,
                "lr": cfg.lr,
                "teacher_authored": True,
            }
            cand_id = f"{cfg.name}-g{generation}"
            out_dir = os.path.join(out_root, cfg.name, f"gen{generation}")
            log(f"{cand_id}: training from {state.tip_id} "
                f"(strategy={cfg.strategy}, growth=+{growth}, "
                f"inherited_lr_scale={cfg.inherited_lr_scale})")
            record = produce_candidate(
                state.tip_dir, mix, val_seqs, out_dir,
                strategy=cfg.strategy, variation=variation,
                epochs=cfg.epochs, lr=cfg.lr,
                seed=cfg.seed_base + generation,
                hidden_growth=growth,
                inherited_lr_scale=cfg.inherited_lr_scale,
                target_probes=target_probes, tolerance=tolerance,
                ledger=ledger)
            record["id"] = cand_id
            record["path"] = cfg.name
            record["parent_id"] = state.tip_id
            record["checkpoint"] = os.path.relpath(out_dir, out_root)
            manifest["candidates"].append(record)
            last_records[cfg.name] = record
            state.history.append(cand_id)
            if record["verdict"] == "advanced":
                state.tip_dir = out_dir
                state.tip_id = cand_id
                log(f"{cand_id}: ADVANCED — ppl "
                    f"{record['parent_metrics']['val_perplexity']:.2f} -> "
                    f"{record['child_metrics']['val_perplexity']:.2f}, target "
                    f"{record['parent_target_score']} -> {record['child_target_score']}")
            else:
                state.alive = False
                state.extinct_at = generation
                state.extinction_reason = "; ".join(record["refusal_reasons"])
                log(f"{cand_id}: REFUSED — {state.extinction_reason}; "
                    f"path {cfg.name} extinct at generation {generation}")
            _write_manifest(manifest_path, manifest)
        pruned = prune_paths(states, last_records)
        for state in pruned:
            state.extinct_at = generation
            log(f"path {state.config.name}: pruned at generation {generation}")
        manifest["paths"] = {
            s.config.name: {
                "alive": s.alive, "tip": s.tip_id, "tip_dir": s.tip_dir,
                "extinct_at": s.extinct_at,
                "extinction_reason": s.extinction_reason,
                "history": s.history,
            } for s in states}
        _write_manifest(manifest_path, manifest)

    # Final: every surviving tip judged against the ROOT (Gen 0) on
    # root-encoded ground — the strict promotion-candidate test.
    finals = {}
    for state in states:
        if not state.alive or state.tip_id == "gen0":
            continue
        tip = SteelboundModel.load_checkpoint(state.tip_dir)
        tip_metrics = evaluate_model(tip, X_val_root, y_val_root)
        tip_target = _probe_fraction(tip, target_probes)
        verdict = compare(root_metrics, tip_metrics, root_target, tip_target)
        strict_ok = (tip_metrics["val_perplexity"] <= root_metrics["val_perplexity"]
                     and tip_metrics["probe_score"] >= root_metrics["probe_score"]
                     and (root_target is None or tip_target is None
                          or tip_target >= root_target))
        finals[state.config.name] = {
            "tip": state.tip_id,
            "tip_metrics": _slim(tip_metrics),
            "tip_target_score": tip_target,
            "compare_vs_root": verdict,
            "strict_no_regression_vs_root": bool(strict_ok and verdict == "promotion_worthy"),
            "promotion_candidate": bool(strict_ok and verdict == "promotion_worthy"),
        }
    best = None
    for name, fin in finals.items():
        if fin["promotion_candidate"] and (
                best is None
                or fin["tip_metrics"]["val_perplexity"]
                < finals[best]["tip_metrics"]["val_perplexity"]):
            best = name
    manifest["final"] = {
        "root_metrics": _slim(root_metrics),
        "root_target_score": root_target,
        "tips_vs_root": finals,
        "promotion_candidate_path": best,
        "note": ("Promotion is the owner's decision (owner path only). "
                 "This experiment never writes model_data/current.json."),
    }
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)
    if ledger is not None:
        ledger.append("evolution", "MULTIVERSE_RUN", {
            "candidates": len(manifest["candidates"]),
            "promotion_candidate_path": best,
            "paths": {name: p["tip"] for name, p in manifest["paths"].items()},
        })
    return manifest
