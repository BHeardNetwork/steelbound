"""Silent (zero-init) growth — the Gen 1 retry for growth shock.

Diagnosis being tested (MULTIVERSE_REPORT.md, WORK_LEDGER 0039):
multiverse Gen 1 children started epoch 1 already ~+0.35 nats over
their parent. A child there adds hundreds of vocabulary rows/columns
and dozens of hidden units, all random-init via
``SteelboundModel.expand_vocab`` / ``expand_hidden``; the new output
columns immediately steal softmax mass and the new hidden units
perturb every logit, before a single gradient step can quiet them.

Silent growth is the opposite entry condition. New capacity enters
at ZERO, so at step 0 the child's function is *exactly* the parent's
function on the parent's vocabulary:

  * new embedding rows are 0;
  * the output projection for new tokens (new ``W2`` columns and
    ``b2`` entries) is 0 — a new token's logit is exactly 0 before
    training;
  * new hidden units get small-random ``W1`` columns and ``b1`` = 0,
    but their ``W2`` rows are 0, so they contribute exactly 0 to
    every logit regardless of their activation.

Consequence, stated precisely: for any context of old tokens, the
child's *logits over old tokens* are bit-for-bit the parent's, so
the child's next-token distribution over old tokens, RENORMALIZED
over the old vocabulary, is identical to the parent's distribution,
and renormalized perplexity on old-vocab examples is unchanged.
(The raw, un-renormalized probabilities over old tokens are scaled
down by the new tokens' exp(0)=1 softmax mass — that is why the
invariant is stated renormalized. Training then earns the new
capacity in from zero under differential learning rates.)

This module never edits ``model.py``: it operates on the model's
arrays and ``Vocab`` directly, mirroring what ``expand_vocab`` /
``expand_hidden`` do structurally, only with zero initialisation
for the output side. It also never promotes: no live pointer is
written here or by the experiment runner that uses it.

Training here is sub-epoch: ``train_subepoch`` snapshots the model
every quarter epoch (by default) and evaluates each snapshot on a
frozen validation split plus probes, producing per-snapshot records
in the same dict shape ``multiverse``'s epoch selectors consume
(``epoch`` / ``val_loss`` / ``core_score`` / ``target_score``), so
``multiverse.select_epoch_a2`` can choose among 4x more selection
points than the per-epoch protocol offered.
"""
from __future__ import annotations

import math

import numpy as np

from .evolution import DEFAULT_PROBES
from .model import SteelboundModel, Vocab
from .multiverse import _probe_fraction, build_lr_scales

__all__ = ["grow_silent", "train_subepoch", "build_lr_scales"]


# ---------------------------------------------------------------------------
# Silent (zero-init) growth
# ---------------------------------------------------------------------------

def grow_silent(model: SteelboundModel, new_tokens, extra_hidden: int,
                seed: int = 0) -> tuple[list[str], int]:
    """Grow ``model`` in place with zero-init output capacity.

    ``new_tokens``: tokens to append to the vocabulary (existing
    tokens and duplicates are skipped; existing ids never move).
    ``extra_hidden``: hidden units to add. Returns
    ``(added_tokens, added_units)``.

    Invariant (proved in tests/test_silentgrowth.py): after growth,
    for contexts of old tokens, logits over old tokens are unchanged,
    so the renormalized distribution over old tokens and
    renormalized perplexity on old-vocab examples are unchanged.
    """
    added: list[str] = []
    seen = set(model.vocab.itos)
    for tok in new_tokens or []:
        if tok not in seen:
            added.append(tok)
            seen.add(tok)
    extra = int(extra_hidden) if extra_hidden else 0
    if extra < 0:
        raise ValueError("extra_hidden must be >= 0")
    if not added and extra == 0:
        return ([], 0)

    rng = np.random.default_rng(seed)
    old_vocab = len(model.vocab)
    old_hidden = model.hidden_dim
    d_in = model.W1.shape[0]

    # New hidden units: W1 columns small-random (so units differ once
    # their W2 rows grow away from zero), b1 = 0.
    if extra:
        w1_new = rng.normal(0, math.sqrt(1.0 / d_in), (d_in, extra))
        model.W1 = np.hstack([model.W1, w1_new])
        model.b1 = np.concatenate([model.b1, np.zeros(extra)])
        model.hidden_dim = old_hidden + extra

    # Embeddings for new tokens: exactly zero rows.
    if added:
        model.E = np.vstack(
            [model.E, np.zeros((len(added), model.embed_dim))])

    # Output projection: build the enlarged W2 with every new entry
    # zero — new columns (new tokens, old hidden rows), new rows
    # (new hidden units, all tokens) and the new/new block.
    new_hidden = model.hidden_dim
    new_vocab_size = old_vocab + len(added)
    if added or extra:
        new_w2 = np.zeros((new_hidden, new_vocab_size),
                          dtype=model.W2.dtype)
        new_w2[:old_hidden, :old_vocab] = model.W2
        model.W2 = new_w2
    if added:
        model.b2 = np.concatenate([model.b2, np.zeros(len(added))])
        model.vocab = Vocab(model.vocab.itos + added)

    # Optimizer state references the old shapes; reset it exactly as
    # expand_vocab / expand_hidden do.
    model._adam = None
    return (added, extra)


# ---------------------------------------------------------------------------
# Sub-epoch training with snapshot selection points
# ---------------------------------------------------------------------------

def train_subepoch(model: SteelboundModel,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, *,
                   epochs: int = 6, lr: float = 0.0005, seed: int = 11,
                   batch: int = 128,
                   lr_scales: dict | None = None,
                   target_probes=None,
                   snapshots_per_epoch: int = 4,
                   log=None) -> tuple[list[dict], list[dict]]:
    """Train ``model`` with ``loss_and_grads`` + ``_adam_step``,
    snapshotting every 1/``snapshots_per_epoch`` of an epoch.

    Each snapshot is evaluated on the frozen validation split
    (``model.loss``) and on the core / target probes, and recorded
    as ``{"epoch": <float, e.g. 1.25>, "val_loss": ..., "core_score":
    ..., "target_score": ...}`` — the same shape ``multiverse``'s
    selectors (``select_epoch_a2`` / ``score_epoch_a2``) consume,
    with fractional epochs naming the sub-epoch point.

    Returns ``(records, snapshots)``: ``records`` are the slim
    selector-ready dicts; ``snapshots`` are the same dicts plus a
    ``"params"`` copy of the weights at that point, so the caller
    can restore the selected snapshot (mirroring
    ``multiverse.produce_candidate``).

    ``lr_scales`` is passed straight through to ``_adam_step`` —
    build it with ``multiverse.build_lr_scales`` for differential
    rates (inherited weights slow, grown capacity full rate).
    """
    if snapshots_per_epoch < 1:
        raise ValueError("snapshots_per_epoch must be >= 1")
    if len(X_train) == 0:
        raise ValueError("no training examples")
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    snapshots: list[dict] = []
    step = 0
    n = len(X_train)
    seg_size = (n + snapshots_per_epoch - 1) // snapshots_per_epoch
    for epoch in range(1, int(epochs) + 1):
        order = rng.permutation(n)
        xs_all, ys_all = X_train[order], y_train[order]
        for seg in range(snapshots_per_epoch):
            start = seg * seg_size
            if start >= n:
                break
            seg_x = xs_all[start:start + seg_size]
            seg_y = ys_all[start:start + seg_size]
            for i in range(0, len(seg_x), batch):
                step += 1
                _, grads = model.loss_and_grads(seg_x[i:i + batch],
                                                seg_y[i:i + batch])
                model._adam_step(grads, lr, step, lr_scales=lr_scales)
            val_loss = float(model.loss(X_val, y_val)) if len(X_val) else float("nan")
            record = {
                "epoch": (epoch - 1) + (seg + 1) / snapshots_per_epoch,
                "val_loss": val_loss,
                "core_score": _probe_fraction(model, DEFAULT_PROBES),
                "target_score": _probe_fraction(model, target_probes),
            }
            records.append(record)
            snapshots.append({**record,
                              "params": {k: p.copy()
                                         for k, p in model._params().items()}})
            if log:
                log(record)
    return records, snapshots
