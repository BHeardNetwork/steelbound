"""Gen 2's one variation — full-coverage teacher exposure.

Diagnosis (GEN2_DESIGN.md, WORK_LEDGER 0054): Gen 1's teacher
recall stalled at 12/78 not from capacity, selection or format,
but from EXPOSURE STARVATION. Its ~5% teacher share was achieved
with a strided subset of 93 of the 471 rendered teacher sequences,
so only 20 of the 78 probe-aligned Q&A pairings were ever trained
and 171 of the 327 vocab-added teacher tokens received zero
gradient in the entire run.

This module is the replacement mix mechanism: the FULL rendered
teacher set is always included — every lesson's facts, examples,
counterexample, Q&A line and exercise line — and the set is
repeated (whole-set repeats plus a deterministic strided top-up)
until teacher EXAMPLES reach the target share of the training mix.
Coverage is a floor, not a trade-off: for any positive target
share every teacher sequence appears at least once.

Everything else about Gen 2 (silent zero-init growth, differential
learning rates, A2 sub-epoch snapshot selection, own-corpus replay)
is unchanged from Gen 1 and lives in ``silentgrowth.py`` /
``multiverse.py``; the experiment runner is
``training/run_gen2_exposure.py``. Nothing here trains, promotes,
or writes the live pointer.
"""
from __future__ import annotations

__all__ = ["TEACHER_TARGET_SHARE", "build_teacher_exposure",
           "example_count"]

#: Gen 2's target teacher share of training examples (design §2.2).
TEACHER_TARGET_SHARE = 0.25


def example_count(seqs) -> int:
    """Training examples a sequence list yields: ``make_examples``
    produces ``len(seq) + 1`` per sequence (the convention the
    Gen 1 runner recorded its shares with)."""
    return sum(len(s) + 1 for s in seqs)


def _strided(seqs, n: int):
    """Deterministic evenly-spaced pick of ``n`` sequences — the
    same rule the Gen 1 runner's subset used, kept so the top-up
    here is reproducible and comparable."""
    if n >= len(seqs):
        return list(seqs)
    if n <= 0:
        return []
    step = len(seqs) / n
    return [seqs[int(i * step)] for i in range(n)]


def build_teacher_exposure(teacher_seqs, base_examples: int,
                           target_share: float = TEACHER_TARGET_SHARE):
    """Build the teacher block of a training mix at ~``target_share``.

    ``teacher_seqs``: the full rendered teacher set.
    ``base_examples``: example count of the rest of the mix (own
    train split + curriculum replay) the share is measured against
    — share = teacher_examples / (base_examples + teacher_examples),
    the definition every run since Gen 1 has recorded.

    Returns ``(block, info)``: ``block`` is the full set repeated
    ``info["full_repeats"]`` times followed by a strided top-up of
    ``info["topup_count"]`` sequences — the (repeats, top-up) pair
    whose achieved example share is closest to ``target_share``.
    ``info`` records the exact achieved shares, by examples and by
    sequences, so the run manifest states what actually happened
    rather than what was targeted.
    """
    if not 0.0 < target_share < 1.0:
        raise ValueError("target_share must be in (0, 1)")
    seqs = list(teacher_seqs)
    info = {
        "target_share_examples": target_share,
        "full_repeats": 0,
        "topup_count": 0,
        "teacher_examples": 0,
        "teacher_sequences_with_repeats": 0,
        "teacher_unique_sequences": len(seqs),
        "teacher_share_examples": 0.0,
        "teacher_share_sequences": 0.0,
        "coverage_fraction": 0.0,
    }
    if not seqs or base_examples < 0:
        return [], info

    full_ex = example_count(seqs)
    desired = base_examples * target_share / (1.0 - target_share)
    # Precompute the strided top-up example counts once; the search
    # below is then pure arithmetic.
    topup_ex = [example_count(_strided(seqs, n))
                for n in range(len(seqs) + 1)]
    max_repeats = max(1, int(desired // full_ex) + 1)
    best = None  # (|share - target|, repeats, topup_n, teacher_ex)
    for repeats in range(1, max_repeats + 1):
        for n in range(len(seqs) + 1):
            t_ex = repeats * full_ex + topup_ex[n]
            share = t_ex / (base_examples + t_ex)
            diff = abs(share - target_share)
            if best is None or diff < best[0]:
                best = (diff, repeats, n, t_ex)
    _, repeats, topup_n, teacher_ex = best
    block = seqs * repeats + _strided(seqs, topup_n)
    info.update({
        "full_repeats": repeats,
        "topup_count": topup_n,
        "teacher_examples": teacher_ex,
        "teacher_sequences_with_repeats": len(block),
        "teacher_share_examples": teacher_ex / (base_examples + teacher_ex),
        "coverage_fraction": 1.0,
    })
    return block, info
