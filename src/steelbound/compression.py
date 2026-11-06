"""Compression core — Synthesis S1 (docs/SYNTHESIS.md, decision D-016).

The synthesis principle is that prediction = compression =
intelligence: a model that predicts well is a model that compresses
well, because assigning high probability to what actually happens is
exactly what a good code does. This module makes that measurable and
gives Steelbound a learned encoding:

  * ``BPEEncoder`` — a byte-pair-encoding-style encoder learned from a
    token corpus. Merge learning is deterministic: at each step the
    most frequent adjacent pair is merged, ties broken by the pair
    itself (lexicographic on the token tuples), so training twice on
    the same corpus always yields the same merges. Encoding applies
    the merges in learned order; decoding expands merged symbols back
    to their parts, so encode -> decode is a LOSSLESS roundtrip of the
    original token sequence. ``to_dict``/``from_dict`` serialize the
    merge table (JSON-safe) so an encoder can be rebuilt identically.

  * ``bits_per_token`` — a model's mean negative log-likelihood (its
    loss, in nats) converted to bits: loss / ln 2. This is the average
    codelength per token of an ideal arithmetic code driven by the
    model, and it is the first-class evolution metric S1 adds.

  * ``compression_report`` — codelength accounting for token
    sequences under a model versus a naive fixed-width code.

Clean-room, stdlib only except numpy, which is imported lazily so the
rest of Steelbound stays zero-dependency (same pattern as model.py).
Tokens here are strings; a merged symbol is represented internally as
a tuple of its constituent tokens, which keeps decode trivially
lossless and serialization unambiguous.
"""
from __future__ import annotations

import json
import math
from collections import Counter

_MERGED_SEP = "\u0001"  # separator inside a serialized merged symbol;
# corpus tokens come from model.tokenize and never contain it.


class BPEEncoder:
    """Learned BPE-style encoder over string tokens.

    ``merges`` is the ordered list of pairs that were merged, in the
    order learned. A symbol is either a base token (str) or a merged
    symbol (tuple of constituent base tokens, flattened).
    """

    def __init__(self, merges: list[tuple[tuple, tuple]] | None = None):
        # Each merge: (left_symbol, right_symbol) -> concatenated symbol.
        self.merges: list[tuple[tuple, tuple]] = list(merges or [])
        self._merge_rank: dict[tuple[tuple, tuple], int] = {
            pair: i for i, pair in enumerate(self.merges)}

    # -- training ---------------------------------------------------------
    @classmethod
    def train(cls, token_seqs: list[list[str]], num_merges: int) -> "BPEEncoder":
        """Learn up to ``num_merges`` merges from a token corpus.

        Deterministic: pair counts are exact, and ties are broken by
        the pair key itself, never by iteration order.
        """
        seqs: list[list[tuple]] = [[(t,) for t in seq] for seq in token_seqs]
        merges: list[tuple[tuple, tuple]] = []
        for _ in range(max(0, int(num_merges))):
            counts: Counter = Counter()
            for seq in seqs:
                for a, b in zip(seq, seq[1:]):
                    counts[(a, b)] += 1
            if not counts:
                break
            best_count = max(counts.values())
            if best_count < 2:
                break
            # Fixed tie-break: smallest pair (tuple comparison) among
            # those with the maximal count.
            best_pair = min(pair for pair, c in counts.items() if c == best_count)
            merges.append(best_pair)
            seqs = [cls._apply_pair(seq, best_pair) for seq in seqs]
        return cls(merges)

    @staticmethod
    def _apply_pair(seq: list[tuple], pair: tuple[tuple, tuple]) -> list[tuple]:
        out: list[tuple] = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                out.append(seq[i] + seq[i + 1])
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    # -- encoding / decoding ----------------------------------------------
    def encode_symbols(self, tokens: list[str]) -> list[tuple]:
        """Encode tokens to symbols, applying merges in learned order
        (lowest rank first), as in classic BPE."""
        seq: list[tuple] = [(t,) for t in tokens]
        while len(seq) >= 2:
            best_rank, best_pair = None, None
            for a, b in zip(seq, seq[1:]):
                rank = self._merge_rank.get((a, b))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, (a, b)
            if best_pair is None:
                break
            seq = self._apply_pair(seq, best_pair)
        return seq

    def encode(self, tokens: list[str]) -> list[str]:
        """Encode tokens to a list of symbol strings. Merged symbols
        are joined with a control separator that base tokens never
        contain, so the encoding is reversible."""
        return [_MERGED_SEP.join(sym) for sym in self.encode_symbols(tokens)]

    def decode(self, encoded: list[str]) -> list[str]:
        """Expand encoded symbols back to the original tokens. Exact
        inverse of ``encode``: decode(encode(tokens)) == tokens."""
        out: list[str] = []
        for item in encoded:
            out.extend(item.split(_MERGED_SEP))
        return out

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        """JSON-safe serialization: merges in learned order, each
        symbol as a list of its constituent base tokens."""
        return {"format": "steelbound-bpe", "version": 1,
                "merges": [[list(a), list(b)] for a, b in self.merges]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, doc: dict) -> "BPEEncoder":
        if doc.get("format") != "steelbound-bpe":
            raise ValueError("not a steelbound-bpe encoder document")
        merges = [(tuple(a), tuple(b)) for a, b in doc["merges"]]
        return cls(merges)

    @classmethod
    def from_json(cls, text: str) -> "BPEEncoder":
        return cls.from_dict(json.loads(text))

    def __eq__(self, other) -> bool:
        return isinstance(other, BPEEncoder) and self.merges == other.merges

    def __len__(self) -> int:
        return len(self.merges)


def bits_per_token(model, X, y) -> float:
    """The model's mean negative log-likelihood in bits per token:
    exactly ``model.loss(X, y) / ln 2``. Lower is better compression
    (and, by the synthesis principle, better prediction)."""
    return float(model.loss(X, y)) / math.log(2.0)


def compression_report(model, token_seqs: list[list[int]]) -> dict:
    """Codelength accounting for integer token sequences under a model.

    Returns the model's bits per token, the total codelength (bits)
    of the sequences under the model (teacher-forced next-token code,
    contexts built the same way training examples are), and a naive
    baseline: a fixed-width code using ceil(log2(vocab_size)) bits per
    token. ``ratio`` is model codelength / naive codelength — below
    1.0 means the model compresses better than the naive code.
    """
    from .model import make_examples  # lazy: keeps numpy out of import time

    X, y = make_examples(token_seqs, model.context_len)
    if len(X) == 0:
        raise ValueError("no examples to evaluate")
    bpt = bits_per_token(model, X, y)
    total_bits = bpt * int(len(y))
    vocab_size = len(model.vocab)
    naive_bpt = max(1, math.ceil(math.log2(max(2, vocab_size))))
    naive_bits = float(naive_bpt * int(len(y)))
    return {
        "bits_per_token": bpt,
        "total_codelength_bits": total_bits,
        "tokens": int(len(y)),
        "vocab_size": int(vocab_size),
        "naive_bits_per_token": naive_bpt,
        "naive_codelength_bits": naive_bits,
        "ratio": total_bits / naive_bits if naive_bits else 1.0,
    }
