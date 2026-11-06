"""Predictive inference — Synthesis S2 (docs/SYNTHESIS.md, decision D-016).

Gen 2 of the synthesis is "predict and notice": the core predicts
outcomes *before* they happen, measures the prediction error as
**surprise** (in bits, the same unit as S1's bits-per-token), and
lets surprise drive what is learned next and what becomes a known
unknown. Introspection then reads the model's own uncertainty, not
just its records.

Two prediction levels live here, both clean-room and deterministic:

1. **Event prediction** (``FrequencyModel`` / ``PredictionEngine``).
   An outcome stream is a named sequence of categorical outcomes —
   e.g. ``ledger_action`` (the next ledger action the core records),
   or any caller-defined stream such as a tool result or a scheduler
   job outcome. The predictor is the stream's own history with
   Laplace smoothing over the observed categories plus one virtual
   "<unseen>" category, so a genuinely novel outcome gets a finite,
   large surprise instead of an infinite one or a silent zero. The
   predicted outcome is the most frequent past outcome, ties broken
   lexicographically, so prediction is fully deterministic.

   The engine stores every prediction and every observed outcome,
   and ledgers ``PREDICTION_MADE`` / ``PREDICTION_OBSERVED`` /
   ``PREDICTION_LEARNED``. An observation's surprise is computed
   *before* the observation updates the model — the error is against
   the prediction that was actually held, not against hindsight.
   Outcomes above the surprise threshold are "surprising"; the
   learning queue is the unlearned outcomes in surprise order, and
   surprising outcomes can be promoted into cognition's known
   unknowns, which is exactly how surprise feeds introspection.

2. **Token prediction** (``predict_next_token`` /
   ``token_surprise_bits``). The same surprise unit, applied to the
   core's own neural model: the model's next-token distribution over
   a context, and the negative log2 probability of the token that
   actually followed. This is S1's bits-per-token at the single-token
   level, and it needs no new model — it reads the distribution the
   bloodline model already produces.

Honesty guards, all enforced in code:

* A stream with no history has no prediction and no surprise: the
  first observation is recorded with ``surprise_bits=None`` and
  ``surprising=False``, never with a fabricated 0-bit surprise.
  (With Laplace smoothing alone the first outcome would score 0
  bits, which would claim a prediction that never existed.)
* The engine's own ledger entries (``PREDICTION_MADE`` and friends)
  are excluded from the ``ledger_action`` seed history, so predicting
  does not change the distribution being predicted — no self-feedback
  loop through the ledger.
* A prediction can be resolved exactly once; resolving twice, or
  resolving a prediction against a different stream, is refused.
* Nothing here changes the gate, the ledger chain, or any eval
  suite, and no network is used. Promotion of a surprising outcome
  to a known unknown is explicit (``promote_to_unknowns``), never
  automatic on observation, so a surprising event cannot silently
  rewrite what the core claims to know.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass

# Ledger actions this engine writes itself. They are excluded from the
# ledger_action seed history (see the module docstring).
_OWN_LEDGER_ACTIONS = frozenset({
    "PREDICTION_MADE", "PREDICTION_OBSERVED", "PREDICTION_LEARNED",
})

_UNSEEN = "<unseen>"


def _require_text(value, what: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{what} requires a non-empty value")
    return str(value)


@dataclass(frozen=True)
class Prediction:
    id: int
    stream: str
    context: str
    predicted: str | None
    probability: float | None
    distribution: dict
    entropy_bits: float
    ts: float


@dataclass(frozen=True)
class ObservedOutcome:
    id: int
    stream: str
    actual: str
    predicted: str | None
    prediction_id: int | None
    surprise_bits: float | None
    surprising: bool
    context: str
    ts: float
    learned: bool = False
    unknown_id: int | None = None


@dataclass(frozen=True)
class TokenPrediction:
    token: str
    token_id: int
    probability: float
    entropy_bits: float


class FrequencyModel:
    """A smoothed categorical distribution over one outcome stream.

    ``counts`` maps outcome -> times observed. The distribution is
    Laplace-smoothed with ``alpha`` over the observed categories plus
    one virtual unseen category, so every outcome — seen or novel —
    has a strictly positive probability and a finite surprise.
    """

    def __init__(self, counts: dict[str, int] | None = None, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.counts = {str(k): int(v) for k, v in (counts or {}).items()
                       if int(v) > 0}
        self.total = sum(self.counts.values())

    def distribution(self) -> dict:
        """Outcome -> probability, including the unseen category."""
        if self.total == 0:
            return {}
        denom = self.total + self.alpha * (len(self.counts) + 1)
        dist = {outcome: (count + self.alpha) / denom
                for outcome, count in self.counts.items()}
        dist[_UNSEEN] = self.alpha / denom
        return dist

    def _probability(self, outcome: str) -> float | None:
        if self.total == 0:
            return None
        dist = self.distribution()
        return dist.get(outcome, dist[_UNSEEN])

    def predict(self) -> tuple[str | None, float | None, dict, float]:
        """(predicted outcome, its probability, distribution, entropy).

        With no history there is no prediction: predicted and
        probability are None and the distribution is empty."""
        if self.total == 0:
            return None, None, {}, 0.0
        dist = self.distribution()
        observed = {o: p for o, p in dist.items() if o != _UNSEEN}
        # Deterministic tie-break: highest probability, then smallest
        # outcome string, so the same history always predicts the same.
        predicted = min(observed, key=lambda o: (-observed[o], o))
        return predicted, observed[predicted], dist, self.entropy_bits()

    def entropy_bits(self) -> float:
        dist = self.distribution()
        if not dist:
            return 0.0
        return float(-sum(p * math.log2(p) for p in dist.values() if p > 0))

    def surprise_bits(self, actual: str) -> float | None:
        """Prediction error for ``actual``, in bits: -log2 P(actual).

        None when the model has no history — there was no prediction
        to be surprised against. A novel outcome after some history
        scores the unseen-category probability, which is large but
        finite and shrinks as the history grows."""
        prob = self._probability(str(actual))
        if prob is None or prob <= 0:
            return None
        return float(-math.log2(prob))

    def observe(self, actual: str) -> None:
        actual = str(actual)
        self.counts[actual] = self.counts.get(actual, 0) + 1
        self.total += 1


class PredictionEngine:
    """Predict-then-observe for named outcome streams, ledgered.

    Streams are caller-defined except ``ledger_action``, which is
    seeded from the core's own ledger history (excluding this
    engine's own entries). ``cognition`` is optional; when supplied,
    ``promote_to_unknowns`` turns surprising outcomes into
    cognition known unknowns.
    """

    def __init__(self, ledger, conn: sqlite3.Connection, cognition=None,
                 surprise_threshold_bits: float = 3.0, alpha: float = 1.0):
        if surprise_threshold_bits <= 0:
            raise ValueError("surprise_threshold_bits must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.ledger = ledger
        self.conn = conn
        self.cognition = cognition
        self.surprise_threshold_bits = float(surprise_threshold_bits)
        self.alpha = float(alpha)
        conn.execute("""CREATE TABLE IF NOT EXISTS prediction_predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT, stream TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT '', predicted TEXT,
            probability REAL, distribution TEXT NOT NULL DEFAULT '{}',
            entropy REAL NOT NULL DEFAULT 0.0, ts REAL NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS prediction_outcomes(
            id INTEGER PRIMARY KEY AUTOINCREMENT, stream TEXT NOT NULL,
            actual TEXT NOT NULL, predicted TEXT, prediction_id INTEGER,
            surprise_bits REAL, surprising INTEGER NOT NULL DEFAULT 0,
            context TEXT NOT NULL DEFAULT '', ts REAL NOT NULL,
            learned INTEGER NOT NULL DEFAULT 0, unknown_id INTEGER)""")
        conn.commit()

    # -- model construction ------------------------------------------------
    def _model_for(self, stream: str) -> FrequencyModel:
        counts: dict[str, int] = {}
        for (actual,) in self.conn.execute(
                "SELECT actual FROM prediction_outcomes WHERE stream=?",
                (stream,)).fetchall():
            counts[actual] = counts.get(actual, 0) + 1
        if stream == "ledger_action" and self.ledger is not None:
            for entry in self.ledger.entries(10000):
                if entry.action in _OWN_LEDGER_ACTIONS:
                    continue
                counts[entry.action] = counts.get(entry.action, 0) + 1
        return FrequencyModel(counts, alpha=self.alpha)

    # -- prediction ----------------------------------------------------------
    def predict(self, stream: str, context: str = "") -> Prediction:
        stream = _require_text(stream, "predict stream")
        model = self._model_for(stream)
        predicted, probability, dist, entropy = model.predict()
        ts = time.time()
        cur = self.conn.execute(
            "INSERT INTO prediction_predictions(stream,context,predicted,"
            "probability,distribution,entropy,ts) VALUES(?,?,?,?,?,?,?)",
            (stream, str(context), predicted, probability,
             json.dumps(dist, sort_keys=True), entropy, ts))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("prediction", "PREDICTION_MADE", {
                "prediction_id": cur.lastrowid, "stream": stream,
                "context": str(context), "predicted": predicted,
                "probability": probability})
        return Prediction(cur.lastrowid, stream, str(context), predicted,
                          probability, dist, entropy, ts)

    def surprise_for(self, stream: str, actual: str) -> float | None:
        """Surprise of ``actual`` under the current model, without
        recording anything. None when the stream has no history."""
        stream = _require_text(stream, "surprise stream")
        actual = _require_text(actual, "surprise outcome")
        return self._model_for(stream).surprise_bits(actual)

    # -- observation ---------------------------------------------------------
    def observe(self, stream: str, actual: str, prediction_id: int | None = None,
                context: str = "") -> ObservedOutcome:
        stream = _require_text(stream, "observe stream")
        actual = _require_text(actual, "observe outcome")
        stored_predicted = None
        if prediction_id is not None:
            row = self.conn.execute(
                "SELECT stream,predicted,resolved FROM prediction_predictions"
                " WHERE id=?", (prediction_id,)).fetchone()
            if row is None:
                raise KeyError(f"no prediction {prediction_id}")
            if row[0] != stream:
                raise ValueError(
                    f"prediction {prediction_id} is for stream {row[0]!r}, "
                    f"not {stream!r}")
            if row[2]:
                raise ValueError(f"prediction {prediction_id} is already resolved")
            stored_predicted = row[1]
        # Surprise is measured against the model BEFORE this outcome
        # updates it: the error of the prediction actually held.
        model = self._model_for(stream)
        surprise = model.surprise_bits(actual)
        if stored_predicted is None:
            predicted, _, _, _ = model.predict()
        else:
            predicted = stored_predicted
        surprising = bool(surprise is not None
                          and surprise > self.surprise_threshold_bits)
        ts = time.time()
        cur = self.conn.execute(
            "INSERT INTO prediction_outcomes(stream,actual,predicted,"
            "prediction_id,surprise_bits,surprising,context,ts)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (stream, actual, predicted, prediction_id, surprise,
             int(surprising), str(context), ts))
        if prediction_id is not None:
            self.conn.execute(
                "UPDATE prediction_predictions SET resolved=1 WHERE id=?",
                (prediction_id,))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("prediction", "PREDICTION_OBSERVED", {
                "outcome_id": cur.lastrowid, "stream": stream,
                "actual": actual, "predicted": predicted,
                "prediction_id": prediction_id, "surprise_bits": surprise,
                "surprising": surprising})
        return ObservedOutcome(cur.lastrowid, stream, actual, predicted,
                               prediction_id, surprise, surprising,
                               str(context), ts)

    # -- surprise-driven learning -------------------------------------------
    def learning_queue(self, limit: int = 10) -> list[ObservedOutcome]:
        """Unlearned outcomes, most surprising first. These are what
        the core should learn next — surprise, not recency, orders
        the queue. Outcomes with no surprise measure (no prior
        history) sort last."""
        if int(limit) <= 0:
            raise ValueError("learning_queue limit must be positive")
        rows = self.conn.execute(
            "SELECT id,stream,actual,predicted,prediction_id,surprise_bits,"
            "surprising,context,ts,learned,unknown_id FROM prediction_outcomes"
            " WHERE learned=0 ORDER BY (surprise_bits IS NULL),"
            " surprise_bits DESC, id ASC LIMIT ?",
            (int(limit),)).fetchall()
        return [ObservedOutcome(r[0], r[1], r[2], r[3], r[4], r[5],
                                bool(r[6]), r[7], r[8], bool(r[9]), r[10])
                for r in rows]

    def mark_learned(self, outcome_id: int) -> ObservedOutcome:
        """Mark an outcome learned. Refused twice: learning is a
        one-time acknowledgement, not a counter that can be inflated."""
        row = self.conn.execute(
            "SELECT stream,actual,predicted,prediction_id,surprise_bits,"
            "surprising,context,ts,learned,unknown_id FROM prediction_outcomes"
            " WHERE id=?", (outcome_id,)).fetchone()
        if row is None:
            raise KeyError(f"no prediction outcome {outcome_id}")
        if row[8]:
            raise ValueError(f"outcome {outcome_id} is already learned")
        self.conn.execute(
            "UPDATE prediction_outcomes SET learned=1 WHERE id=?",
            (outcome_id,))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("prediction", "PREDICTION_LEARNED", {
                "outcome_id": outcome_id, "stream": row[0],
                "actual": row[1], "surprise_bits": row[4]})
        return ObservedOutcome(outcome_id, row[0], row[1], row[2], row[3],
                               row[4], bool(row[5]), row[6], row[7],
                               True, row[9])

    def promote_to_unknowns(self, limit: int | None = None,
                            threshold_bits: float | None = None) -> list[dict]:
        """Turn surprising, not-yet-promoted outcomes into cognition
        known unknowns, most surprising first. This is the S2 feed
        from surprise into introspection: the core names what
        surprised it instead of absorbing it silently. Idempotent —
        an outcome already carrying an unknown_id is never promoted
        twice. Requires a cognition engine; without one this is a
        configuration error, not a silent no-op."""
        if self.cognition is None:
            raise RuntimeError(
                "promote_to_unknowns requires a cognition engine")
        threshold = (self.surprise_threshold_bits if threshold_bits is None
                     else float(threshold_bits))
        if threshold <= 0:
            raise ValueError("threshold_bits must be positive")
        if limit is not None and int(limit) <= 0:
            raise ValueError("promote limit must be positive")
        rows = self.conn.execute(
            "SELECT id,stream,actual,predicted,surprise_bits,context"
            " FROM prediction_outcomes WHERE unknown_id IS NULL"
            " AND surprise_bits IS NOT NULL AND surprise_bits > ?"
            " ORDER BY surprise_bits DESC, id ASC",
            (threshold,)).fetchall()
        if limit is not None:
            rows = rows[:int(limit)]
        promoted = []
        for outcome_id, stream, actual, predicted, surprise, context in rows:
            question = (f"Surprising {stream} outcome: {actual!r}"
                        f" (surprise {surprise:.2f} bits"
                        + (f", predicted {predicted!r}" if predicted else
                           ", no prediction held")
                        + ")")
            ctx = f"prediction stream {stream}"
            if context:
                ctx += f"; context {context}"
            unknown = self.cognition.record_unknown(question, context=ctx)
            self.conn.execute(
                "UPDATE prediction_outcomes SET unknown_id=? WHERE id=?",
                (unknown.id, outcome_id))
            promoted.append({"outcome_id": outcome_id,
                             "unknown_id": unknown.id,
                             "question": question})
        self.conn.commit()
        return promoted

    def scheduled_surprise_review(self) -> dict:
        """Zero-argument body for a scheduler custom job, following
        the scheduled_self_query pattern: promote any surprising
        outcomes not yet named as unknowns, and report the summary.
        Without a cognition engine it still reports the summary and
        promotes nothing."""
        promoted = []
        if self.cognition is not None:
            promoted = self.promote_to_unknowns()
        return {"promoted_unknowns": [p["unknown_id"] for p in promoted],
                "summary": self.summary()}

    # -- reporting -------------------------------------------------------------
    def summary(self, stream: str | None = None) -> dict:
        if stream is None:
            rows = self.conn.execute(
                "SELECT surprise_bits,surprising,learned,unknown_id"
                " FROM prediction_outcomes").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT surprise_bits,surprising,learned,unknown_id"
                " FROM prediction_outcomes WHERE stream=?",
                (stream,)).fetchall()
        surprises = [float(r[0]) for r in rows if r[0] is not None]
        return {
            "stream": stream,
            "outcomes": len(rows),
            "mean_surprise_bits": (sum(surprises) / len(surprises)
                                   if surprises else None),
            "max_surprise_bits": max(surprises) if surprises else None,
            "surprising": sum(1 for r in rows if r[1]),
            "learned": sum(1 for r in rows if r[2]),
            "unlearned": sum(1 for r in rows if not r[2]),
            "promoted_to_unknowns": sum(1 for r in rows if r[3] is not None),
        }


# -- token-level prediction (the model's own distribution) -----------------

def next_token_distribution(model, context_ids: list[int]) -> list[float]:
    """The model's next-token probability distribution for a context
    (a list of token ids, as in model.generate). Returns plain floats
    summing to 1, in vocabulary order."""
    probs = model._next_probs([int(i) for i in context_ids])
    return [float(p) for p in probs]


def predict_next_token(model, context_ids: list[int]) -> TokenPrediction:
    """The model's most likely next token, excluding the PAD and BOS
    ids exactly as generate() does — the model never produces those
    as continuations. Probability is the token's probability in the
    unmodified distribution; entropy is over the full distribution."""
    from .model import PAD, BOS  # local: keeps numpy out of module import
    probs = model._next_probs([int(i) for i in context_ids])
    entropy = float(-sum(float(p) * math.log2(float(p))
                         for p in probs if float(p) > 0))
    masked = probs.copy()
    masked[PAD] = 0.0
    masked[BOS] = 0.0
    token_id = int(masked.argmax())
    return TokenPrediction(model.vocab.decode([token_id])[0], token_id,
                           float(probs[token_id]), entropy)


def token_surprise_bits(model, context_ids: list[int], actual_id: int) -> float:
    """Surprise of one actual next token, in bits: -log2 P(token).

    This is S1's bits-per-token for a single position: a token the
    model assigned high probability to costs few bits, a token it
    found unlikely costs many. The probability is strictly positive
    (softmax), so the surprise is always finite."""
    actual_id = int(actual_id)
    probs = model._next_probs([int(i) for i in context_ids])
    if not 0 <= actual_id < len(probs):
        raise ValueError(f"token id {actual_id} out of vocabulary range")
    prob = float(probs[actual_id])
    if prob <= 0:
        raise ValueError("model assigned zero probability to the actual token")
    return float(-math.log2(prob))


def predict_next_token_text(model, prompt: str) -> TokenPrediction:
    """Text form of predict_next_token: tokenize the prompt, prepend
    BOS as generate() does, and predict the continuation token."""
    from .model import BOS, tokenize
    ids = [BOS] + model.vocab.encode(tokenize(prompt))
    return predict_next_token(model, ids)


def token_surprise_text(model, prompt: str, actual_token: str) -> float:
    """Text form of token_surprise_bits: the surprise of one actual
    continuation token (a single token string) after the prompt."""
    from .model import BOS, tokenize
    ids = model.vocab.encode(tokenize(actual_token))
    if len(ids) != 1:
        raise ValueError("actual_token must tokenize to exactly one token")
    context = [BOS] + model.vocab.encode(tokenize(prompt))
    return token_surprise_bits(model, context, ids[0])
