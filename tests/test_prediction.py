import math
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator
from steelbound.cli import main as cli_main
from steelbound.prediction import (
    FrequencyModel, PredictionEngine, predict_next_token,
    token_surprise_bits, predict_next_token_text, token_surprise_text)
from steelbound.model import SteelboundModel, Vocab, tokenize


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = Core(Settings(data_dir=self.tmp.name))
        self.orch = Orchestrator(self.core)
        self.pred = self.core.prediction

    def tearDown(self):
        self.core.close()
        self.tmp.cleanup()


class TestFrequencyModel(unittest.TestCase):
    def test_distribution_sums_to_one_including_unseen(self):
        m = FrequencyModel({"ok": 3, "error": 1})
        dist = m.distribution()
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)
        # Laplace alpha=1 over 2 observed + 1 unseen: denom 4 + 3 = 7.
        self.assertAlmostEqual(dist["ok"], 4 / 7, places=9)
        self.assertAlmostEqual(dist["error"], 2 / 7, places=9)
        self.assertAlmostEqual(dist["<unseen>"], 1 / 7, places=9)

    def test_surprise_is_negative_log2_probability(self):
        m = FrequencyModel({"ok": 3, "error": 1})
        self.assertAlmostEqual(m.surprise_bits("ok"), -math.log2(4 / 7), places=9)
        self.assertAlmostEqual(m.surprise_bits("timeout"),
                               -math.log2(1 / 7), places=9)

    def test_empty_model_has_no_prediction_and_no_surprise(self):
        m = FrequencyModel({})
        self.assertEqual(m.predict(), (None, None, {}, 0.0))
        self.assertIsNone(m.surprise_bits("anything"))
        self.assertEqual(m.entropy_bits(), 0.0)

    def test_tie_break_is_lexicographic(self):
        m = FrequencyModel({"beta": 2, "alpha": 2})
        predicted, _, _, _ = m.predict()
        self.assertEqual(predicted, "alpha")

    def test_bad_alpha_rejected(self):
        with self.assertRaises(ValueError):
            FrequencyModel({}, alpha=0)


class TestEngine(Base):
    def test_predict_on_empty_stream_has_no_prediction(self):
        p = self.pred.predict("brand_new_stream")
        self.assertIsNone(p.predicted)
        self.assertIsNone(p.probability)
        self.assertEqual(p.distribution, {})
        actions = [e.action for e in self.core.ledger.entries(100)]
        self.assertIn("PREDICTION_MADE", actions)
        self.assertTrue(self.core.ledger.verify())

    def test_first_observation_has_no_surprise(self):
        o = self.pred.observe("tool_result", "ok")
        self.assertIsNone(o.surprise_bits)
        self.assertFalse(o.surprising)

    def test_predict_after_history_and_surprise_before_update(self):
        for _ in range(3):
            self.pred.observe("tool_result", "ok")
        self.pred.observe("tool_result", "error")
        p = self.pred.predict("tool_result")
        self.assertEqual(p.predicted, "ok")
        self.assertAlmostEqual(p.probability, 4 / 7, places=9)
        # Surprise is measured BEFORE this observation updates the model:
        # counts are still ok=3, error=1, so P(error) = 2/7.
        o = self.pred.observe("tool_result", "error")
        self.assertAlmostEqual(o.surprise_bits, -math.log2(2 / 7), places=9)
        self.assertFalse(o.surprising)  # 1.81 bits < 3.0 threshold

    def test_novel_outcome_after_long_history_is_surprising(self):
        for _ in range(20):
            self.pred.observe("job", "ok")
        o = self.pred.observe("job", "fail")
        self.assertGreater(o.surprise_bits, 3.0)
        self.assertTrue(o.surprising)

    def test_prediction_resolution_rules(self):
        self.pred.observe("s", "a")
        self.pred.observe("s", "a")
        p = self.pred.predict("s")
        o = self.pred.observe("s", "a", prediction_id=p.id)
        self.assertEqual(o.prediction_id, p.id)
        self.assertEqual(o.predicted, p.predicted)
        with self.assertRaises(ValueError):
            self.pred.observe("s", "a", prediction_id=p.id)
        with self.assertRaises(ValueError):
            self.pred.observe("other", "a", prediction_id=p.id)
        with self.assertRaises(KeyError):
            self.pred.observe("s", "a", prediction_id=999999)

    def test_empty_stream_and_outcome_rejected(self):
        with self.assertRaises(ValueError):
            self.pred.predict("   ")
        with self.assertRaises(ValueError):
            self.pred.observe("s", "")

    def test_ledger_action_seeded_from_ledger_without_self_feedback(self):
        # A fresh core's ledger holds CORE_INIT; predicting twice must
        # not let the engine's own PREDICTION_MADE entries take over.
        first = self.pred.predict("ledger_action")
        self.assertEqual(first.predicted, "CORE_INIT")
        second = self.pred.predict("ledger_action")
        self.assertEqual(second.predicted, "CORE_INIT")
        self.orch.remember_fact("A fact.", source="owner")
        third = self.pred.predict("ledger_action")
        # CORE_INIT and FACT_STORED tie 1-1; lexicographic tie-break.
        self.assertEqual(third.predicted, "CORE_INIT")

    def test_learning_queue_order_and_mark_learned(self):
        for _ in range(20):
            self.pred.observe("job", "ok")
        surprising = self.pred.observe("job", "fail")
        dull = self.pred.observe("job", "ok")
        queue = self.pred.learning_queue(limit=10)
        self.assertEqual(queue[0].id, surprising.id)
        self.assertIn(dull.id, [o.id for o in queue])
        learned = self.pred.mark_learned(surprising.id)
        self.assertTrue(learned.learned)
        self.assertNotIn(surprising.id,
                         [o.id for o in self.pred.learning_queue(limit=10)])
        with self.assertRaises(ValueError):
            self.pred.mark_learned(surprising.id)
        with self.assertRaises(KeyError):
            self.pred.mark_learned(999999)
        with self.assertRaises(ValueError):
            self.pred.learning_queue(limit=0)

    def test_promote_surprises_to_known_unknowns_is_idempotent(self):
        for _ in range(20):
            self.pred.observe("job", "ok")
        self.pred.observe("job", "fail")
        made = self.pred.promote_to_unknowns()
        self.assertEqual(len(made), 1)
        self.assertIn("fail", made[0]["question"])
        unknowns = self.core.cognition.unknowns(status="open")
        self.assertEqual(len(unknowns), 1)
        self.assertEqual(unknowns[0].id, made[0]["unknown_id"])
        self.assertEqual(self.pred.promote_to_unknowns(), [])
        summary = self.pred.summary("job")
        self.assertEqual(summary["promoted_to_unknowns"], 1)
        self.assertEqual(summary["outcomes"], 21)

    def test_promote_without_cognition_is_an_error(self):
        conn = sqlite3.connect(":memory:")
        try:
            engine = PredictionEngine(None, conn, cognition=None)
            with self.assertRaises(RuntimeError):
                engine.promote_to_unknowns()
        finally:
            conn.close()

    def test_summary_aggregates(self):
        self.pred.observe("s", "a")
        self.pred.observe("s", "a")
        summary = self.pred.summary("s")
        self.assertEqual(summary["outcomes"], 2)
        # First observation has no surprise measure and is excluded;
        # the second scores -log2(2/3) under counts {a: 1}.
        self.assertAlmostEqual(summary["mean_surprise_bits"],
                               -math.log2(2 / 3), places=9)
        self.assertEqual(summary["unlearned"], 2)
        self.assertEqual(self.pred.summary("missing")["outcomes"], 0)

    def test_surprise_for_does_not_record(self):
        self.pred.observe("s", "a")
        self.pred.observe("s", "a")
        self.assertAlmostEqual(self.pred.surprise_for("s", "a"),
                               -math.log2(3 / 4), places=9)
        self.assertEqual(self.pred.summary("s")["outcomes"], 2)

    def test_scheduled_review_promotes_and_reports(self):
        for _ in range(20):
            self.pred.observe("job", "ok")
        self.pred.observe("job", "fail")
        review = self.pred.scheduled_surprise_review()
        self.assertEqual(len(review["promoted_unknowns"]), 1)
        self.assertEqual(review["summary"]["surprising"], 1)
        again = self.pred.scheduled_surprise_review()
        self.assertEqual(again["promoted_unknowns"], [])


class TestTokenPrediction(unittest.TestCase):
    def setUp(self):
        corpus = [tokenize("the ledger is append only"),
                  tokenize("the gate fails closed")] * 4
        self.vocab = Vocab.build(corpus, min_count=1)
        self.model = SteelboundModel(self.vocab, context_len=4,
                                     embed_dim=8, hidden_dim=16, seed=3)

    def test_predict_next_token_matches_distribution(self):
        ctx = self.vocab.encode(tokenize("the ledger is"))
        pred = predict_next_token(self.model, ctx)
        probs = self.model._next_probs(ctx)
        self.assertIn(pred.token, self.vocab.itos)
        self.assertAlmostEqual(pred.probability, float(probs[pred.token_id]),
                               places=9)
        expected_entropy = -sum(float(p) * math.log2(float(p))
                                for p in probs if float(p) > 0)
        self.assertAlmostEqual(pred.entropy_bits, expected_entropy, places=9)
        self.assertNotEqual(pred.token_id, 0)  # never PAD
        self.assertNotEqual(pred.token_id, 1)  # never BOS

    def test_token_surprise_is_negative_log2_probability(self):
        ctx = self.vocab.encode(tokenize("the ledger is"))
        actual = self.vocab.encode(tokenize("append"))[0]
        probs = self.model._next_probs(ctx)
        self.assertAlmostEqual(token_surprise_bits(self.model, ctx, actual),
                               -math.log2(float(probs[actual])), places=9)
        with self.assertRaises(ValueError):
            token_surprise_bits(self.model, ctx, len(self.vocab.itos) + 5)

    def test_text_helpers(self):
        pred = predict_next_token_text(self.model, "the ledger is")
        self.assertIn(pred.token, self.vocab.itos)
        surprise = token_surprise_text(self.model, "the ledger is", "append")
        self.assertGreater(surprise, 0)
        with self.assertRaises(ValueError):
            token_surprise_text(self.model, "the ledger is", "append only")


class TestCLI(Base):
    def test_cli_surface(self):
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "predict",
                                   "predict", "tool_result"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "predict",
                                   "observe", "tool_result",
                                   "--actual", "ok"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "predict",
                                   "summary"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "predict",
                                   "summary", "tool_result"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "predict",
                                   "queue"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "predict",
                                   "promote"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "predict",
                                   "review"]), 0)
        # Observe without --actual is a usage error.
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "predict",
                                   "observe", "tool_result"]), 2)
        # Learning an unknown outcome id fails cleanly, no traceback.
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "predict",
                                   "learn", "999999"]), 1)
        self.assertTrue(self.core.ledger.verify())


if __name__ == "__main__":
    unittest.main()
