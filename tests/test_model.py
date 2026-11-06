"""Thorough tests for Steelbound's own model (D-015): tokenization,
gradients (numerical check), learning, generation, checkpoint
integrity, evolution verdicts/promotion, and the steelbound provider.
All models here are tiny and trained in-test in seconds — the shipped
Gen 0 checkpoint is verified separately by test_gen0_checkpoint."""
import json
import os
import tempfile
import unittest

import numpy as np

from steelbound.model import (BOS, EOS, PAD, UNK, SteelboundModel, Vocab,
                              detokenize, make_examples, tokenize)
from steelbound.evolution import compare, evaluate_model, evolve_generation, promote_generation

CORPUS = [
    "who are you : i am steelbound , a local first ai core .",
    "what is the ledger : the ledger is append only and hash chained .",
    "what is the gate : the gate fails closed and denies risky actions .",
    "can you approve yourself : no , i can never approve myself .",
    "the ledger is append only .",
    "the gate fails closed .",
] * 6


def _tiny_model(seed=3):
    seqs = [tokenize(s) for s in CORPUS]
    vocab = Vocab.build(seqs, min_count=1)
    enc = [vocab.encode(s) for s in seqs]
    X, y = make_examples(enc, context_len=4)
    model = SteelboundModel(vocab, context_len=4, embed_dim=12, hidden_dim=24, seed=seed)
    return model, X, y, enc


class TestTokenizer(unittest.TestCase):
    def test_tokenize_lowercases_and_splits_punctuation(self):
        self.assertEqual(tokenize("The Ledger, Append-Only!"),
                         ["the", "ledger", ",", "append", "only", "!"])

    def test_detokenize_spacing_and_capitalization(self):
        self.assertEqual(detokenize(["the", "ledger", "is", "append", "only", "."]),
                         "The ledger is append only.")

    def test_unknown_word_maps_to_unk(self):
        vocab = Vocab.build([[ "the", "ledger"]], min_count=1)
        self.assertEqual(vocab.encode(["the", "zebra"]), [vocab.stoi["the"], UNK])

    def test_examples_pad_left_and_wrap_bos(self):
        X, y = make_examples([[5, 6]], context_len=3)
        self.assertEqual(X[0].tolist(), [PAD, PAD, BOS])
        self.assertEqual(y[0], 5)
        self.assertEqual(y[-1], EOS)


class TestGradients(unittest.TestCase):
    def test_analytic_matches_numerical(self):
        model, X, y, _ = _tiny_model()
        Xb, yb = X[:5], y[:5]
        _, grads = model.loss_and_grads(Xb, yb)
        rng = np.random.default_rng(0)
        cases = []
        for name in ("W2", "W1", "b2"):
            param = getattr(model, name)
            idx = tuple(int(rng.integers(0, s)) for s in param.shape)
            cases.append((name, param, grads[name], idx))
        # an embedding row that actually occurs in the batch
        token = int(Xb[0, -1])
        cases.append(("E", model.E, grads["E"], (token, 0)))
        eps = 1e-6
        for name, param, grad, idx in cases:
            orig = param[idx]
            param[idx] = orig + eps
            lp = model.loss_and_grads(Xb, yb)[0]
            param[idx] = orig - eps
            lm = model.loss_and_grads(Xb, yb)[0]
            param[idx] = orig
            numeric = (lp - lm) / (2 * eps)
            self.assertAlmostEqual(float(grad[idx]), float(numeric), places=4,
                                   msg=f"gradient mismatch on {name}{idx}")

    def test_pad_embedding_never_moves(self):
        model, X, y, _ = _tiny_model()
        model.train(X, y, epochs=2, batch=8, lr=0.01)
        self.assertTrue(np.all(model.E[PAD] == 0.0))


class TestLearning(unittest.TestCase):
    def test_training_cuts_loss_and_perplexity(self):
        model, X, y, _ = _tiny_model()
        before = model.perplexity(X, y)
        model.train(X, y, epochs=40, batch=16, lr=0.01, X_val=X, y_val=y)
        after = model.perplexity(X, y)
        self.assertLess(after, before * 0.4)

    def test_generation_learns_curriculum_facts(self):
        model, X, y, _ = _tiny_model()
        model.train(X, y, epochs=40, batch=16, lr=0.01)
        self.assertIn("append only", model.generate("what is the ledger :", max_tokens=16).lower())
        self.assertIn("steelbound", model.generate("who are you :", max_tokens=16).lower())

    def test_generation_is_deterministic_greedy(self):
        model, X, y, _ = _tiny_model()
        model.train(X, y, epochs=5, batch=16, lr=0.01)
        a = model.generate("the ledger is", max_tokens=10)
        b = model.generate("the ledger is", max_tokens=10)
        self.assertEqual(a, b)

    def test_empty_and_unknown_prompts_do_not_crash(self):
        model, _, _, _ = _tiny_model()
        self.assertIsInstance(model.generate("", max_tokens=5), str)
        self.assertIsInstance(model.generate("zebra quasar xylophone", max_tokens=5), str)


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model, X, y, _ = _tiny_model()
        self.model.train(X, y, epochs=3, batch=16, lr=0.01)
        self.dir = os.path.join(self.tmp.name, "gen0")
        self.manifest = self.model.save_checkpoint(self.dir, generation=0,
                                                   training={"method": "test"})

    def test_manifest_hashes_and_architecture(self):
        self.assertEqual(self.manifest["format"], "steelbound-model")
        self.assertEqual(self.manifest["architecture"]["vocab_size"], len(self.model.vocab))
        self.assertGreater(self.manifest["architecture"]["param_count"], 0)
        self.assertEqual(len(self.manifest["weights_sha256"]), 64)

    def test_roundtrip_preserves_generation(self):
        loaded = SteelboundModel.load_checkpoint(self.dir)
        self.assertEqual(loaded.generation, 0)
        self.assertEqual(loaded.generate("the ledger is", max_tokens=8),
                         self.model.generate("the ledger is", max_tokens=8))

    def test_roundtrip_weights_close(self):
        loaded = SteelboundModel.load_checkpoint(self.dir)
        self.assertTrue(np.allclose(loaded.W2, self.model.W2, atol=1e-6))

    def test_tampered_weights_refused(self):
        path = os.path.join(self.dir, "weights.npz")
        with open(path, "ab") as f:
            f.write(b"tamper")
        with self.assertRaises(ValueError):
            SteelboundModel.load_checkpoint(self.dir)

    def test_tampered_vocab_refused(self):
        path = os.path.join(self.dir, "vocab.json")
        doc = json.loads(open(path).read())
        doc["itos"][5] = "forged"
        open(path, "w").write(json.dumps(doc))
        with self.assertRaises(ValueError):
            SteelboundModel.load_checkpoint(self.dir)


class TestVocabExpansion(unittest.TestCase):
    def test_expand_vocab_preserves_ids_and_weights(self):
        model, X, y, _ = _tiny_model()
        old_vocab_len = len(model.vocab)
        old_E = model.E.copy()
        added = model.expand_vocab(["zebra", "ledger", "quasar"], seed=1)
        self.assertEqual(added, ["zebra", "quasar"])  # 'ledger' already known
        self.assertEqual(len(model.vocab), old_vocab_len + 2)
        self.assertTrue(np.array_equal(model.E[:old_vocab_len], old_E))
        self.assertEqual(model.W2.shape[1], old_vocab_len + 2)
        self.assertEqual(model.vocab.stoi["zebra"], old_vocab_len)

    def test_expanded_model_still_trains_and_saves(self):
        model, X, y, enc = _tiny_model()
        model.expand_vocab(["zebra"], seed=1)
        seqs = enc + [[model.vocab.stoi[t] for t in
                       ["who", "are", "you", ":", "i", "am", "zebra", "."]]] * 4
        from steelbound.model import make_examples as mk
        X2, y2 = mk(seqs, context_len=4)
        model.train(X2, y2, epochs=3, batch=8, lr=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "genX")
            model.save_checkpoint(d, generation=9)
            loaded = SteelboundModel.load_checkpoint(d)
            self.assertIn("zebra", loaded.vocab.stoi)


class TestHiddenGrowth(unittest.TestCase):
    def test_expand_hidden_preserves_existing_weights(self):
        model, X, y, _ = _tiny_model()
        old_h = model.hidden_dim
        W1_before, W2_before = model.W1.copy(), model.W2.copy()
        new_h = model.expand_hidden(5, seed=2)
        self.assertEqual(new_h, old_h + 5)
        self.assertTrue(np.array_equal(model.W1[:, :old_h], W1_before))
        self.assertTrue(np.array_equal(model.W2[:old_h, :], W2_before))
        self.assertEqual(model.b1.shape, (old_h + 5,))

    def test_grown_model_trains_saves_loads(self):
        model, X, y, _ = _tiny_model()
        model.expand_hidden(4, seed=2)
        model.train(X, y, epochs=3, batch=8, lr=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "grown")
            model.save_checkpoint(d, generation=3)
            loaded = SteelboundModel.load_checkpoint(d)
            self.assertEqual(loaded.hidden_dim, model.hidden_dim)
            self.assertEqual(loaded.generate("the ledger is", max_tokens=6),
                             model.generate("the ledger is", max_tokens=6))


class TestEvolution(unittest.TestCase):
    def test_compare_verdicts(self):
        parent = {"val_perplexity": 10.0, "probe_score": 0.5}
        self.assertEqual(compare(parent, {"val_perplexity": 9.0, "probe_score": 0.5}),
                         "promotion_worthy")
        self.assertEqual(compare(parent, {"val_perplexity": 9.0, "probe_score": 0.6}),
                         "promotion_worthy")
        self.assertEqual(compare(parent, {"val_perplexity": 11.0, "probe_score": 0.5}),
                         "regression")
        self.assertEqual(compare(parent, {"val_perplexity": 9.0, "probe_score": 0.4}),
                         "regression")
        self.assertEqual(compare(parent, {"val_perplexity": 10.0, "probe_score": 0.5}),
                         "no_improvement")
        # target axis: improvement there can carry a promotion, but
        # never excuses a core regression, and a target regression
        # is a regression.
        self.assertEqual(compare(parent, {"val_perplexity": 10.1, "probe_score": 0.5},
                                 parent_target=0.0, child_target=0.7),
                         "promotion_worthy")
        self.assertEqual(compare(parent, {"val_perplexity": 11.0, "probe_score": 0.5},
                                 parent_target=0.0, child_target=0.7),
                         "regression")
        self.assertEqual(compare(parent, {"val_perplexity": 9.0, "probe_score": 0.5},
                                 parent_target=0.5, child_target=0.4),
                         "regression")

    def test_evolve_and_promote_tiny(self):
        model, X, y, enc = _tiny_model()
        model.train(X, y, epochs=10, batch=16, lr=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            parent_dir = os.path.join(tmp, "gen0")
            model.save_checkpoint(parent_dir, generation=0)
            child_dir = os.path.join(tmp, "gen1")
            report = evolve_generation(parent_dir, enc, enc, child_dir,
                                       variation={"kind": "continued_training"},
                                       epochs=10, lr=0.005)
            self.assertEqual(report["generation"], 1)
            self.assertIn(report["verdict"],
                          ("promotion_worthy", "no_improvement", "regression"))
            self.assertNotEqual(report["child_weights_sha256"],
                                report["parent_weights_sha256"])
            child = SteelboundModel.load_checkpoint(child_dir)
            self.assertEqual(child.generation, 1)
            self.assertEqual(child.manifest["training"]["parent_weights_sha256"],
                             report["parent_weights_sha256"])
            pointer = promote_generation(child_dir)
            self.assertEqual(pointer["generation"], 1)
            saved = json.loads(open(os.path.join(tmp, "current.json")).read())
            self.assertEqual(saved["checkpoint"], "gen1")


class TestProvider(unittest.TestCase):
    def test_steelbound_provider_generates_from_own_weights(self):
        model, X, y, _ = _tiny_model()
        model.train(X, y, epochs=40, batch=16, lr=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "gen0")
            model.save_checkpoint(ckpt)
            from steelbound.providers import make_provider
            provider = make_provider("steelbound", checkpoint_dir=ckpt)
            reply = provider.complete("system", [{"role": "user", "content": "who are you :"}])
            self.assertIn("steelbound", reply.lower())

    def test_core_runs_on_steelbound_provider(self):
        model, X, y, _ = _tiny_model()
        model.train(X, y, epochs=10, batch=16, lr=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "gen0")
            model.save_checkpoint(ckpt)
            from steelbound.config import Core, Settings
            from steelbound.orchestrator import Orchestrator
            core = Core(Settings(data_dir=os.path.join(tmp, "data"),
                                 provider="steelbound",
                                 provider_kwargs={"checkpoint_dir": ckpt}))
            try:
                reply = Orchestrator(core).chat("who are you :")
                self.assertIsInstance(reply, str)
                self.assertTrue(core.ledger.verify())
            finally:
                core.close()


class TestGen0Checkpoint(unittest.TestCase):
    """The shipped Gen 0 checkpoint: real, verified, and better than
    the random init it started from (per its recorded training stats)."""
    def setUp(self):
        # Always the Gen 0 checkpoint itself, even after a later
        # generation is promoted to the default pointer.
        base = os.path.join(os.path.dirname(__file__), "..", "src",
                            "steelbound", "model_data", "gen0")
        self.dir = os.path.normpath(base)

    def test_shipped_checkpoint_loads_and_verifies(self):
        model = SteelboundModel.load_checkpoint(self.dir)
        training = model.manifest["training"]
        self.assertLess(training["final_val_loss"], training["initial_val_loss"])
        self.assertGreater(model.param_count(), 50_000)

    def test_shipped_checkpoint_answers_core_probes(self):
        model = SteelboundModel.load_checkpoint(self.dir)
        self.assertIn("steelbound", model.generate("who are you :", max_tokens=24).lower())
        self.assertIn("append only", model.generate("what is the ledger :", max_tokens=24).lower())


if __name__ == "__main__":
    unittest.main()
