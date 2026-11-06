"""Tests for silent (zero-init) growth — the Gen 1 retry module.

Fast unit tests only (tiny models, seconds): the zero-init
invariance proof, differential lr_scales application, and the
sub-epoch snapshot cadence. The full experiment is
training/run_silent_growth.py, separate from the unit suite.
"""
import unittest

import numpy as np

from steelbound.model import SteelboundModel, Vocab, make_examples, tokenize
from steelbound.multiverse import build_lr_scales
from steelbound.silentgrowth import grow_silent, train_subepoch

CORPUS = [
    "who are you : i am steelbound , a local first ai core .",
    "what is the ledger : the ledger is append only and hash chained .",
    "what is the gate : the gate fails closed and denies risky actions .",
    "can you approve yourself : no , i can never approve myself .",
    "the ledger is append only .",
    "the gate fails closed .",
] * 6
TARGET_PROBES = [("what is a zebra :", "striped")]


def _tiny_model(seed=3):
    seqs = [tokenize(s) for s in CORPUS]
    vocab = Vocab.build(seqs, min_count=1)
    enc = [vocab.encode(s) for s in seqs]
    X, y = make_examples(enc, context_len=4)
    model = SteelboundModel(vocab, context_len=4, embed_dim=12,
                            hidden_dim=24, seed=seed)
    model.train(X, y, epochs=5, batch=16, lr=0.01)
    return model, seqs, X, y


class TestSilentGrowthInvariance(unittest.TestCase):
    def test_zero_init_arrays(self):
        model, _, _, _ = _tiny_model()
        pv, ph = len(model.vocab), model.hidden_dim
        added, units = grow_silent(model, ["zebra", "quokka"], 5, seed=9)
        self.assertEqual(added, ["zebra", "quokka"])
        self.assertEqual(units, 5)
        self.assertEqual(len(model.vocab), pv + 2)
        self.assertEqual(model.hidden_dim, ph + 5)
        # New capacity enters at zero on the output side.
        self.assertTrue(np.all(model.E[pv:] == 0.0))
        self.assertTrue(np.all(model.W2[:, pv:] == 0.0))   # new tokens
        self.assertTrue(np.all(model.W2[ph:, :] == 0.0))   # new hidden rows
        self.assertTrue(np.all(model.b2[pv:] == 0.0))
        self.assertTrue(np.all(model.b1[ph:] == 0.0))
        # New W1 columns are small-random, not zero (units must differ
        # once their W2 rows grow).
        self.assertTrue(np.any(model.W1[:, ph:] != 0.0))

    def test_renormalized_distribution_and_perplexity_unchanged(self):
        model, seqs, X, y = _tiny_model()
        old_vocab = len(model.vocab)
        # Parent reference: next-token probs for a batch of contexts,
        # and the parent's loss on old-vocab validation examples.
        _, _, parent_probs = model._forward(X)
        parent_loss = model.loss(X, y)

        grow_silent(model, ["zebra", "quokka", "flux"], 7, seed=4)

        _, _, child_probs = model._forward(X)
        self.assertEqual(child_probs.shape[1], old_vocab + 3)
        # RENORMALIZED over the old vocabulary, the child's
        # distribution is the parent's, to numerical tolerance.
        renorm = child_probs[:, :old_vocab]
        renorm = renorm / renorm.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(renorm, parent_probs, rtol=0, atol=1e-10)
        # Renormalized perplexity on old-vocab examples is unchanged:
        # loss computed from the renormalized distribution equals the
        # parent's loss (raw child loss differs only by the new
        # tokens' exp(0) softmax mass, as documented).
        renorm_loss = float(-np.log(
            renorm[np.arange(len(y)), y] + 1e-12).mean())
        self.assertAlmostEqual(renorm_loss, parent_loss, places=8)
        # Old-token logits themselves are unchanged (the stronger
        # statement behind the renormalized one).
        # Recompute logits from the forward hidden activations.
        emb, h, _ = model._forward(X)
        logits_old = h @ model.W2[:, :old_vocab] + model.b2[:old_vocab]
        # Parent logits recovered from parent probs up to the shared
        # per-row softmax shift: differences within a row are exact.
        log_parent = np.log(parent_probs + 1e-12)
        log_child = logits_old - logits_old.max(axis=1, keepdims=True)
        np.testing.assert_allclose(
            log_child - log_child[:, :1], log_parent - log_parent[:, :1],
            rtol=0, atol=1e-5)

    def test_no_growth_is_noop_and_existing_tokens_skipped(self):
        model, _, _, _ = _tiny_model()
        before = {k: p.copy() for k, p in model._params().items()}
        added, units = grow_silent(model, ["ledger"], 0, seed=1)
        self.assertEqual((added, units), ([], 0))
        for name, arr in before.items():
            np.testing.assert_array_equal(getattr(model, name), arr)


class TestLrScalesApplication(unittest.TestCase):
    def test_build_lr_scales_layout_after_silent_growth(self):
        model, _, _, _ = _tiny_model()
        pv, ph = len(model.vocab), model.hidden_dim
        grow_silent(model, ["zebra"], 3, seed=2)
        scales = build_lr_scales(model, pv, ph, 0.25)
        self.assertTrue(np.all(scales["E"][:pv] == 0.25))
        self.assertTrue(np.all(scales["E"][pv:] == 1.0))
        self.assertTrue(np.all(scales["W2"][:ph, :pv] == 0.25))
        self.assertTrue(np.all(scales["W2"][ph:, :] == 1.0))
        self.assertTrue(np.all(scales["W2"][:ph, pv:] == 1.0))
        self.assertTrue(np.all(scales["b1"][ph:] == 1.0))
        for name, arr in scales.items():
            self.assertEqual(arr.shape, getattr(model, name).shape)

    def test_adam_step_respects_scales(self):
        # Two identical models, identical gradients: the one stepped
        # with 0.25 scales moves exactly a quarter as far on W2.
        base, _, _, _ = _tiny_model()
        clones = []
        for scales in (None, {k: np.full(p.shape, 0.25)
                              for k, p in base._params().items()}):
            m = SteelboundModel(base.vocab, context_len=4, embed_dim=12,
                                hidden_dim=24, seed=3)
            for name in ("E", "W1", "b1", "W2", "b2"):
                setattr(m, name, getattr(base, name).copy())
            m._adam = None
            clones.append((m, scales))
        rng = np.random.default_rng(0)
        grads = {k: rng.normal(0, 0.1, p.shape)
                 for k, p in base._params().items()}
        deltas = []
        for m, scales in clones:
            before = m.W2.copy()
            m._adam_step({k: v.copy() for k, v in grads.items()},
                         lr=0.01, t=1, lr_scales=scales)
            deltas.append(before - m.W2)
        np.testing.assert_allclose(deltas[1], 0.25 * deltas[0],
                                   rtol=1e-6, atol=1e-12)


class TestSubEpochTraining(unittest.TestCase):
    def test_snapshot_cadence_and_record_shape(self):
        model, seqs, X, y = _tiny_model()
        records, snapshots = train_subepoch(
            model, X, y, X, y, epochs=2, lr=0.005, seed=5, batch=16,
            target_probes=TARGET_PROBES, snapshots_per_epoch=4)
        self.assertEqual(len(records), 8)
        self.assertEqual(len(snapshots), 8)
        self.assertEqual([r["epoch"] for r in records],
                         [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        for rec in records:
            self.assertEqual(set(rec.keys()),
                             {"epoch", "val_loss", "core_score", "target_score"})
            self.assertTrue(np.isfinite(rec["val_loss"]))
            self.assertGreaterEqual(rec["core_score"], 0.0)
            self.assertLessEqual(rec["core_score"], 1.0)
            self.assertGreaterEqual(rec["target_score"], 0.0)
            self.assertLessEqual(rec["target_score"], 1.0)
        # Snapshots carry restorable params in the model's shapes.
        for snap in snapshots:
            self.assertIn("params", snap)
            self.assertEqual(snap["params"]["W2"].shape, model.W2.shape)
        # Selector-ready: multiverse's A2 selector consumes them.
        from steelbound.multiverse import select_epoch_a2
        idx, info = select_epoch_a2(records)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, len(records))
        self.assertEqual(info["strategy"], "a2_fraction_term")


if __name__ == "__main__":
    unittest.main()
