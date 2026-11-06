"""Tests for the multiverse evolution experiment (owner direction
2026-11-04): branch creation, both epoch-selection strategies (A1
two-stage gate, A2 fraction term) plus the legacy score they replace,
the no-regression advancement rule, path pruning, differential
learning-rate scales, manifest integrity, and a tiny end-to-end run.

Everything here uses tiny models trained in-test in seconds; the full
10-generation experiment is training/run_multiverse.py, separate from
the unit suite by design."""
import json
import os
import tempfile
import unittest

from steelbound.model import SteelboundModel, Vocab, make_examples, tokenize
from steelbound.multiverse import (
    STRATEGY_A1, STRATEGY_A2, PathConfig, PathState, advancement,
    build_lr_scales, produce_candidate, prune_paths, run_multiverse,
    score_epoch_a2, score_epoch_legacy, select_epoch)

CORPUS = [
    "who are you : i am steelbound , a local first ai core .",
    "what is the ledger : the ledger is append only and hash chained .",
    "what is the gate : the gate fails closed and denies risky actions .",
    "can you approve yourself : no , i can never approve myself .",
    "the ledger is append only .",
    "the gate fails closed .",
] * 6
LESSONS = [
    "what is a zebra : a zebra is a striped animal .",
    "a zebra is a striped animal .",
] * 4
TARGET_PROBES = [("what is a zebra :", "striped")]


def _tiny_parent(tmp):
    seqs = [tokenize(s) for s in CORPUS]
    vocab = Vocab.build(seqs, min_count=1)
    enc = [vocab.encode(s) for s in seqs]
    X, y = make_examples(enc, context_len=4)
    model = SteelboundModel(vocab, context_len=4, embed_dim=12,
                            hidden_dim=24, seed=3)
    model.train(X, y, epochs=30, batch=16, lr=0.01)
    parent_dir = os.path.join(tmp, "parent")
    model.save_checkpoint(parent_dir, generation=0)
    return parent_dir, seqs


def _rec(epoch, val_loss, core, target):
    return {"epoch": epoch, "val_loss": val_loss,
            "core_score": core, "target_score": target}


class TestSelectionStrategies(unittest.TestCase):
    def test_a1_prefers_admissible_lowest_val_loss(self):
        records = [_rec(1, 3.0, 1.0, 0.5),   # admissible, worse loss
                   _rec(2, 2.0, 0.9, 1.0),   # inadmissible (core miss), best loss
                   _rec(3, 2.5, 1.0, 0.6)]   # admissible, best admissible loss
        idx, info = select_epoch(STRATEGY_A1, records, parent_target_score=0.5)
        self.assertEqual(records[idx]["epoch"], 3)
        self.assertFalse(info["fallback"])

    def test_a1_rejects_epochs_below_parent_target(self):
        records = [_rec(1, 2.0, 1.0, 0.4),   # target worse than parent
                   _rec(2, 3.0, 1.0, 0.9)]   # target fine
        idx, _ = select_epoch(STRATEGY_A1, records, parent_target_score=0.8)
        self.assertEqual(records[idx]["epoch"], 2)

    def test_a1_fallback_protects_core_first(self):
        records = [_rec(1, 1.0, 0.5, 1.0),   # best loss, worst core
                   _rec(2, 5.0, 0.9, 0.0)]   # bad loss, best core
        idx, info = select_epoch(STRATEGY_A1, records, parent_target_score=0.0)
        self.assertEqual(records[idx]["epoch"], 2)
        self.assertTrue(info["fallback"])

    def test_a2_penalty_is_bounded_regardless_of_probe_count(self):
        perfect = _rec(1, 4.0, 1.0, 1.0)
        worst = _rec(2, 4.0, 0.0, 0.0)
        self.assertAlmostEqual(score_epoch_a2(worst) - score_epoch_a2(perfect), 1.0)
        # The legacy score the strategies replace grows without bound
        # as the declared probe set grows — the Gen 1 failure mode.
        legacy_gap_10 = (score_epoch_legacy(worst, 10, 10)
                         - score_epoch_legacy(perfect, 10, 10))
        legacy_gap_78 = (score_epoch_legacy(worst, 10, 78)
                         - score_epoch_legacy(perfect, 10, 78))
        self.assertGreater(legacy_gap_78, legacy_gap_10)
        self.assertGreater(legacy_gap_78, 20.0)

    def test_a2_balances_loss_against_recall(self):
        # Epoch 1 fits the corpus slightly better but learned nothing;
        # epoch 2 gives up 0.2 nats and learned the lessons. A2's
        # bounded term (0.5) still lets the lesson win here, but a
        # 2-nat corpus sacrifice could not be bought by recall alone.
        records = [_rec(1, 4.0, 1.0, 0.0), _rec(2, 4.2, 1.0, 1.0)]
        idx, _ = select_epoch(STRATEGY_A2, records)
        self.assertEqual(records[idx]["epoch"], 2)
        records = [_rec(1, 4.0, 1.0, 0.0), _rec(2, 6.0, 1.0, 1.0)]
        idx, _ = select_epoch(STRATEGY_A2, records)
        self.assertEqual(records[idx]["epoch"], 1)

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            select_epoch("nope", [_rec(1, 1.0, 1.0, 1.0)])


class TestAdvancement(unittest.TestCase):
    PARENT = {"val_perplexity": 50.0, "probe_score": 1.0}

    def test_within_tolerance_advances(self):
        child = {"val_perplexity": 50.9, "probe_score": 1.0}
        ok, reasons = advancement(self.PARENT, child, 0.5, 0.6)
        self.assertTrue(ok, reasons)

    def test_perplexity_beyond_band_refused(self):
        child = {"val_perplexity": 51.5, "probe_score": 1.0}
        ok, reasons = advancement(self.PARENT, child, 0.5, 0.6)
        self.assertFalse(ok)
        self.assertTrue(any("perplexity" in r for r in reasons))

    def test_probe_regression_refused(self):
        child = {"val_perplexity": 40.0, "probe_score": 0.9}
        ok, _ = advancement(self.PARENT, child, 0.5, 0.6)
        self.assertFalse(ok)

    def test_target_regression_refused(self):
        child = {"val_perplexity": 40.0, "probe_score": 1.0}
        ok, _ = advancement(self.PARENT, child, 0.5, 0.4)
        self.assertFalse(ok)


class TestLrScales(unittest.TestCase):
    def test_grown_entries_full_rate_inherited_scaled(self):
        vocab = Vocab.build([[ "a", "b", "c"]], min_count=1)
        model = SteelboundModel(vocab, context_len=2, embed_dim=4,
                                hidden_dim=6, seed=1)
        pv, ph = len(vocab), model.hidden_dim
        model.expand_vocab(["zebra"], seed=1)
        model.expand_hidden(3, seed=1)
        scales = build_lr_scales(model, pv, ph, 0.25)
        self.assertTrue(all(v == 0.25 for v in scales["E"][:pv].flat))
        self.assertTrue(all(v == 1.0 for v in scales["E"][pv:].flat))
        self.assertTrue(all(v == 0.25 for v in scales["W2"][:ph, :pv].flat))
        self.assertTrue(all(v == 1.0 for v in scales["W2"][ph:, :].flat))
        self.assertTrue(all(v == 1.0 for v in scales["W2"][:ph, pv:].flat))
        self.assertTrue(all(v == 1.0 for v in scales["b1"][ph:].flat))
        for name, arr in scales.items():
            self.assertEqual(arr.shape, getattr(model, name).shape)


class TestProduceCandidate(unittest.TestCase):
    def test_candidate_checkpoint_and_record_both_strategies(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent_dir, seqs = _tiny_parent(tmp)
            lessons = [tokenize(s) for s in LESSONS]
            for i, strategy in enumerate((STRATEGY_A1, STRATEGY_A2)):
                out = os.path.join(tmp, f"child{i}")
                record = produce_candidate(
                    parent_dir, seqs + lessons, seqs, out,
                    strategy=strategy,
                    variation={"kind": "test_variation"},
                    epochs=2, lr=0.01, seed=5, hidden_growth=4,
                    inherited_lr_scale=0.25, target_probes=TARGET_PROBES)
                self.assertEqual(record["generation"], 1)
                self.assertIn(record["verdict"], ("advanced", "refused"))
                self.assertIn("zebra", record["vocab_added_tokens"])
                # Checkpoint is intact and hash-verified on load, and
                # the manifest hash matches the candidate record.
                child = SteelboundModel.load_checkpoint(out)
                self.assertEqual(child.manifest["weights_sha256"],
                                 record["child_weights_sha256"])
                self.assertEqual(record["parent_weights_sha256"],
                                 SteelboundModel.load_checkpoint(parent_dir)
                                 .manifest["weights_sha256"])


class TestPruneAndRun(unittest.TestCase):
    def test_prune_keeps_best_three(self):
        states = []
        metrics = {}
        for i, ppl in enumerate((40.0, 30.0, 50.0, 35.0)):
            cfg = PathConfig(name=f"p{i}", strategy=STRATEGY_A2)
            states.append(PathState(config=cfg, tip_dir="/x", tip_id=f"p{i}-g1"))
            metrics[f"p{i}"] = {"child_metrics": {"val_perplexity": ppl},
                                "child_target_score": 0.5}
        pruned = prune_paths(states, metrics, max_living=3)
        self.assertEqual([s.config.name for s in pruned], ["p2"])
        self.assertFalse(states[2].alive)

    def test_tiny_multiverse_run_writes_manifest_and_no_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent_dir, seqs = _tiny_parent(tmp)
            lessons = [tokenize(s) for s in LESSONS]
            out_root = os.path.join(tmp, "mv")
            configs = [
                PathConfig(name="ta", strategy=STRATEGY_A1, epochs=2,
                           teacher_repeat=1, curriculum_repeat=1,
                           hidden_growth_first=4, hidden_growth_later=2,
                           inherited_lr_scale=0.25, seed_base=1),
                PathConfig(name="tb", strategy=STRATEGY_A2, epochs=2,
                           teacher_repeat=1, curriculum_repeat=1,
                           hidden_growth_first=4, hidden_growth_later=2,
                           seed_base=2),
            ]
            manifest = run_multiverse(
                root_checkpoint=parent_dir, out_root=out_root,
                own_train_seqs=seqs, curriculum_seqs=seqs[:6],
                teacher_seqs=lessons, val_seqs=seqs[:6],
                target_probes=TARGET_PROBES, path_configs=configs,
                max_generations=2)
            self.assertEqual(manifest["root"]["generation"], 0)
            self.assertGreaterEqual(len(manifest["candidates"]), 2)
            self.assertIn("final", manifest)
            # Manifest on disk matches the returned tree, and every
            # recorded child hash matches its checkpoint manifest.
            with open(os.path.join(out_root, "manifest.json")) as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk["experiment"], "multiverse")
            for cand in on_disk["candidates"]:
                with open(os.path.join(out_root, cand["checkpoint"],
                                       "manifest.json")) as f:
                    ck = json.load(f)
                self.assertEqual(ck["weights_sha256"],
                                 cand["child_weights_sha256"])
            # The promotion rule: no live pointer is ever written.
            self.assertFalse(os.path.exists(os.path.join(out_root, "current.json")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "current.json")))


if __name__ == "__main__":
    unittest.main()
