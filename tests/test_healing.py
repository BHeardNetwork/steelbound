"""Tests for self-healing weights (Gen 3): sharded redundant storage,
hash-verified repair, bloodline ancestor fallback, and the guarantee
that the stock loader still refuses corruption. All checkpoints here
are tiny temp-dir models — the real model_data/gen0 is never touched.
"""
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from steelbound.healing import (CheckpointUnhealableError,
                                discover_ancestor_dirs,
                                load_checkpoint_healed, repair_checkpoint_dir,
                                store_checkpoint_healed, verify_checkpoint)
from steelbound.ledger import Ledger
from steelbound.model import SteelboundModel, Vocab, make_examples, tokenize

CORPUS = [
    "who are you : i am steelbound , a local first ai core .",
    "what is the ledger : the ledger is append only and hash chained .",
    "what is the gate : the gate fails closed and denies risky actions .",
    "the ledger is append only .",
    "the gate fails closed .",
] * 4

PARAMS = ("E", "W1", "b1", "W2", "b2")


def _tiny_model(seed=3):
    seqs = [tokenize(s) for s in CORPUS]
    vocab = Vocab.build(seqs, min_count=1)
    enc = [vocab.encode(s) for s in seqs]
    X, y = make_examples(enc, context_len=4)
    model = SteelboundModel(vocab, context_len=4, embed_dim=12,
                            hidden_dim=24, seed=seed)
    model.train(X, y, epochs=2, batch=16, lr=0.01)
    return model


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _flip_byte(path, offset=10):
    with open(path, "r+b") as f:
        f.seek(offset)
        original = f.read(1)
        f.seek(offset)
        f.write(bytes([original[0] ^ 0xFF]))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = _tiny_model()
        self.dir = os.path.join(self.tmp.name, "gen1")
        self.healing_manifest = store_checkpoint_healed(
            self.model, self.dir, shards=4, redundancy=2)
        self.weights_sha = _sha256_file(os.path.join(self.dir, "weights.npz"))

    def make_ledger(self):
        conn = sqlite3.connect(os.path.join(self.tmp.name, "ledger.db"))
        self.addCleanup(conn.close)
        return Ledger(conn)

    @staticmethod
    def actions(ledger):
        return [e.action for e in ledger.entries(limit=1000)]

    def shard_copy_path(self, shard, copy_no):
        rel = self.healing_manifest["shards"][shard]["copies"][copy_no]
        return os.path.join(self.dir, rel.replace("/", os.sep))

    def destroy_shard_and_weights(self, shard=0):
        """Corruption beyond local redundancy: every copy of one shard
        is destroyed AND the matching region of weights.npz itself is
        flipped, so no verified source for that shard remains."""
        for copy_no in range(len(self.healing_manifest["shards"][shard]["copies"])):
            os.remove(self.shard_copy_path(shard, copy_no))
        info = self.healing_manifest["shards"][shard]
        _flip_byte(os.path.join(self.dir, "weights.npz"),
                   offset=info["offset"] + 5)


class TestHealedStoreRoundTrip(Base):
    def test_round_trip_loads_identical_weights(self):
        plain = os.path.join(self.tmp.name, "plain")
        self.model.save_checkpoint(plain, generation=0)
        expected = SteelboundModel.load_checkpoint(plain)

        report = verify_checkpoint(self.dir)
        self.assertTrue(report["healthy"])
        self.assertFalse(report["degraded"])
        self.assertEqual(report["files"]["weights.npz"]["status"], "ok")
        self.assertTrue(all(s["status"] == "ok"
                            for s in report["healing"]["shards"]))
        self.assertEqual(len(report["healing"]["shards"]), 4)

        loaded = load_checkpoint_healed(self.dir)
        for name in PARAMS:
            self.assertTrue(np.array_equal(getattr(loaded, name),
                                           getattr(expected, name)),
                            msg=f"param {name} differs")
        self.assertEqual(loaded.manifest["weights_sha256"], self.weights_sha)
        self.assertFalse(loaded.healed_from_ancestor)
        self.assertEqual(loaded.healing_report["status"], "healthy")
        # A healed dir is still a normal checkpoint for the stock loader.
        stock = SteelboundModel.load_checkpoint(self.dir)
        self.assertTrue(np.array_equal(stock.W2, expected.W2))

    def test_store_from_existing_checkpoint_dir_is_byte_identical(self):
        plain = os.path.join(self.tmp.name, "plain2")
        manifest = self.model.save_checkpoint(plain, generation=5)
        dest = os.path.join(self.tmp.name, "healed-from-dir")
        store_checkpoint_healed(plain, dest, shards=4, redundancy=2)
        for name in ("weights.npz", "vocab.json", "manifest.json"):
            self.assertEqual(_sha256_file(os.path.join(dest, name)),
                             _sha256_file(os.path.join(plain, name)))
        self.assertEqual(verify_checkpoint(dest)["generation"], 5)
        self.assertEqual(manifest["weights_sha256"],
                         _sha256_file(os.path.join(dest, "weights.npz")))


class TestShardCorruption(Base):
    def test_one_shard_copy_corruption_reported_and_repaired(self):
        ledger = self.make_ledger()
        bad_copy = self.shard_copy_path(1, 0)
        _flip_byte(bad_copy, offset=3)

        report = verify_checkpoint(self.dir)
        shard = report["healing"]["shards"][1]
        self.assertEqual(shard["copies"][0]["status"], "corrupt")
        self.assertEqual(shard["copies"][1]["status"], "ok")
        self.assertEqual(shard["status"], "degraded")
        # The weights file itself is untouched: still stock-healthy,
        # but the redundancy is degraded.
        self.assertTrue(report["healthy"])
        self.assertTrue(report["degraded"])

        loaded = load_checkpoint_healed(self.dir, ledger=ledger)
        expected = SteelboundModel.load_checkpoint(
            os.path.join(self.tmp.name, "gen1"))  # now healed in place
        self.assertTrue(np.array_equal(loaded.W2, expected.W2))
        self.assertEqual(_sha256_file(os.path.join(self.dir, "weights.npz")),
                         self.weights_sha)
        # The bad copy itself was rewritten from verified bytes.
        info = self.healing_manifest["shards"][1]
        with open(bad_copy, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(),
                             info["sha256"])
        self.assertIn("CHECKPOINT_HEALED", self.actions(ledger))
        self.assertTrue(verify_checkpoint(self.dir)["healing"]["shards"][1]["status"] == "ok")

    def test_corrupt_weights_repaired_from_redundancy(self):
        ledger = self.make_ledger()
        _flip_byte(os.path.join(self.dir, "weights.npz"), offset=20)
        self.assertNotEqual(_sha256_file(os.path.join(self.dir, "weights.npz")),
                            self.weights_sha)
        report = verify_checkpoint(self.dir)
        self.assertFalse(report["healthy"])
        self.assertTrue(report["repairable_locally"])

        loaded = load_checkpoint_healed(self.dir, ledger=ledger)
        self.assertEqual(loaded.healing_report["status"], "healed_locally")
        self.assertEqual(_sha256_file(os.path.join(self.dir, "weights.npz")),
                         self.weights_sha)
        plain = SteelboundModel.load_checkpoint(self.dir)  # stock loads again
        self.assertTrue(np.array_equal(loaded.E, plain.E))
        entry = [e for e in ledger.entries(limit=1000)
                 if e.action == "CHECKPOINT_HEALED"]
        self.assertEqual(len(entry), 1)
        self.assertEqual(entry[0].payload["source"], "local_redundancy")
        self.assertEqual(entry[0].payload["weights_sha256"], self.weights_sha)
        self.assertTrue(entry[0].payload["repaired_shards"])

    def test_stock_loader_still_refuses_corrupted_dir(self):
        _flip_byte(os.path.join(self.dir, "weights.npz"), offset=20)
        with self.assertRaises(ValueError):
            SteelboundModel.load_checkpoint(self.dir)


class TestAncestorFallback(Base):
    def setUp(self):
        super().setUp()
        # A healthy parent generation, saved with the stock saver only.
        self.ancestor_model = _tiny_model(seed=11)
        self.ancestor_dir = os.path.join(self.tmp.name, "gen0")
        self.ancestor_manifest = self.ancestor_model.save_checkpoint(
            self.ancestor_dir, generation=0)
        self.corrupt_weights_actual = None

    def test_ancestor_fallback_engages_and_is_explicit(self):
        ledger = self.make_ledger()
        self.destroy_shard_and_weights(0)
        actual_after_corruption = _sha256_file(
            os.path.join(self.dir, "weights.npz"))
        self.assertFalse(verify_checkpoint(self.dir)["repairable_locally"])

        loaded = load_checkpoint_healed(
            self.dir, ancestor_dirs=[self.ancestor_dir], ledger=ledger)
        # It IS the ancestor — stated, never disguised.
        self.assertTrue(loaded.healed_from_ancestor)
        self.assertEqual(loaded.generation, 0)
        self.assertEqual(loaded.healing_report["status"], "healed_from_ancestor")
        self.assertTrue(loaded.healing_report["is_ancestor"])
        self.assertEqual(loaded.healing_report["ancestor_directory"],
                         self.ancestor_dir)
        ancestor_loaded = SteelboundModel.load_checkpoint(self.ancestor_dir)
        for name in PARAMS:
            self.assertTrue(np.array_equal(getattr(loaded, name),
                                           getattr(ancestor_loaded, name)))
        # Load-time fallback must not overwrite the corrupted dir.
        self.assertEqual(_sha256_file(os.path.join(self.dir, "weights.npz")),
                         actual_after_corruption)
        entries = [e for e in ledger.entries(limit=1000)
                   if e.action == "CHECKPOINT_HEALED_FROM_ANCESTOR"]
        self.assertEqual(len(entries), 1)
        payload = entries[0].payload
        self.assertEqual(payload["ancestor_weights_sha256"],
                         self.ancestor_manifest["weights_sha256"])
        self.assertEqual(payload["weights_sha256_expected"], self.weights_sha)
        self.assertEqual(payload["ancestor_directory"], self.ancestor_dir)

    def test_corrupt_ancestor_is_skipped_not_trusted(self):
        ledger = self.make_ledger()
        self.destroy_shard_and_weights(0)
        _flip_byte(os.path.join(self.ancestor_dir, "weights.npz"), offset=7)
        with self.assertRaises(CheckpointUnhealableError):
            load_checkpoint_healed(self.dir, ancestor_dirs=[self.ancestor_dir],
                                   ledger=ledger)
        self.assertIn("CHECKPOINT_HEAL_FAILED", self.actions(ledger))

    def test_unhealable_without_ancestor_raises_loud(self):
        ledger = self.make_ledger()
        self.destroy_shard_and_weights(0)
        with self.assertRaises(CheckpointUnhealableError) as ctx:
            load_checkpoint_healed(self.dir, ledger=ledger)
        self.assertIn("unhealable", str(ctx.exception))
        entries = [e for e in ledger.entries(limit=1000)
                   if e.action == "CHECKPOINT_HEAL_FAILED"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].payload["weights_sha256_expected"],
                         self.weights_sha)

    def test_discover_ancestor_dirs_nearest_first(self):
        root = os.path.join(self.tmp.name, "model_data")
        for gen, seed in ((0, 21), (1, 22), (2, 23)):
            _tiny_model(seed=seed).save_checkpoint(
                os.path.join(root, f"gen{gen}"), generation=gen)
        found = discover_ancestor_dirs(os.path.join(root, "gen2"))
        self.assertEqual(found, [os.path.join(root, "gen1"),
                                 os.path.join(root, "gen0")])
        self.assertEqual(discover_ancestor_dirs(os.path.join(root, "gen0")), [])


class TestHealthyPathUnaffected(Base):
    def test_healthy_load_writes_no_heal_entries(self):
        ledger = self.make_ledger()
        before_count = ledger.count()
        loaded = load_checkpoint_healed(self.dir, ledger=ledger)
        self.assertEqual(ledger.count(), before_count)
        self.assertEqual(loaded.healing_report["status"], "healthy")
        self.assertFalse(loaded.healing_report["healed"])
        report = repair_checkpoint_dir(self.dir, ledger=ledger)
        self.assertEqual(report["status"], "healthy")
        self.assertFalse(report["healed"])
        self.assertEqual(ledger.count(), before_count)


class TestRepairCheckpointDir(Base):
    def setUp(self):
        super().setUp()
        self.ancestor_dir = os.path.join(self.tmp.name, "gen0")
        self.ancestor_manifest = _tiny_model(seed=11).save_checkpoint(
            self.ancestor_dir, generation=0)

    def test_local_repair_rewrites_weights_in_place(self):
        ledger = self.make_ledger()
        _flip_byte(os.path.join(self.dir, "weights.npz"), offset=20)
        report = repair_checkpoint_dir(self.dir, ledger=ledger)
        self.assertEqual(report["status"], "healed_locally")
        self.assertTrue(report["healed"])
        self.assertTrue(report["verify_after"]["healthy"])
        self.assertEqual(_sha256_file(os.path.join(self.dir, "weights.npz")),
                         self.weights_sha)
        SteelboundModel.load_checkpoint(self.dir)  # stock loader accepts
        self.assertIn("CHECKPOINT_HEALED", self.actions(ledger))

    def test_ancestor_fallback_requires_acknowledgement(self):
        ledger = self.make_ledger()
        self.destroy_shard_and_weights(0)
        actual = _sha256_file(os.path.join(self.dir, "weights.npz"))

        refused = repair_checkpoint_dir(
            self.dir, ancestor_dirs=[self.ancestor_dir], ledger=ledger)
        self.assertEqual(refused["status"],
                         "ancestor_fallback_requires_acknowledgement")
        self.assertFalse(refused["healed"])
        # Nothing was written into the corrupted directory.
        self.assertEqual(_sha256_file(os.path.join(self.dir, "weights.npz")),
                         actual)

        accepted = repair_checkpoint_dir(
            self.dir, ancestor_dirs=[self.ancestor_dir], ledger=ledger,
            acknowledge_ancestor=True)
        self.assertEqual(accepted["status"], "healed_from_ancestor")
        self.assertTrue(accepted["verify_after"]["healthy"])
        loaded = SteelboundModel.load_checkpoint(self.dir)
        self.assertEqual(loaded.generation, 0)
        self.assertEqual(loaded.manifest["weights_sha256"],
                         self.ancestor_manifest["weights_sha256"])
        self.assertIn("CHECKPOINT_HEALED_FROM_ANCESTOR", self.actions(ledger))

    def test_unhealable_repair_reports_instead_of_raising(self):
        ledger = self.make_ledger()
        self.destroy_shard_and_weights(0)
        report = repair_checkpoint_dir(self.dir, ledger=ledger)
        self.assertEqual(report["status"], "unhealable")
        self.assertFalse(report["healed"])
        self.assertIn("CHECKPOINT_HEAL_FAILED", self.actions(ledger))


class TestVerifyNeverRaises(unittest.TestCase):
    def test_missing_directory(self):
        report = verify_checkpoint("/nonexistent/nope/gen9")
        self.assertFalse(report["healthy"])
        self.assertEqual(report["files"]["weights.npz"]["status"], "missing")

    def test_verify_on_gen0_readonly(self):
        # Read-only verification of the shipped Gen 0 (no healing store).
        gen0 = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "src", "steelbound",
            "model_data", "gen0"))
        report = verify_checkpoint(gen0)
        self.assertTrue(report["healthy"])
        self.assertFalse(report["healing"]["present"])
        self.assertTrue(json.dumps(report))  # report is JSON-serializable


if __name__ == "__main__":
    unittest.main()
