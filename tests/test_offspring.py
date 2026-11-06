"""Offspring cores: spawn with one variation, head-to-head eval,
owner-approved promotion or retirement, all ledgered in the parent.

Covers: spawn (seeded memory, genesis entry, parent ledger, sandbox
isolation), the one-living cap, variation validation (including the
permissions-subset rule), the unpromoted-child spawn ban, head-to-head
evaluation (promotion_worthy / no_improvement / regression), promotion
(approval gate, genome adoption persisted across reopen, child gains
the right to spawn), retirement (cap freed, sandbox left on disk),
seed-hash determinism, and the CLI surface.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator
from steelbound.cli import main as cli_main
from steelbound.gate import Risk
from steelbound.offspring import capture_genome, seed_hash


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = Core(Settings(data_dir=self.tmp.name))
        self.orch = Orchestrator(self.core)
        self.off = self.core.offspring

    def tearDown(self):
        self.core.close(); self.tmp.cleanup()

    def actions(self, core=None):
        c = core or self.core
        return [e.action for e in c.ledger.entries(10000)]

    def seed_parent(self):
        self.orch.remember_fact("Steelbound is a local-first AI core.",
                                source="owner")
        self.orch.remember_fact("The gate fails closed.", source="owner")

    def heldout(self):
        return [{"content": f"Held-out fact {i}.", "source": "heldout",
                 "confidence": conf}
                for i, conf in enumerate((0.85, 0.86, 0.87))]


class TestSpawn(Base):
    def test_spawn_seeds_memory_and_ledgers(self):
        self.seed_parent()
        record = self.off.spawn({"kind": "consolidation_auto_threshold",
                                 "value": 0.85})
        self.assertEqual(record.status, "spawned")
        self.assertEqual(record.generation, 1)
        self.assertEqual(record.genome["consolidation_auto_threshold"], 0.85)
        self.assertIn("OFFSPRING_SPAWN", self.actions())
        # The child sandbox is isolated: its own ledger, its own chain.
        child = Core(Settings(data_dir=record.sandbox_dir))
        try:
            self.assertIn("OFFSPRING_GENESIS", self.actions(child))
            self.assertTrue(child.ledger.verify())
            contents = [i.content for i in
                        child.memory.all_active(kind="semantic")]
            self.assertIn("Steelbound is a local-first AI core.", contents)
            self.assertIn("The gate fails closed.", contents)
            genesis = [e for e in child.ledger.entries(10000)
                       if e.action == "OFFSPRING_GENESIS"][0]
            self.assertEqual(genesis.payload["seed_hash"], record.seed_hash)
            self.assertIsNotNone(genesis.payload["parent_ledger_tip"])
        finally:
            child.close()
        # The child never wrote to the parent ledger beyond the spawn.
        self.assertEqual(self.actions().count("OFFSPRING_SPAWN"), 1)
        self.assertTrue(self.core.ledger.verify())

    def test_spawn_with_seed_limit_and_floor(self):
        self.seed_parent()
        self.orch.remember_fact("A weak fact.", source="owner")
        # Lower the weak fact's confidence below the floor via a fresh
        # episodic item is not possible for semantic items, so limit
        # alone decides here: only the first fact is seeded.
        record = self.off.spawn({"kind": "consolidation_auto_threshold",
                                 "value": 0.85}, seed_limit=1)
        child = Core(Settings(data_dir=record.sandbox_dir))
        try:
            self.assertEqual(len(child.memory.all_active(kind="semantic")), 1)
        finally:
            child.close()

    def test_one_living_cap(self):
        self.off.spawn({"kind": "consolidation_auto_threshold", "value": 0.85})
        with self.assertRaises(ValueError):
            self.off.spawn({"kind": "consolidation_auto_threshold",
                            "value": 0.8})
        self.assertEqual(len(self.off.offspring()), 1)

    def test_max_living_configurable(self):
        self.off.max_living = 2
        self.off.spawn({"kind": "consolidation_auto_threshold", "value": 0.85})
        second = self.off.spawn({"kind": "consolidation_min_confidence",
                                 "value": 0.4})
        self.assertEqual(second.generation, 2)
        with self.assertRaises(ValueError):
            self.off.spawn({"kind": "consolidation_auto_threshold",
                            "value": 0.8})

    def test_generation_counts_lineage_and_attempts(self):
        first = self.off.spawn({"kind": "consolidation_auto_threshold",
                                "value": 0.85})
        self.off.retire(first.id, reason="test")
        second = self.off.spawn({"kind": "consolidation_auto_threshold",
                                 "value": 0.8})
        self.assertEqual(second.generation, 2)

    def test_variation_validation(self):
        with self.assertRaises(ValueError):
            self.off.spawn({"kind": "nope", "value": 1})
        with self.assertRaises(ValueError):
            self.off.spawn({"kind": "consolidation_auto_threshold"})
        with self.assertRaises(ValueError):
            self.off.spawn({"kind": "consolidation_auto_threshold",
                            "value": 1.5})
        with self.assertRaises(ValueError):
            # Auto threshold below the parent's min confidence.
            self.off.spawn({"kind": "consolidation_auto_threshold",
                            "value": 0.4})
        with self.assertRaises(ValueError):
            # Min confidence above the parent's auto threshold.
            self.off.spawn({"kind": "consolidation_min_confidence",
                            "value": 0.95})
        with self.assertRaises(ValueError):
            # Unknown provider.
            self.off.spawn({"kind": "provider", "value": "mystery"})
        with self.assertRaises(ValueError):
            # OpenAI-compatible without model/base_url cannot be built.
            self.off.spawn({"kind": "provider", "value": "openai-compatible"})
        self.assertEqual(len(self.off.offspring()), 0)
        self.assertNotIn("OFFSPRING_SPAWN", self.actions())

    def test_permissions_subset_rule(self):
        # Parent does not auto-approve local writes: a child may not.
        with self.assertRaises(ValueError):
            self.off.spawn({"kind": "auto_approve_local_writes", "value": True})
        # A parent that does may spawn a narrower child.
        self.core.close()
        self.core = Core(Settings(data_dir=self.tmp.name,
                                  auto_approve_local_writes=True))
        self.off = self.core.offspring
        record = self.off.spawn({"kind": "auto_approve_local_writes",
                                 "value": False})
        child = Core(Settings(data_dir=record.sandbox_dir))
        try:
            self.assertFalse(child.gate.auto_approve_local_writes)
            # The child gate denies an unapproved local write.
            self.assertFalse(child.gate.evaluate(
                "write", Risk.WRITE_LOCAL, approved=False).allowed)
        finally:
            child.close()

    def test_unpromoted_child_cannot_spawn(self):
        record = self.off.spawn({"kind": "consolidation_auto_threshold",
                                 "value": 0.85})
        child = Core(Settings(data_dir=record.sandbox_dir))
        try:
            with self.assertRaises(ValueError):
                child.offspring.spawn(
                    {"kind": "consolidation_auto_threshold", "value": 0.8})
        finally:
            child.close()

    def test_seed_hash_deterministic(self):
        self.seed_parent()
        genome = capture_genome(self.core)
        variation = {"kind": "consolidation_auto_threshold", "value": 0.85}
        seeded = [{"content": i.content, "source": i.source,
                   "confidence": i.confidence}
                  for i in sorted(self.core.memory.all_active(kind="semantic"),
                                  key=lambda x: x.id)]
        first = self.off.spawn(variation)
        self.assertEqual(first.seed_hash, seed_hash(genome, variation, seeded))
        self.off.retire(first.id, reason="test")
        second = self.off.spawn(variation)
        self.assertEqual(second.seed_hash, first.seed_hash)


class TestEvaluate(Base):
    def test_promotion_worthy_threshold_drop(self):
        record = self.off.spawn({"kind": "consolidation_auto_threshold",
                                 "value": 0.85})
        ev = self.off.evaluate(record.id, heldout_items=self.heldout())
        self.assertEqual(ev.verdict, "promotion_worthy")
        self.assertEqual(ev.parent_targeted, 0)
        self.assertEqual(ev.child_targeted, 3)
        self.assertTrue(all(s["passed"] == s["total"]
                            for s in ev.child_evals.values()))
        self.assertEqual(ev.smoke, "ok")
        self.assertIn("OFFSPRING_EVAL", self.actions())
        self.assertEqual(self.off._get(record.id).status, "evaluated")
        self.assertTrue(self.core.ledger.verify())

    def test_no_improvement_on_tie(self):
        record = self.off.spawn({"kind": "consolidation_auto_threshold",
                                 "value": 0.85})
        # Held-out items at 0.95 promote under both genomes: a tie.
        ev = self.off.evaluate(record.id, heldout_items=[
            {"content": "Already high fact.", "source": "heldout",
             "confidence": 0.95}])
        self.assertEqual(ev.verdict, "no_improvement")
        self.assertEqual(ev.parent_targeted, 1)
        self.assertEqual(ev.child_targeted, 1)

    def test_regression_on_threshold_raise(self):
        record = self.off.spawn({"kind": "consolidation_auto_threshold",
                                 "value": 0.95})
        ev = self.off.evaluate(record.id, heldout_items=[
            {"content": "Mid fact.", "source": "heldout", "confidence": 0.9}])
        self.assertEqual(ev.verdict, "regression")
        self.assertEqual(ev.parent_targeted, 1)
        self.assertEqual(ev.child_targeted, 0)

    def test_provider_variation_unreachable_is_error(self):
        record = self.off.spawn({"kind": "provider", "value": "ollama"})
        ev = self.off.evaluate(record.id, heldout_items=self.heldout())
        # No Ollama server runs here, so the child's honesty suite
        # cannot complete: an unmeasurable provider swap scores as an
        # error, never as a claimed tie or improvement.
        self.assertEqual(ev.verdict, "error")

    def test_evaluate_unknown_and_wrong_status(self):
        with self.assertRaises(KeyError):
            self.off.evaluate(999)
        record = self.off.spawn({"kind": "consolidation_auto_threshold",
                                 "value": 0.85})
        self.off.retire(record.id, reason="test")
        with self.assertRaises(ValueError):
            self.off.evaluate(record.id)

    def test_evaluate_default_heldout_uses_parent_episodic(self):
        for conf in (0.85, 0.86, 0.87):
            self.core.memory.episode(f"Parent episodic {conf}.",
                                     source="owner", confidence=conf)
        record = self.off.spawn({"kind": "consolidation_auto_threshold",
                                 "value": 0.85})
        ev = self.off.evaluate(record.id)
        self.assertEqual(ev.verdict, "promotion_worthy")
        self.assertEqual(ev.child_targeted, 3)


class TestPromoteRetire(Base):
    def _worthy(self):
        record = self.off.spawn({"kind": "consolidation_auto_threshold",
                                 "value": 0.85})
        self.off.evaluate(record.id, heldout_items=self.heldout())
        return record

    def test_promote_requires_approval(self):
        record = self._worthy()
        with self.assertRaises(ValueError):
            self.off.promote(record.id, approved=False)
        self.assertEqual(self.off._get(record.id).status, "evaluated")

    def test_promote_refuses_non_worthy(self):
        record = self.off.spawn({"kind": "consolidation_auto_threshold",
                                 "value": 0.85})
        self.off.evaluate(record.id, heldout_items=[
            {"content": "High fact.", "source": "heldout", "confidence": 0.95}])
        with self.assertRaises(ValueError):
            self.off.promote(record.id, approved=True)

    def test_promote_adopts_genome_and_persists(self):
        record = self._worthy()
        promoted = self.off.promote(record.id, approved=True)
        self.assertEqual(promoted.status, "promoted")
        self.assertAlmostEqual(self.core.consolidation.auto_threshold, 0.85)
        self.assertIn("OFFSPRING_PROMOTE", self.actions())
        # The child ledger records its promotion, and the parent's
        # ledger continues unbroken.
        child = Core(Settings(data_dir=record.sandbox_dir))
        try:
            self.assertIn("OFFSPRING_PROMOTED", self.actions(child))
        finally:
            child.close()
        self.assertTrue(self.core.ledger.verify())
        # Adoption survives a restart.
        self.core.close()
        self.core = Core(Settings(data_dir=self.tmp.name))
        self.assertAlmostEqual(self.core.consolidation.auto_threshold, 0.85)

    def test_promoted_child_may_spawn(self):
        record = self._worthy()
        self.off.promote(record.id, approved=True)
        child = Core(Settings(data_dir=record.sandbox_dir))
        try:
            grandchild = child.offspring.spawn(
                {"kind": "consolidation_auto_threshold", "value": 0.8})
            self.assertEqual(grandchild.generation, 2)
            self.assertIn("OFFSPRING_SPAWN", self.actions(child))
        finally:
            child.close()

    def test_retire_frees_cap_and_keeps_sandbox(self):
        record = self.off.spawn({"kind": "consolidation_auto_threshold",
                                 "value": 0.85})
        self.off.evaluate(record.id, heldout_items=[
            {"content": "High fact.", "source": "heldout", "confidence": 0.95}])
        retired = self.off.retire(record.id, reason="tie; ends the line")
        self.assertEqual(retired.status, "retired")
        self.assertIn("OFFSPRING_RETIRE", self.actions())
        self.assertTrue(os.path.isdir(record.sandbox_dir))
        # The cap is freed: a new explicit spawn is allowed.
        nxt = self.off.spawn({"kind": "consolidation_auto_threshold",
                              "value": 0.8})
        self.assertEqual(nxt.generation, 2)
        with self.assertRaises(ValueError):
            self.off.retire(record.id, reason="again")
        with self.assertRaises(ValueError):
            self.off.promote(record.id, approved=True)


class TestCli(Base):
    def _run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_main(["--data-dir", self.tmp.name, *argv])
        return code, buf.getvalue()

    def test_cli_full_cycle(self):
        code, out = self._run_cli(
            "offspring", "spawn", "--variation",
            json.dumps({"kind": "consolidation_auto_threshold", "value": 0.85}))
        self.assertEqual(code, 0)
        self.assertIn("offspring 1: generation 1", out)
        code, out = self._run_cli("offspring", "list")
        self.assertEqual(code, 0)
        self.assertIn("status=spawned", out)
        heldout_path = os.path.join(self.tmp.name, "heldout.json")
        with open(heldout_path, "w", encoding="utf-8") as f:
            json.dump([{"content": f"Held-out fact {i}.", "source": "heldout",
                        "confidence": conf}
                       for i, conf in enumerate((0.85, 0.86, 0.87))], f)
        code, out = self._run_cli("offspring", "evaluate", "1",
                                  "--heldout", heldout_path)
        self.assertEqual(code, 0)
        self.assertIn("promotion_worthy", out)
        # Promotion without --approved is refused.
        code, out = self._run_cli("offspring", "promote", "1")
        self.assertEqual(code, 1)
        code, out = self._run_cli("offspring", "promote", "1", "--approved")
        self.assertEqual(code, 0)
        self.assertIn("promoted", out)
        code, out = self._run_cli("offspring", "list")
        self.assertIn("status=promoted", out)
        self.assertIn("verdict=promotion_worthy", out)

    def test_cli_spawn_bad_variation(self):
        code, out = self._run_cli(
            "offspring", "spawn", "--variation",
            json.dumps({"kind": "provider", "value": "mystery"}))
        self.assertEqual(code, 1)
