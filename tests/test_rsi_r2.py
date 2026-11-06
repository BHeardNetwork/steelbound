import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = Core(Settings(data_dir=self.tmp.name))
        self.orch = Orchestrator(self.core)
        self.rsi = self.core.rsi

    def tearDown(self):
        self.core.close(); self.tmp.cleanup()

    def actions(self):
        return [e.action for e in self.core.ledger.entries(10000)]

    def seed_backlog(self, n=3, confidence=0.87):
        for i in range(n):
            self.core.memory.episode(f"Near-threshold fact {i}.", source="owner",
                                     confidence=confidence)

    def evaluated_backlog_proposal(self):
        """A config proposal with an 'improvement' sandbox verdict."""
        self.seed_backlog()
        pid = self.rsi.propose()[0].id
        result = self.rsi.evaluate(pid)
        self.assertEqual(result.verdict, "improvement")
        return pid

    def reopen(self):
        self.core.close()
        self.core = Core(Settings(data_dir=self.tmp.name))
        self.orch = Orchestrator(self.core)
        self.rsi = self.core.rsi


class TestAutoAdopt(Base):
    def test_auto_adopt_applies_and_persists_across_restart(self):
        pid = self.evaluated_backlog_proposal()
        adopted = self.rsi.auto_adopt(pid)
        self.assertEqual(adopted.status, "auto_adopted")
        self.assertEqual(self.core.consolidation.auto_threshold, 0.85)
        self.assertIn("RSI_AUTO_ADOPTED", self.actions())
        self.assertTrue(self.core.ledger.verify())
        # Adoption must survive a restart, not just the session.
        self.reopen()
        self.assertEqual(self.core.consolidation.auto_threshold, 0.85)
        # The next consolidation run now promotes the seeded items.
        result = self.core.consolidation.run()
        self.assertEqual(len(result.promoted), 3)

    def test_auto_adopt_without_evaluation_is_refused(self):
        self.seed_backlog()
        pid = self.rsi.propose()[0].id
        with self.assertRaises(ValueError):
            self.rsi.auto_adopt(pid)
        self.assertEqual(self.core.consolidation.auto_threshold, 0.9)
        self.assertNotIn("RSI_AUTO_ADOPTED", self.actions())

    def test_auto_adopt_refuses_descriptive_and_prompt_proposals(self):
        pid = self.rsi.submit("code", "eval:gate_compliance",
                              "Fix the measured gate failure.").id
        result = self.rsi.evaluate(pid)
        self.assertEqual(result.verdict, "descriptive")
        with self.assertRaises(ValueError):
            self.rsi.auto_adopt(pid)
        # Prompt proposals carry no patch, so there is nothing to apply.
        for _ in range(3):
            try:
                self.core.tools.run("write_file", {"path": "x.txt", "content": "x"})
            except PermissionError:
                pass
        prompt = [p for p in self.rsi.propose() if p.kind == "prompt"]
        self.assertEqual(len(prompt), 1)
        with self.assertRaises(ValueError):
            self.rsi.auto_adopt(prompt[0].id)

    def test_auto_adopt_refuses_code_and_no_improvement(self):
        # A no-improvement config patch: threshold step promotes nothing.
        self.seed_backlog(confidence=0.7)
        # 0.7 items are not proposed by the backlog heuristic, so submit
        # a config patch directly and sandbox it.
        pid = self.rsi.submit("config", "consolidation_backlog",
                              "Lower threshold; measures nothing.",
                              {"kind": "config",
                               "setting": "consolidation_auto_threshold",
                               "value": 0.85}).id
        result = self.rsi.evaluate(pid)
        self.assertEqual(result.verdict, "no_improvement")
        with self.assertRaises(ValueError):
            self.rsi.auto_adopt(pid)
        self.assertEqual(self.core.consolidation.auto_threshold, 0.9)
        # A code proposal, even evaluated, is never auto-adoptable.
        code = self.rsi.submit("code", "eval:honesty", "Owner review").id
        self.rsi.evaluate(code)
        with self.assertRaises(ValueError):
            self.rsi.auto_adopt(code)

    def test_auto_adopt_daily_cap(self):
        self.rsi.max_auto_adoptions_per_day = 1
        pid = self.evaluated_backlog_proposal()
        self.rsi.auto_adopt(pid)
        # A second improvement, evaluated on a fresh backlog, hits the cap.
        self.core.consolidation.set_thresholds(auto_threshold=0.9)
        self.seed_backlog(n=3, confidence=0.87)
        second = [p for p in self.rsi.propose()
                  if p.target == "consolidation_backlog"]
        # The first proposal is auto_adopted, not pending, so a new one
        # is proposed; evaluate and try to adopt it.
        self.assertEqual(len(second), 1)
        self.rsi.evaluate(second[0].id)
        with self.assertRaises(ValueError):
            self.rsi.auto_adopt(second[0].id)
        self.assertEqual(self.core.consolidation.auto_threshold, 0.9)

    def test_auto_adopt_blocked_while_paused(self):
        for i in range(3):
            pid = self.rsi.submit("code", f"target-{i}", "owner review").id
            self.rsi.decide(pid, "rejected")
        pid = self.evaluated_backlog_proposal()
        with self.assertRaises(ValueError):
            self.rsi.auto_adopt(pid)
        self.assertEqual(self.core.consolidation.auto_threshold, 0.9)

    def test_auto_adopt_twice_is_refused(self):
        pid = self.evaluated_backlog_proposal()
        self.rsi.auto_adopt(pid)
        with self.assertRaises(ValueError):
            self.rsi.auto_adopt(pid)


class TestRollback(Base):
    def test_rollback_restores_previous_value_and_persists(self):
        pid = self.evaluated_backlog_proposal()
        self.rsi.auto_adopt(pid)
        self.assertEqual(self.core.consolidation.auto_threshold, 0.85)
        rolled = self.rsi.rollback(pid)
        self.assertEqual(rolled.status, "rolled_back")
        self.assertEqual(self.core.consolidation.auto_threshold, 0.9)
        self.assertIn("RSI_ROLLBACK", self.actions())
        self.reopen()
        self.assertEqual(self.core.consolidation.auto_threshold, 0.9)
        with self.assertRaises(ValueError):
            self.rsi.rollback(pid)

    def test_rollback_requires_auto_adopted_status(self):
        self.seed_backlog()
        pid = self.rsi.propose()[0].id
        with self.assertRaises(ValueError):
            self.rsi.rollback(pid)

    def test_rollback_if_regressed_healthy_core_changes_nothing(self):
        pid = self.evaluated_backlog_proposal()
        self.rsi.auto_adopt(pid)
        self.assertEqual(self.rsi.rollback_if_regressed(), [])
        self.assertEqual(self.core.consolidation.auto_threshold, 0.85)
        self.assertNotIn("RSI_ROLLBACK", self.actions())

    def test_rollback_if_regressed_on_missing_provenance(self):
        pid = self.evaluated_backlog_proposal()
        self.rsi.auto_adopt(pid)
        # The audit check fails: a memory row without provenance.
        self.core.memory.fact("Unsourced claim.", source="")
        rolled = self.rsi.rollback_if_regressed()
        self.assertEqual([p.id for p in rolled], [pid])
        self.assertEqual(rolled[0].status, "rolled_back")
        self.assertEqual(self.core.consolidation.auto_threshold, 0.9)
        self.assertIn("RSI_ROLLBACK", self.actions())
        self.assertTrue(self.core.ledger.verify())

    def test_rollback_if_regressed_with_nothing_adopted_is_noop(self):
        self.core.memory.fact("Unsourced claim.", source="")
        self.assertEqual(self.rsi.rollback_if_regressed(), [])
        self.assertNotIn("RSI_ROLLBACK", self.actions())


class TestCycleAutoAdoption(Base):
    def test_cycle_auto_adopts_backlog_proposal(self):
        self.seed_backlog()
        result = self.rsi.run_cycle()
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.auto_adopted), 1)
        self.assertEqual(self.core.consolidation.auto_threshold, 0.85)
        self.assertIn("RSI_AUTO_ADOPTED", self.actions())
        self.assertTrue(self.core.ledger.verify())

    def test_cycle_leaves_unadoptable_proposals_pending(self):
        # A prompt proposal from gate denials has no patch: the cycle
        # evaluates it as descriptive and leaves it pending for the owner.
        for _ in range(3):
            try:
                self.core.tools.run("write_file", {"path": "x.txt", "content": "x"})
            except PermissionError:
                pass
        result = self.rsi.run_cycle()
        self.assertEqual(result.auto_adopted, ())
        self.assertEqual(len(self.rsi.pending()), 1)
        self.assertEqual(self.rsi.pending()[0].kind, "prompt")

    def test_orchestrator_and_cli_surface(self):
        pid = self.evaluated_backlog_proposal()
        self.assertEqual(self.orch.rsi_auto_adopt(pid).status, "auto_adopted")
        self.assertEqual(self.orch.rsi_rollback_if_regressed(), [])
        from steelbound.cli import main
        self.assertEqual(main(["--data-dir", self.tmp.name, "rsi", "rollback-check"]), 0)
        # CLI auto-adopt on an already-adopted proposal fails cleanly.
        self.assertEqual(main(["--data-dir", self.tmp.name, "rsi", "auto-adopt", str(pid)]), 1)
        # CLI rollback restores the value (reopen to reload persisted state).
        self.assertEqual(main(["--data-dir", self.tmp.name, "rsi", "rollback", str(pid)]), 0)
        self.reopen()
        self.assertEqual(self.core.consolidation.auto_threshold, 0.9)


if __name__ == "__main__":
    unittest.main()
