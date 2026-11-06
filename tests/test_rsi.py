import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator
from steelbound.rsi import Observation, validate_patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_SANDBOX = bool(os.environ.get("STEELBOUND_RSI_SANDBOX"))


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


class TestObservePropose(Base):
    def test_observe_healthy_core_has_no_failing_suite(self):
        o = self.rsi.observe()
        self.assertEqual(o.failing_suites, ())
        self.assertTrue(o.all_evals_pass)
        self.assertEqual(o.eval_scores["memory"], {"passed": 2, "total": 2})
        self.assertEqual(o.backlog, 0)
        self.assertTrue(self.core.ledger.verify())

    def test_propose_on_healthy_core_makes_nothing(self):
        self.assertEqual(self.rsi.propose(), [])
        self.assertEqual(self.rsi.pending(), [])
        self.assertNotIn("RSI_PROPOSAL", self.actions())

    def test_backlog_proposal_and_idempotency(self):
        self.seed_backlog()
        made = self.rsi.propose()
        self.assertEqual(len(made), 1)
        p = made[0]
        self.assertEqual(p.kind, "config")
        self.assertEqual(p.target, "consolidation_backlog")
        self.assertEqual(p.patch, {"kind": "config",
                                   "setting": "consolidation_auto_threshold",
                                   "value": 0.85})
        self.assertIn("RSI_PROPOSAL", self.actions())
        # second run proposes nothing new for the same pending target
        self.assertEqual(self.rsi.propose(), [])
        self.assertEqual(len(self.rsi.pending()), 1)
        self.assertTrue(self.core.ledger.verify())

    def test_backlog_far_below_threshold_proposes_nothing(self):
        # 0.7 items would not be promoted by a 0.05 threshold step, so
        # proposing that step would be busywork, not an improvement.
        self.seed_backlog(confidence=0.7)
        self.assertEqual(self.rsi.propose(), [])

    def test_failing_suite_proposal_from_observation(self):
        obs = Observation(
            eval_scores={"gate_compliance": {"passed": 5, "total": 6}},
            failing_suites=("gate_compliance",), weakest_suite="gate_compliance",
            ledger_counts={}, gate_denials=0, failed_scheduler_runs=0,
            episodic_active=0, semantic_active=0, pending_proposals=0, backlog=0)
        made = self.rsi.propose(obs)
        self.assertEqual(len(made), 1)
        self.assertEqual(made[0].kind, "code")
        self.assertEqual(made[0].target, "eval:gate_compliance")
        self.assertIsNone(made[0].patch)

    def test_gate_denials_proposal_never_lowers_risk(self):
        for _ in range(3):
            try:
                self.core.tools.run("write_file", {"path": "x.txt", "content": "x"})
            except PermissionError:
                pass
        made = self.rsi.propose()
        self.assertEqual(len(made), 1)
        self.assertEqual(made[0].kind, "prompt")
        self.assertIn("never lower", made[0].rationale)


class TestSandboxEvaluation(Base):
    def test_config_patch_measures_improvement(self):
        self.seed_backlog()
        pid = self.rsi.propose()[0].id
        result = self.rsi.evaluate(pid)
        self.assertEqual(result.verdict, "improvement")
        self.assertEqual((result.targeted_before, result.targeted_after), (0.0, 3.0))
        self.assertEqual(result.baseline_evals, result.sandbox_evals)
        self.assertIn("RSI_EVALUATED", self.actions())
        # live core untouched: the seeded items are still episodic there
        self.assertEqual(len(self.core.memory.all_active(kind="episodic")), 3)
        self.assertTrue(self.core.ledger.verify())

    def test_descriptive_proposal_is_not_scored_as_improvement(self):
        pid = self.rsi.submit("code", "eval:gate_compliance",
                              "Fix the measured gate failure.").id
        result = self.rsi.evaluate(pid)
        self.assertEqual(result.verdict, "descriptive")
        self.assertIsNone(result.targeted_before)
        self.assertEqual(self.rsi.pending()[0].score["verdict"], "descriptive")

    def test_forbidden_patches_rejected_before_any_copy(self):
        for bad in (
            {"kind": "replace", "file": "src/steelbound/ledger.py",
             "old": "x", "new": "y"},
            {"kind": "replace", "file": "src/steelbound/gate.py",
             "old": "x", "new": "y"},
            {"kind": "replace", "file": "src/steelbound/evals.py",
             "old": "x", "new": "y"},
            {"kind": "replace", "file": "tests/test_rsi.py",
             "old": "x", "new": "y"},
            {"kind": "config", "setting": "unknown_setting", "value": 0.5},
            {"kind": "config", "setting": "consolidation_auto_threshold",
             "value": 1.5},
        ):
            with self.assertRaises(ValueError):
                self.rsi.submit("code", "bad", "forbidden", bad)
        self.assertEqual(self.rsi.pending(), [])

    @unittest.skipIf(IN_SANDBOX, "sandbox runs must not spawn a nested sandbox")
    def test_replace_patch_no_improvement_and_live_repo_untouched(self):
        with open(os.path.join(REPO_ROOT, "src/steelbound/tools.py"),
                  encoding="utf-8") as f:
            before = f.read()
        pid = self.rsi.submit("code", "tools-listing",
                              "Annotate list_files; no behaviour change.",
                              {"kind": "replace",
                               "file": "src/steelbound/tools.py",
                               "old": "def list_files() -> list[str]:",
                               "new": "def list_files() -> list[str]:  # listed"}).id
        result = self.rsi.evaluate(pid, repo_root=REPO_ROOT)
        self.assertEqual(result.verdict, "no_improvement")
        self.assertTrue(result.sandbox_tests["ok"])
        self.assertGreater(result.sandbox_tests["ran"], 0)
        with open(os.path.join(REPO_ROOT, "src/steelbound/tools.py"),
                  encoding="utf-8") as f:
            after = f.read()
        self.assertEqual(before, after)

    @unittest.skipIf(IN_SANDBOX, "sandbox runs must not spawn a nested sandbox")
    def test_replace_patch_regression_detected(self):
        pid = self.rsi.submit("code", "eval:honesty",
                              "Break the honesty reply on purpose.",
                              {"kind": "replace",
                               "file": "src/steelbound/providers.py",
                               "old": 'return "I don\'t know, and I won\'t fabricate an answer. Here is what I can verify from memory, if anything."',
                               "new": 'return "Invented source: a paper I just made up."'}).id
        result = self.rsi.evaluate(pid, repo_root=REPO_ROOT)
        self.assertEqual(result.verdict, "regression")
        self.assertLess(result.sandbox_evals["honesty"]["passed"],
                        result.baseline_evals["honesty"]["passed"])

    @unittest.skipIf(IN_SANDBOX, "sandbox runs must not spawn a nested sandbox")
    def test_replace_patch_ambiguous_old_string_is_error(self):
        pid = self.rsi.submit("code", "ambiguous",
                              "Old string occurs many times.",
                              {"kind": "replace",
                               "file": "src/steelbound/tools.py",
                               "old": "return", "new": "return"}).id
        result = self.rsi.evaluate(pid, repo_root=REPO_ROOT)
        self.assertEqual(result.verdict, "error")


class TestOwnerGateAndCycle(Base):
    def test_decide_approves_without_applying_patch(self):
        self.seed_backlog()
        pid = self.rsi.propose()[0].id
        decided = self.rsi.decide(pid, "approved")
        self.assertEqual(decided.status, "approved")
        self.assertIn("RSI_ADOPTED", self.actions())
        # approval alone changes nothing: threshold still 0.9, items still episodic
        self.assertEqual(self.core.consolidation.auto_threshold, 0.9)
        self.assertEqual(len(self.core.memory.all_active(kind="episodic")), 3)
        with self.assertRaises(ValueError):
            self.rsi.decide(pid, "rejected")
        with self.assertRaises(KeyError):
            self.rsi.decide(9999, "approved")
        with self.assertRaises(ValueError):
            self.rsi.decide(pid, "maybe")

    def test_cycle_on_healthy_core_reports_no_proposal(self):
        result = self.rsi.run_cycle()
        self.assertEqual(result.status, "no_proposal")
        self.assertEqual(result.proposals, ())
        self.assertIn("RSI_CYCLE", self.actions())

    def test_cycle_cap_enforced(self):
        engine = self.core.rsi
        engine.max_cycles_per_day = 1
        self.assertEqual(engine.run_cycle().status, "no_proposal")
        capped = engine.run_cycle()
        self.assertEqual(capped.status, "capped")

    def test_pause_after_three_consecutive_rejections(self):
        for i in range(3):
            pid = self.rsi.submit("code", f"target-{i}", "owner review").id
            self.rsi.decide(pid, "rejected")
        result = self.rsi.run_cycle()
        self.assertEqual(result.status, "paused")
        self.assertEqual(result.proposals, ())

    def test_orchestrator_and_cli_surface(self):
        self.seed_backlog()
        made = self.orch.rsi_propose()
        self.assertEqual(len(made), 1)
        result = self.orch.rsi_evaluate(made[0].id)
        self.assertEqual(result.verdict, "improvement")
        self.assertEqual(self.orch.rsi_decide(made[0].id, "rejected").status, "rejected")
        from steelbound.cli import main
        self.assertEqual(main(["--data-dir", self.tmp.name, "rsi", "observe"]), 0)
        self.assertEqual(main(["--data-dir", self.tmp.name, "rsi", "pending"]), 0)


if __name__ == "__main__":
    unittest.main()
