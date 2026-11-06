"""RSI R3: scheduled recursion via the in-core scheduler's rsi_cycle job.

Covers: job registration/binding, the scheduled sweep on a healthy
core, auto-adoption through the schedule, the daily cycle cap, the
pause after consecutive rejections, disable/interval configuration,
failure isolation for an unbound job, rollback-first behaviour, the
joined history view, persistence across a Core reopen, and the CLI.
"""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = Core(Settings(data_dir=self.tmp.name))
        self.orch = Orchestrator(self.core)
        self.sched = self.core.scheduler

    def tearDown(self):
        self.core.close(); self.tmp.cleanup()

    def actions(self):
        return [e.action for e in self.core.ledger.entries(10000)]

    def seed_backlog(self):
        # Three episodic items just under the 0.9 auto-promote
        # threshold: lowering it to 0.85 promotes all three, so the
        # scheduled cycle should propose, evaluate, and auto-adopt.
        for i, conf in enumerate((0.85, 0.86, 0.87)):
            self.core.memory.episode(f"Backlog fact {i}.",
                                     source="owner", confidence=conf)


class TestRsiR3(Base):
    def test_rsi_cycle_job_registered_and_bound(self):
        job = self.sched.get_job("rsi_cycle")
        self.assertTrue(job.enabled)
        self.assertEqual(job.interval_seconds, 6 * 3600)
        self.assertIsNone(job.last_run)
        self.assertIsNotNone(self.sched._rsi_cycle_func)

    def test_scheduled_cycle_healthy_core_ledgers_run_and_cycle(self):
        run = self.sched.run_job("rsi_cycle", now=1000.0)
        self.assertEqual(run.status, "ok")
        self.assertEqual(run.detail["cycle_status"], "no_proposal")
        self.assertEqual(run.detail["generation"], 1)
        self.assertEqual(run.detail["proposals"], [])
        self.assertEqual(run.detail["rolled_back"], [])
        self.assertIn("RSI_CYCLE", self.actions())
        self.assertIn("SCHEDULER_RUN", self.actions())
        self.assertTrue(self.core.ledger.verify())

    def test_scheduled_cycle_auto_adopts_backlog(self):
        self.seed_backlog()
        run = self.sched.run_job("rsi_cycle", now=1000.0)
        self.assertEqual(run.status, "ok")
        self.assertEqual(run.detail["cycle_status"], "completed")
        self.assertEqual(len(run.detail["proposals"]), 1)
        self.assertEqual(run.detail["evaluations"],
                         {str(run.detail["proposals"][0]): "improvement"})
        self.assertEqual(run.detail["auto_adopted"], run.detail["proposals"])
        proposal = self.core.rsi.proposals()[0]
        self.assertEqual(proposal.status, "auto_adopted")
        self.assertEqual(proposal.score["verdict"], "improvement")
        self.assertAlmostEqual(self.core.consolidation.auto_threshold, 0.85)
        for action in ("RSI_PROPOSAL", "RSI_EVALUATED", "RSI_AUTO_ADOPTED",
                       "RSI_CYCLE", "SCHEDULER_RUN"):
            self.assertIn(action, self.actions())
        self.assertTrue(self.core.ledger.verify())

    def test_daily_cap_blocks_fifth_scheduled_cycle(self):
        for _ in range(4):
            run = self.sched.run_job("rsi_cycle", now=1000.0)
            self.assertEqual(run.detail["cycle_status"], "no_proposal")
        capped = self.sched.run_job("rsi_cycle", now=1000.0)
        self.assertEqual(capped.status, "ok")  # the job ran; the cycle capped
        self.assertEqual(capped.detail["cycle_status"], "capped")
        self.assertEqual(capped.detail["generation"], 4)
        self.assertEqual(capped.detail["proposals"], [])
        self.assertEqual(self.actions().count("RSI_CYCLE"), 4)

    def test_pause_after_three_rejections_blocks_scheduled_cycle(self):
        for i in range(3):
            p = self.core.rsi.submit("code", f"target-{i}", "owner test")
            self.core.rsi.decide(p.id, "rejected")
        detail = self.orch.rsi_scheduled_cycle()
        self.assertEqual(detail["cycle_status"], "paused")
        self.assertEqual(detail["proposals"], [])
        self.assertEqual(detail["generation"], 0)

    def test_disable_refuses_and_interval_reconfigures(self):
        self.sched.disable("rsi_cycle")
        with self.assertRaises(ValueError):
            self.sched.run_job("rsi_cycle", now=1000.0)
        self.assertEqual(self.sched.set_interval("rsi_cycle", 60).interval_seconds, 60)
        self.sched.enable("rsi_cycle")
        run = self.sched.run_job("rsi_cycle", now=1000.0)
        self.assertEqual(run.status, "ok")

    def test_unbound_rsi_cycle_fails_in_isolation(self):
        self.sched._rsi_cycle_func = None  # simulate a Scheduler built bare
        runs = self.sched.run_due(now=1000.0)
        by_job = {r.job: r for r in runs}
        self.assertEqual(by_job["rsi_cycle"].status, "failed")
        self.assertIn("no RSI engine bound", by_job["rsi_cycle"].detail["error"])
        self.assertEqual(by_job["audit"].status, "ok")
        self.assertEqual(by_job["consolidation"].status, "ok")

    def test_scheduled_cycle_rolls_back_first_on_regression(self):
        self.seed_backlog()
        first = self.sched.run_job("rsi_cycle", now=1000.0)
        adopted_id = first.detail["auto_adopted"][0]
        self.assertEqual(self.core.rsi.proposals()[0].status, "auto_adopted")
        # Break the audit checks (a memory row without provenance);
        # the next sweep must roll the adoption back before cycling.
        self.core.memory.fact("Unsourced claim.", source="")
        second = self.sched.run_job("rsi_cycle", now=1001.0)
        self.assertEqual(second.detail["rolled_back"], [adopted_id])
        self.assertIn("RSI_ROLLBACK", self.actions())
        self.assertEqual(self.core.rsi.proposals()[0].status, "rolled_back")

    def test_history_joins_proposals_scores_and_cycles(self):
        self.seed_backlog()
        self.sched.run_job("rsi_cycle", now=1000.0)
        history = self.orch.rsi_history()
        self.assertEqual(len(history["proposals"]), 1)
        entry = history["proposals"][0]
        self.assertEqual(entry["status"], "auto_adopted")
        self.assertEqual(entry["score"]["verdict"], "improvement")
        self.assertEqual({e["action"] for e in entry["events"]},
                         {"RSI_PROPOSAL", "RSI_EVALUATED", "RSI_AUTO_ADOPTED"})
        self.assertEqual(len(history["cycles"]), 1)
        self.assertEqual(history["cycles"][0]["payload"]["status"], "completed")
        # CLI surface: history prints the proposal and the cycle.
        from steelbound.cli import main
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--data-dir", self.tmp.name, "rsi", "history"]), 0)
        self.assertIn("auto_adopted", out.getvalue())
        self.assertIn("cycle: completed", out.getvalue())

    def test_cli_schedule_run_rsi_cycle(self):
        from steelbound.cli import main
        self.assertEqual(main(["--data-dir", self.tmp.name,
                               "schedule", "run", "rsi_cycle"]), 0)
        self.assertEqual(main(["--data-dir", self.tmp.name,
                               "schedule", "status"]), 0)
        self.assertEqual(self.sched.get_job("rsi_cycle").run_count, 1)

    def test_rsi_cycle_persists_and_rebinds_across_reopen(self):
        self.sched.run_job("rsi_cycle", now=1000.0)
        self.core.close()
        core2 = Core(Settings(data_dir=self.tmp.name))
        try:
            job = core2.scheduler.get_job("rsi_cycle")
            self.assertEqual(job.run_count, 1)
            self.assertEqual(job.last_run, 1000.0)
            # Audit and consolidation never ran (only rsi_cycle was
            # forced), so they are still due — but rsi_cycle is not.
            due = core2.scheduler.run_due(now=1001.0)
            self.assertEqual([r.job for r in due], ["audit", "consolidation"])
            # Re-binding happened on construction: a forced run works
            # and is generation 2 in the ledgered history.
            run = core2.scheduler.run_job("rsi_cycle", now=1001.0)
            self.assertEqual(run.status, "ok")
            self.assertEqual(run.detail["generation"], 2)
        finally:
            core2.close()


if __name__ == "__main__":
    unittest.main()
