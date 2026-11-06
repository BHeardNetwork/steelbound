import os, sys, tempfile, unittest
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
        return [e.action for e in self.core.ledger.entries(1000)]

    def run_actions(self):
        return [(e.action, e.payload) for e in self.core.ledger.entries(1000)
                if e.action == "SCHEDULER_RUN"]


class TestScheduler(Base):
    def test_builtin_jobs_registered_and_due_immediately(self):
        jobs = {j.name: j for j in self.sched.jobs()}
        self.assertEqual(set(jobs), {"consolidation", "audit", "rsi_cycle"})
        self.assertTrue(all(j.enabled for j in jobs.values()))
        self.assertEqual(jobs["consolidation"].interval_seconds, 6 * 3600)
        self.assertEqual(jobs["audit"].interval_seconds, 3600)
        self.assertEqual(jobs["rsi_cycle"].interval_seconds, 6 * 3600)
        self.assertTrue(all(j.last_run is None for j in jobs.values()))
        self.assertEqual({j.name for j in self.sched.due_jobs(now=1000.0)},
                         {"consolidation", "audit", "rsi_cycle"})

    def test_run_due_runs_both_and_ledgers_each(self):
        self.core.memory.episode("Scheduler fact.", source="owner", confidence=0.95)
        runs = self.sched.run_due(now=1000.0)
        self.assertEqual({r.job for r in runs}, {"consolidation", "audit", "rsi_cycle"})
        self.assertTrue(all(r.status == "ok" for r in runs))
        con = next(r for r in runs if r.job == "consolidation")
        self.assertEqual(len(con.detail["promoted"]), 1)
        audit = next(r for r in runs if r.job == "audit")
        self.assertTrue(audit.detail["valid"])
        # RSI R3: the scheduled sweep runs too; a healthy core cycles
        # as no_proposal, and its generation is ledgered in the run.
        rsi = next(r for r in runs if r.job == "rsi_cycle")
        self.assertEqual(rsi.detail["cycle_status"], "no_proposal")
        self.assertEqual(rsi.detail["generation"], 1)
        # semantic promotion actually happened, and all runs are ledgered
        self.assertIn("Scheduler fact.",
                      [i.content for i in self.core.memory.all_active(kind="semantic")])
        self.assertEqual(len(self.run_actions()), 3)
        self.assertTrue(self.core.ledger.verify())

    def test_run_due_is_not_due_twice(self):
        first = self.sched.run_due(now=1000.0)
        self.assertEqual(len(first), 3)
        second = self.sched.run_due(now=1001.0)
        self.assertEqual(second, [])
        self.assertEqual(len(self.run_actions()), 3)
        # audit becomes due after its hour, consolidation (6h) still not due
        third = self.sched.run_due(now=1000.0 + 3600)
        self.assertEqual([r.job for r in third], ["audit"])
        # at +6h all three are due again (audit's last run was at +1h,
        # due at +2h; consolidation and rsi_cycle ran at 1000, due at +6h)
        fourth = self.sched.run_due(now=1000.0 + 6 * 3600)
        self.assertEqual([r.job for r in fourth], ["audit", "consolidation", "rsi_cycle"])

    def test_run_job_forces_early_run(self):
        self.sched.run_due(now=1000.0)
        run = self.sched.run_job("audit", now=1001.0)
        self.assertEqual(run.status, "ok")
        self.assertEqual(self.sched.get_job("audit").run_count, 2)

    def test_disable_skips_due_and_refuses_forced_run(self):
        self.sched.disable("audit")
        runs = self.sched.run_due(now=1000.0)
        self.assertEqual([r.job for r in runs], ["consolidation", "rsi_cycle"])
        with self.assertRaises(ValueError):
            self.sched.run_job("audit", now=1000.0)
        self.sched.enable("audit")
        run = self.sched.run_job("audit", now=1000.0)
        self.assertEqual(run.status, "ok")
        self.assertIn("SCHEDULER_CONFIG", self.actions())

    def test_unknown_job_and_bad_interval_errors(self):
        with self.assertRaises(KeyError):
            self.sched.get_job("nope")
        with self.assertRaises(KeyError):
            self.sched.run_job("nope")
        with self.assertRaises(KeyError):
            self.sched.disable("nope")
        with self.assertRaises(ValueError):
            self.sched.set_interval("audit", 0)
        with self.assertRaises(ValueError):
            self.sched.set_interval("audit", -5)
        with self.assertRaises(ValueError):
            self.sched.register_job("audit", 60, lambda: {})
        with self.assertRaises(ValueError):
            self.sched.register_job("custom", 0, lambda: {})

    def test_set_interval_changes_due_time(self):
        self.sched.run_due(now=1000.0)
        self.sched.set_interval("audit", 10)
        self.assertEqual([r.job for r in self.sched.run_due(now=1005.0)], [])
        self.assertEqual([r.job for r in self.sched.run_due(now=1010.0)], ["audit"])

    def test_audit_fails_on_tampered_ledger_and_stays_failed(self):
        self.core.conn.execute("DROP TRIGGER ledger_no_update")
        self.core.conn.execute("DROP TRIGGER ledger_no_delete")
        self.core.conn.execute("UPDATE ledger_entries SET action='TAMPERED' WHERE id=1")
        self.core.conn.commit()
        self.assertFalse(self.core.ledger.verify())
        runs = self.sched.run_due(now=1000.0)
        audit = next(r for r in runs if r.job == "audit")
        self.assertEqual(audit.status, "failed")
        self.assertFalse(audit.detail["valid"])
        # the failure itself was ledgered, and the chain is still broken
        audit_entries = [p for _, p in self.run_actions() if p["job"] == "audit"]
        self.assertEqual(audit_entries[-1]["status"], "failed")
        self.assertFalse(self.core.ledger.verify())

    def test_audit_fails_on_missing_provenance(self):
        self.core.memory.fact("Unsourced claim.", source="")
        run = self.sched.run_job("audit", now=1000.0)
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.detail["missing_provenance"], 1)

    def test_job_failure_isolated_other_jobs_still_run(self):
        self.sched.register_job("boom", 60, lambda: 1 / 0)
        runs = self.sched.run_due(now=1000.0)
        by_job = {r.job: r for r in runs}
        self.assertEqual(by_job["boom"].status, "failed")
        self.assertIn("ZeroDivisionError", by_job["boom"].detail["error"])
        self.assertEqual(by_job["audit"].status, "ok")
        self.assertEqual(by_job["consolidation"].status, "ok")
        # run_due itself did not raise, and the failure was ledgered
        self.assertTrue(any(p["job"] == "boom" and p["status"] == "failed"
                            for _, p in self.run_actions()))

    def test_state_persists_across_reopen(self):
        self.sched.run_due(now=1000.0)
        self.sched.disable("consolidation")
        self.core.close()
        core2 = Core(Settings(data_dir=self.tmp.name))
        try:
            jobs = {j.name: j for j in core2.scheduler.jobs()}
            self.assertEqual(jobs["audit"].run_count, 1)
            self.assertEqual(jobs["audit"].last_run, 1000.0)
            self.assertFalse(jobs["consolidation"].enabled)
            self.assertEqual(core2.scheduler.run_due(now=1001.0), [])
        finally:
            core2.close()

    def test_orchestrator_and_cli_surface(self):
        runs = self.orch.schedule_run_due(now=1000.0)
        self.assertEqual(len(runs), 3)
        self.assertEqual(len(self.orch.schedule_status()), 3)
        from steelbound.cli import main
        self.assertEqual(main(["--data-dir", self.tmp.name, "schedule"]), 0)
        self.assertEqual(main(["--data-dir", self.tmp.name, "schedule", "run"]), 0)
        self.assertEqual(main(["--data-dir", self.tmp.name, "schedule", "run", "audit"]), 0)
        self.assertEqual(main(["--data-dir", self.tmp.name, "schedule", "disable", "audit"]), 0)
        self.assertEqual(main(["--data-dir", self.tmp.name, "schedule", "enable", "audit"]), 0)
        self.assertEqual(main(["--data-dir", self.tmp.name, "schedule", "interval", "audit", "30"]), 0)
        self.assertEqual(main(["--data-dir", self.tmp.name, "schedule", "run", "nope"]), 1)
        self.assertEqual(main(["--data-dir", self.tmp.name, "schedule", "interval", "audit", "0"]), 1)
        self.assertTrue(self.core.ledger.verify())


if __name__ == "__main__":
    unittest.main()
