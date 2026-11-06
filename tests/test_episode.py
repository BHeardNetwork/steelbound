import os, sys, tempfile, time, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator
from steelbound.episode import _today, validate_day
from steelbound.cli import main as cli_main

class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = Core(Settings(data_dir=self.tmp.name))
        self.orch = Orchestrator(self.core)
        self.ep = self.core.episode
    def tearDown(self):
        self.core.close(); self.tmp.cleanup()

class TestCompose(Base):
    def test_compose_today_document_memory_ledger(self):
        made = self.ep.compose()
        self.assertFalse(made.already_existed)
        self.assertEqual(made.day, _today())
        self.assertTrue(os.path.isfile(made.document_path))
        self.assertTrue(made.document_path.endswith(
            os.path.join("workspace", "episodes", f"{_today()}.md")))
        with open(made.document_path, encoding="utf-8") as f:
            doc = f.read()
        self.assertIn(made.text, doc)
        self.assertIn("## Trace", doc)
        # First person, from records: the day's CORE_INIT is counted.
        self.assertIn("I recorded", made.text)
        self.assertIn("Tomorrow", made.text)  # plan section, today only
        self.assertIn("consolidation", made.text)
        item = [i for i in self.core.memory.all_active(kind="episodic")
                if i.id == made.memory_id]
        self.assertEqual(len(item), 1)
        self.assertEqual(item[0].source, "daily-episode")
        self.assertIn("daily-episode", item[0].tags)
        entry = [e for e in self.core.ledger.entries(10000)
                 if e.action == "DAILY_EPISODE"]
        self.assertEqual(len(entry), 1)
        self.assertEqual(entry[0].id, made.ledger_entry_id)
        self.assertEqual(entry[0].payload["day"], _today())
        self.assertEqual(entry[0].payload["memory_id"], made.memory_id)
        self.assertTrue(self.core.ledger.verify())

    def test_compose_twice_is_idempotent(self):
        first = self.ep.compose()
        second = self.ep.compose()
        self.assertTrue(second.already_existed)
        self.assertEqual(second.memory_id, first.memory_id)
        self.assertEqual(second.ledger_entry_id, first.ledger_entry_id)
        self.assertEqual(second.text, first.text)
        self.assertEqual(len([e for e in self.core.ledger.entries(10000)
                              if e.action == "DAILY_EPISODE"]), 1)
        self.assertEqual(len([i for i in self.core.memory.all_active(kind="episodic")
                              if i.source == "daily-episode"]), 1)

    def test_activity_reflected(self):
        self.orch.remember_fact("The owner's preferred language is English.")
        self.orch.chat("What is the project, based on memory?")
        with self.assertRaises(PermissionError):
            self.orch.run_tool("write_file", {"path": "note.txt",
                                              "content": "hi"})
        self.core.cognition.request_approval("search the web", "external",
                                             approved=True)
        made = self.ep.compose()
        self.assertIn("English", made.text)
        self.assertIn("chat turn", made.text)
        self.assertIn("refused", made.text)
        self.assertIn("write_file", made.text)
        self.assertIn("no deliberation record", made.text)
        self.assertGreaterEqual(made.counts["refusals"], 2)
        # Traceability: the learned trace lists ledger entries only
        # (the fact's FACT_STORED entry), never memory item ids, and
        # the memory items are listed separately under learned_memory.
        entry = [e for e in self.core.ledger.entries(10000)
                 if e.action == "DAILY_EPISODE"][0]
        by_id = {e.id: e for e in self.core.ledger.entries(10000)}
        for lid in entry.payload["trace"]["learned"]:
            self.assertEqual(by_id[lid].action, "FACT_STORED")
        self.assertIn(1, entry.payload["trace"]["learned_memory"])

    def test_rsi_proposal_reflected(self):
        self.core.rsi.submit("prompt", "offline reply style",
                             "owner asked for shorter replies")
        made = self.ep.compose()
        self.assertIn("self-improvement", made.text)
        self.assertIn("offline reply style", made.text)

    def test_quiet_past_day_is_short_and_planless(self):
        made = self.ep.compose("2020-01-02")
        self.assertEqual(made.text,
                         "I have no record of activity on 2020-01-02.")
        self.assertNotIn("Tomorrow", made.text)
        self.assertTrue(os.path.isfile(made.document_path))
        again = self.ep.compose("2020-01-02")
        self.assertTrue(again.already_existed)

    def test_invalid_and_future_days_rejected(self):
        with self.assertRaises(ValueError):
            self.ep.compose("not-a-date")
        with self.assertRaises(ValueError):
            self.ep.compose("2999-01-01")
        with self.assertRaises(ValueError):
            validate_day("2026-13-40")

    def test_scheduled_episode_body(self):
        detail = self.ep.scheduled_episode()
        self.assertEqual(detail["day"], _today())
        self.assertFalse(detail["already_existed"])
        self.assertIsNotNone(detail["memory_id"])
        again = self.ep.scheduled_episode()
        self.assertTrue(again["already_existed"])
        self.assertEqual(again["memory_id"], detail["memory_id"])

    def test_episode_never_promoted_by_consolidation(self):
        made = self.ep.compose()
        result = self.core.consolidation.run()
        self.assertIn(made.memory_id, result.skipped)
        active = [i.id for i in
                  self.core.memory.all_active(kind="episodic")]
        self.assertIn(made.memory_id, active)
        pending_ids = [p.episodic_id for p in self.core.consolidation.pending()]
        self.assertNotIn(made.memory_id, pending_ids)

    def test_orchestrator_and_listing(self):
        made = self.orch.episode_compose()
        self.assertEqual(self.orch.episode_show().text, made.text)
        self.assertIsNone(self.orch.episode_show("2020-01-03"))
        records = self.orch.episode_list()
        self.assertEqual([r.day for r in records], [_today()])
        self.assertEqual(records[0].memory_id, made.memory_id)

class TestCLI(Base):
    def test_cli_surface(self):
        self.assertEqual(cli_main(["--data-dir", self.tmp.name,
                                   "episode", "compose"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name,
                                   "episode", "compose"]), 0)  # idempotent
        self.assertEqual(cli_main(["--data-dir", self.tmp.name,
                                   "episode", "show"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name,
                                   "episode", "list"]), 0)
        # Unknown day: show exits 1; bad day: compose exits 1.
        self.assertEqual(cli_main(["--data-dir", self.tmp.name,
                                   "episode", "show", "--day", "2020-01-03"]), 1)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name,
                                   "episode", "compose", "--day", "2999-01-01"]), 1)
        self.assertTrue(self.core.ledger.verify())

if __name__ == "__main__":
    unittest.main()
