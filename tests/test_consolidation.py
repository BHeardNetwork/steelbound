import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = Core(Settings(data_dir=self.tmp.name))
        self.orch = Orchestrator(self.core)
        self.con = self.core.consolidation

    def tearDown(self):
        self.core.close(); self.tmp.cleanup()

    def actions(self):
        return [e.action for e in self.core.ledger.entries(1000)]


class TestConsolidation(Base):
    def test_auto_promote_at_threshold(self):
        ep = self.core.memory.episode("The office opens at 9am.", source="owner", confidence=0.95)
        r = self.con.run()
        self.assertEqual(len(r.promoted), 1)
        self.assertEqual(r.queued, ())
        sem = self.core.memory.all_active(kind="semantic")
        self.assertTrue(any(i.content == "The office opens at 9am." and i.source == "owner" for i in sem))
        # episodic consumed (superseded, history preserved in the row)
        self.assertNotIn(ep, [i.id for i in self.core.memory.all_active(kind="episodic")])
        row = self.core.conn.execute(
            "SELECT superseded_by FROM memory_items WHERE id=?", (ep,)).fetchone()
        self.assertEqual(row[0], r.promoted[0])
        self.assertIn("MEMORY_PROMOTE", self.actions())
        self.assertTrue(self.core.ledger.verify())

    def test_boundary_090_promotes_089_queues(self):
        self.core.memory.episode("Boundary high.", source="owner", confidence=0.9)
        self.core.memory.episode("Boundary low.", source="owner", confidence=0.89)
        r = self.con.run()
        self.assertEqual(len(r.promoted), 1)
        self.assertEqual(len(r.queued), 1)
        self.assertEqual(len(self.con.pending()), 1)
        self.assertEqual(self.con.pending()[0].content, "Boundary low.")

    def test_queue_then_approve(self):
        ep = self.core.memory.episode("Queued fact here.", source="owner", confidence=0.7)
        r = self.con.run()
        self.assertEqual(r.promoted, ())
        self.assertEqual(len(r.queued), 1)
        # queued item is NOT yet semantic and stays active as episodic
        self.assertNotIn("Queued fact here.",
                         [i.content for i in self.core.memory.all_active(kind="semantic")])
        self.assertIn(ep, [i.id for i in self.core.memory.all_active(kind="episodic")])
        sem_id = self.con.approve(r.queued[0])
        self.assertIn("Queued fact here.",
                      [i.content for i in self.core.memory.all_active(kind="semantic")])
        self.assertEqual(self.con.pending(), [])
        self.assertEqual(self.con.proposals("promoted")[0].semantic_id, sem_id)
        self.assertIn("CONSOLIDATION_DECISION", self.actions())
        self.assertTrue(self.core.ledger.verify())

    def test_queue_then_reject_leaves_episodic_active(self):
        self.core.memory.episode("Rejected fact here.", source="owner", confidence=0.6)
        r = self.con.run()
        self.assertEqual(len(r.queued), 1)
        self.con.reject(r.queued[0])
        self.assertIn("Rejected fact here.",
                      [i.content for i in self.core.memory.all_active(kind="episodic")])
        self.assertNotIn("Rejected fact here.",
                         [i.content for i in self.core.memory.all_active(kind="semantic")])
        self.assertEqual(self.con.proposals("rejected")[0].status, "rejected")
        self.assertTrue(self.core.ledger.verify())

    def test_run_is_idempotent(self):
        self.core.memory.episode("Once only fact.", source="owner", confidence=0.95)
        self.core.memory.episode("Queued once fact.", source="owner", confidence=0.7)
        first = self.con.run()
        second = self.con.run()
        self.assertEqual(second.promoted, ())
        self.assertEqual(second.queued, ())
        self.assertEqual(len(self.core.memory.all_active(kind="semantic")), 1)
        self.assertEqual(len(self.con.pending()), 1)
        self.assertEqual(first.queued, tuple(p.id for p in self.con.pending()))

    def test_duplicate_of_existing_semantic_skipped(self):
        self.core.memory.fact("Already known fact.", source="owner")
        self.core.memory.episode("Already known fact.", source="owner", confidence=1.0)
        r = self.con.run()
        self.assertEqual(r.promoted, ())
        self.assertEqual(r.queued, ())
        self.assertEqual(len(self.core.memory.all_active(kind="semantic")), 1)

    def test_chat_transcripts_not_promoted(self):
        self.core.memory.episode("user: hello there", source="conversation", confidence=1.0)
        self.core.memory.episode("assistant: hello back", source="conversation", confidence=1.0)
        r = self.con.run()
        self.assertEqual(r.promoted, ())
        self.assertEqual(r.queued, ())

    def test_below_floor_skipped(self):
        self.core.memory.episode("Weak guess.", source="owner", confidence=0.4)
        r = self.con.run()
        self.assertEqual(r.promoted, ())
        self.assertEqual(r.queued, ())
        self.assertEqual(self.con.pending(), [])

    def test_approve_reject_errors(self):
        with self.assertRaises(KeyError):
            self.con.approve(9999)
        with self.assertRaises(KeyError):
            self.con.reject(9999)
        self.core.memory.episode("Decide me.", source="owner", confidence=0.7)
        pid = self.con.run().queued[0]
        self.con.approve(pid)
        with self.assertRaises(ValueError):
            self.con.approve(pid)
        with self.assertRaises(ValueError):
            self.con.reject(pid)

    def test_orchestrator_and_cli_surface(self):
        self.core.memory.episode("CLI fact.", source="owner", confidence=0.95)
        r = self.orch.consolidate()
        self.assertEqual(len(r.promoted), 1)
        # CLI smoke: pending lists nothing after auto-promotion, run is a no-op
        from steelbound.cli import main
        self.assertEqual(main(["--data-dir", self.tmp.name, "consolidate", "pending"]), 0)
        self.assertEqual(main(["--data-dir", self.tmp.name, "consolidate"]), 0)


if __name__ == "__main__":
    unittest.main()
