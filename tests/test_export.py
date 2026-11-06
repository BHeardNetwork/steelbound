import copy, io, json, os, sys, tempfile, unittest
from contextlib import redirect_stdout
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator
from steelbound.exporter import export_bundle, import_bundle, verify_bundle
from steelbound.cli import main as cli_main


def make_source(tmp):
    core = Core(Settings(data_dir=tmp))
    orch = Orchestrator(core)
    orch.remember_fact("Steelbound exports memory and ledger.", source="owner")
    old = orch.remember_fact("Language is French.", source="owner")
    core.memory.correct(old, "Language is English.", source="owner")
    core.memory.episode("Visited the library on Tuesday.", source="journal", confidence=0.7)
    orch.consolidate()  # queues a proposal for the 0.7 episodic item
    return core, orch


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core, self.orch = make_source(os.path.join(self.tmp.name, "src"))
    def tearDown(self):
        self.core.close(); self.tmp.cleanup()
    def fresh_core(self, name="dst"):
        d = os.path.join(self.tmp.name, name)
        core = Core(Settings(data_dir=d))
        self.addCleanup(core.close)
        return core


class TestExportImport(Base):
    def test_round_trip_preserves_content_and_history(self):
        bundle = export_bundle(self.core)
        self.assertEqual(bundle["format"], "steelbound-bundle")
        self.assertEqual(bundle["version"], 1)
        dst = self.fresh_core()
        summary = import_bundle(dst, bundle)
        self.assertTrue(dst.ledger.verify())
        self.assertEqual(summary["memory_imported"], len(bundle["memory"]))
        contents = [i.content for i in dst.memory.all_active()]
        self.assertIn("Steelbound exports memory and ledger.", contents)
        self.assertIn("Language is English.", contents)
        self.assertNotIn("Language is French.", contents)
        # Superseded history survived: the old item exists, superseded.
        all_rows = dst.conn.execute(
            "SELECT content, superseded_by FROM memory_items").fetchall()
        old_rows = [r for r in all_rows if r[0] == "Language is French."]
        self.assertEqual(len(old_rows), 1)
        self.assertIsNotNone(old_rows[0][1])
        new_id = old_rows[0][1]
        new_content = dst.conn.execute(
            "SELECT content FROM memory_items WHERE id=?", (new_id,)).fetchone()[0]
        self.assertEqual(new_content, "Language is English.")
        # Source ledger actions appear in order inside the target chain.
        src_actions = [e["action"] for e in bundle["ledger"]]
        dst_actions = [e.action for e in dst.ledger.entries(1000)]
        it = iter(dst_actions)
        self.assertTrue(all(any(a == want for a in it) for want in src_actions))

    def test_pending_proposal_survives_round_trip(self):
        bundle = export_bundle(self.core)
        dst = self.fresh_core()
        import_bundle(dst, bundle)
        pending = dst.consolidation.pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].content, "Visited the library on Tuesday.")
        sem = dst.consolidation.approve(pending[0].id)
        self.assertIn("Visited the library on Tuesday.",
                      [i.content for i in dst.memory.all_active()])

    def test_import_into_nonempty_core_keeps_chain_valid(self):
        bundle = export_bundle(self.core)
        dst = self.fresh_core()
        dst.memory.fact("Pre-existing fact.", source="owner")
        before = dst.ledger.count()
        import_bundle(dst, bundle)
        self.assertTrue(dst.ledger.verify())
        self.assertGreater(dst.ledger.count(), before)
        self.assertIn("Pre-existing fact.",
                      [i.content for i in dst.memory.all_active()])

    def test_tampered_ledger_entry_rejected_and_nothing_written(self):
        bundle = export_bundle(self.core)
        bundle["ledger"][0]["payload"] = {"action": "forged"}
        dst = self.fresh_core()
        before_ledger = dst.ledger.count()
        before_mem = len(dst.memory.all_active())
        with self.assertRaises(ValueError):
            import_bundle(dst, bundle)
        self.assertEqual(dst.ledger.count(), before_ledger)
        self.assertEqual(len(dst.memory.all_active()), before_mem)

    def test_broken_chain_link_rejected(self):
        bundle = export_bundle(self.core)
        if len(bundle["ledger"]) > 1:
            bundle["ledger"][1]["prev_hash"] = "0" * 64
        with self.assertRaises(ValueError):
            verify_bundle(bundle)

    def test_tampered_memory_rejected_and_nothing_written(self):
        bundle = export_bundle(self.core)
        bundle["memory"][0]["content"] = "Tampered content."
        dst = self.fresh_core()
        before = dst.ledger.count()
        with self.assertRaises(ValueError):
            import_bundle(dst, bundle)
        self.assertEqual(dst.ledger.count(), before)

    def test_bad_format_and_version_rejected(self):
        bundle = export_bundle(self.core)
        bad = copy.deepcopy(bundle); bad["format"] = "something-else"
        with self.assertRaises(ValueError): verify_bundle(bad)
        bad = copy.deepcopy(bundle); bad["version"] = 99
        with self.assertRaises(ValueError): verify_bundle(bad)
        bad = copy.deepcopy(bundle); del bad["ledger"]
        with self.assertRaises(ValueError): verify_bundle(bad)

    def test_empty_bundle_imports(self):
        dst = self.fresh_core()
        import_bundle(dst, {"format": "steelbound-bundle", "version": 1,
                            "exported_at": 0.0, "ledger": [], "memory": [],
                            "proposals": [],
                            "integrity": {"memory_sha256": __import__(
                                "hashlib").sha256(json.dumps(
                                    {"memory": [], "proposals": []},
                                    sort_keys=True, separators=(",", ":")).encode()
                                ).hexdigest()}})
        self.assertTrue(dst.ledger.verify())


class TestExportImportCLI(Base):
    def test_cli_export_then_import(self):
        bundle_path = os.path.join(self.tmp.name, "bundle.json")
        rc = cli_main(["--data-dir", os.path.join(self.tmp.name, "src"),
                       "export", "--out", bundle_path])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(bundle_path))
        rc = cli_main(["--data-dir", os.path.join(self.tmp.name, "cli-dst"),
                       "import", bundle_path])
        self.assertEqual(rc, 0)

    def test_cli_import_of_tampered_bundle_exits_nonzero(self):
        bundle = export_bundle(self.core)
        bundle["ledger"][0]["action"] = "FORGED"
        bad_path = os.path.join(self.tmp.name, "bad.json")
        with open(bad_path, "w") as f: json.dump(bundle, f)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = cli_main(["--data-dir", os.path.join(self.tmp.name, "cli-dst2"),
                           "import", bad_path])
        self.assertNotEqual(rc, 0)

    def test_cli_export_to_stdout_is_valid_bundle(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = cli_main(["--data-dir", os.path.join(self.tmp.name, "src"), "export"])
        self.assertEqual(rc, 0)
        bundle = json.loads(out.getvalue())
        verify_bundle(bundle)


if __name__ == "__main__": unittest.main()
