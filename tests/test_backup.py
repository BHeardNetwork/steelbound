import json, os, shutil, sqlite3, sys, tempfile, unittest, zipfile
from contextlib import redirect_stdout
from io import StringIO
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator
from steelbound.backup import (create_backup, restore_backup, verify_backup)
from steelbound.cli import main as cli_main


def make_source(tmp):
    core = Core(Settings(data_dir=tmp))
    orch = Orchestrator(core)
    orch.remember_fact("Steelbound backup keeps history.", source="owner")
    orch.remember_fact("The capital of France is Paris.", source="owner")
    # A workspace file must survive the round trip too.
    with open(os.path.join(tmp, "workspace", "note.txt"), "w") as f:
        f.write("workspace note\n")
    return core, orch


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core, self.orch = make_source(os.path.join(self.tmp.name, "src"))
    def tearDown(self):
        self.core.close(); self.tmp.cleanup()
    def backup_path(self, name="backup.zip"):
        return os.path.join(self.tmp.name, name)
    def fresh_core(self, name):
        d = os.path.join(self.tmp.name, name)
        core = Core(Settings(data_dir=d))
        self.addCleanup(core.close)
        return core


class TestBackupRestore(Base):
    def test_restore_drill_round_trip(self):
        """The restore drill: backup, wipe, restore, verify chain + recall."""
        out = self.backup_path()
        summary = create_backup(os.path.join(self.tmp.name, "src"), out,
                                conn=self.core.conn)
        self.assertTrue(os.path.isfile(out))
        self.assertGreater(summary["ledger_entries"], 0)
        self.assertEqual(summary["files"], 1)
        with zipfile.ZipFile(out) as zf:
            self.assertIn("steelbound.db", zf.namelist())
            self.assertIn("manifest.json", zf.namelist())
            self.assertIn("workspace/note.txt", zf.namelist())
        # Wipe the source completely, then restore into a fresh dir.
        self.core.close()
        shutil.rmtree(os.path.join(self.tmp.name, "src"))
        target = os.path.join(self.tmp.name, "restored")
        result = restore_backup(out, target)
        self.assertTrue(result["chain_valid"])
        # Open the restored dir itself and check it.
        restored = Core(Settings(data_dir=target))
        self.addCleanup(restored.close)
        self.assertTrue(restored.ledger.verify())
        self.assertEqual(restored.ledger.count(), summary["ledger_entries"] + 1)
        # +1 is this Core's own CORE_INIT; the restored chain itself is exact.
        contents = [i.content for i in restored.memory.all_active()]
        self.assertIn("Steelbound backup keeps history.", contents)
        self.assertIn("The capital of France is Paris.", contents)
        with open(os.path.join(target, "workspace", "note.txt")) as f:
            self.assertEqual(f.read(), "workspace note\n")

    def test_backup_of_live_core_uses_consistent_snapshot(self):
        # Backing up while the source Core is open must not raise and
        # must produce a chain-valid backup.
        out = self.backup_path()
        create_backup(os.path.join(self.tmp.name, "src"), out,
                      conn=self.core.conn)
        self.assertEqual(verify_backup(out)["format"], "steelbound-backup")

    def test_backup_without_live_connection(self):
        out = self.backup_path()
        create_backup(os.path.join(self.tmp.name, "src"), out)
        target = os.path.join(self.tmp.name, "r2")
        restore_backup(out, target)
        core = Core(Settings(data_dir=target))
        self.addCleanup(core.close)
        self.assertTrue(core.ledger.verify())

    def test_restore_refuses_nonempty_target_without_force(self):
        out = self.backup_path()
        create_backup(os.path.join(self.tmp.name, "src"), out,
                      conn=self.core.conn)
        target = os.path.join(self.tmp.name, "busy")
        os.makedirs(target)
        with open(os.path.join(target, "keep.txt"), "w") as f:
            f.write("do not touch\n")
        with self.assertRaises(ValueError):
            restore_backup(out, target)
        self.assertEqual(open(os.path.join(target, "keep.txt")).read(),
                         "do not touch\n")
        self.assertFalse(os.path.exists(os.path.join(target, "steelbound.db")))

    def test_restore_force_replaces(self):
        out = self.backup_path()
        create_backup(os.path.join(self.tmp.name, "src"), out,
                      conn=self.core.conn)
        target = os.path.join(self.tmp.name, "force")
        old = self.fresh_core("force")
        old.memory.fact("Stale fact that must go.", source="owner")
        old.close()
        restore_backup(out, target, force=True)
        core = Core(Settings(data_dir=target))
        self.addCleanup(core.close)
        self.assertTrue(core.ledger.verify())
        contents = [i.content for i in core.memory.all_active()]
        self.assertNotIn("Stale fact that must go.", contents)
        self.assertIn("Steelbound backup keeps history.", contents)

    def test_tampered_workspace_file_rejected_with_nothing_written(self):
        out = self.backup_path()
        create_backup(os.path.join(self.tmp.name, "src"), out,
                      conn=self.core.conn)
        bad = self.backup_path("bad.zip")
        with zipfile.ZipFile(out) as src, \
                zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename == "workspace/note.txt":
                    data = b"tampered\n"
                dst.writestr(item, data)
        target = os.path.join(self.tmp.name, "tampered")
        with self.assertRaises(ValueError):
            restore_backup(bad, target)
        self.assertFalse(os.path.exists(target))

    def test_tampered_ledger_rejected_even_with_fixed_manifest_hash(self):
        """Rewriting the manifest hash is not enough: the chain must fail."""
        out = self.backup_path()
        create_backup(os.path.join(self.tmp.name, "src"), out,
                      conn=self.core.conn)
        tmpdir = tempfile.mkdtemp(dir=self.tmp.name)
        with zipfile.ZipFile(out) as zf:
            zf.extractall(tmpdir)
        db = os.path.join(tmpdir, "steelbound.db")
        conn = sqlite3.connect(db)
        # Drop the append-only triggers, then tamper a payload in place.
        conn.execute("DROP TRIGGER IF EXISTS ledger_no_update")
        conn.execute("UPDATE ledger_entries SET payload='{\"x\":1}' WHERE id=1")
        conn.commit(); conn.close()
        import hashlib
        manifest_path = os.path.join(tmpdir, "manifest.json")
        manifest = json.load(open(manifest_path))
        with open(db, "rb") as f:
            manifest["db_sha256"] = hashlib.sha256(f.read()).hexdigest()
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        bad = self.backup_path("badchain.zip")
        with zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, names in os.walk(tmpdir):
                for name in names:
                    full = os.path.join(root, name)
                    zf.write(full, os.path.relpath(full, tmpdir))
        target = os.path.join(self.tmp.name, "badchain")
        with self.assertRaises(ValueError):
            restore_backup(bad, target)
        self.assertFalse(os.path.exists(target))

    def test_bad_format_and_missing_db_rejected(self):
        out = self.backup_path()
        with zipfile.ZipFile(out, "w") as zf:
            zf.writestr("manifest.json", json.dumps(
                {"format": "something-else", "version": 1}))
        with self.assertRaises(ValueError):
            verify_backup(out)
        with self.assertRaises(FileNotFoundError):
            create_backup(os.path.join(self.tmp.name, "no-such-dir"),
                          self.backup_path("x.zip"))

    def test_cli_backup_and_restore(self):
        src = os.path.join(self.tmp.name, "src")
        out = self.backup_path("cli.zip")
        with redirect_stdout(StringIO()):
            rc = cli_main(["--data-dir", src, "backup", "--out", out])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isfile(out))
        target = os.path.join(self.tmp.name, "cli-restored")
        buf = StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["--data-dir", target, "restore", out])
        self.assertEqual(rc, 0)
        self.assertIn("chain valid: True", buf.getvalue())
        core = Core(Settings(data_dir=target))
        self.addCleanup(core.close)
        self.assertTrue(core.ledger.verify())
        self.assertIn("Steelbound backup keeps history.",
                      [i.content for i in core.memory.all_active()])

    def test_cli_restore_rejects_tampered_backup(self):
        src = os.path.join(self.tmp.name, "src")
        out = self.backup_path("cli.zip")
        with redirect_stdout(StringIO()):
            cli_main(["--data-dir", src, "backup", "--out", out])
        bad = self.backup_path("cli-bad.zip")
        with zipfile.ZipFile(out) as src_zf, \
                zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as dst:
            for item in src_zf.infolist():
                data = src_zf.read(item.filename)
                if item.filename == "steelbound.db":
                    # Flip a byte well inside the file (the last byte of a
                    # SQLite file is often already 0x00, so flipping it
                    # would be a no-op tamper).
                    data = data[:100] + bytes([data[100] ^ 0xFF]) + data[101:]
                dst.writestr(item, data)
        buf = StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["--data-dir", os.path.join(self.tmp.name, "cli-bad-target"),
                           "restore", bad])
        self.assertEqual(rc, 1)
        self.assertFalse(os.path.exists(
            os.path.join(self.tmp.name, "cli-bad-target")))


if __name__ == "__main__":
    unittest.main()
