"""Ecosystem registry: member manifests, the permission ceiling,
gated registration and retirement, the act guard, and persistence.

Covers: register/list/inspect round trip, persistence across a Core
reopen, permission subset enforcement (FORBIDDEN refused outright;
beyond-ceiling permissions rejected — never silently clamped — both
against the default ceiling and a narrower engine-level ceiling),
gate denial without approval leaving no member, duplicate names,
retirement (ledgered; a retired member cannot act), unknown members
failing closed, manifest validation, and the CLI surface.
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
from steelbound.ecosystem import (
    DEFAULT_CEILING, EcosystemEngine, MemberManifest, validate_manifest)


def manifest(name="console-app", **overrides):
    base = {
        "name": name,
        "kind": "app",
        "capabilities": ["chat", "recall"],
        "permissions": ["read", "write_local"],
        "description": "The core console.",
        "metadata": {"version": "1.0"},
    }
    base.update(overrides)
    return base


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = Core(Settings(data_dir=self.tmp.name))
        self.orch = Orchestrator(self.core)
        self.eco = self.core.ecosystem

    def tearDown(self):
        self.core.close(); self.tmp.cleanup()

    def actions(self):
        return [e.action for e in self.core.ledger.entries(10000)]


class TestRegister(Base):
    def test_register_list_inspect_round_trip(self):
        member = self.eco.register(manifest(), approved=True)
        self.assertEqual(member.name, "console-app")
        self.assertEqual(member.kind, "app")
        self.assertEqual(member.capabilities, ("chat", "recall"))
        self.assertEqual(member.permissions, (Risk.READ, Risk.WRITE_LOCAL))
        self.assertEqual(member.description, "The core console.")
        self.assertEqual(member.metadata, {"version": "1.0"})
        self.assertEqual(member.status, "active")
        self.assertIsNone(member.retired_ts)
        self.assertIn("ECOSYSTEM_REGISTER", self.actions())
        # The ledger payload carries the full manifest.
        entry = [e for e in self.core.ledger.entries(10000)
                 if e.action == "ECOSYSTEM_REGISTER"][0]
        self.assertEqual(entry.payload["name"], "console-app")
        self.assertEqual(entry.payload["manifest"]["kind"], "app")
        self.assertEqual(entry.payload["manifest"]["permissions"],
                         ["read", "write_local"])
        self.assertTrue(self.core.ledger.verify())
        # List and inspect see the same member.
        self.assertEqual([m.name for m in self.eco.list_members()],
                         ["console-app"])
        self.assertEqual(self.eco.inspect("console-app"), member)
        self.assertEqual(self.orch.ecosystem_inspect("console-app"), member)
        self.assertEqual([m.name for m in self.orch.ecosystem_list()],
                         ["console-app"])

    def test_register_accepts_manifest_object_and_risk_enums(self):
        m = MemberManifest(name="pack", kind="toolpack",
                           capabilities=("summarise",),
                           permissions=(Risk.READ,),
                           description="A toolpack.", metadata={})
        member = self.eco.register(m, approved=True)
        self.assertEqual(member.permissions, (Risk.READ,))
        self.assertEqual(member.capabilities, ("summarise",))

    def test_all_member_kinds(self):
        for i, kind in enumerate(("app", "offspring", "toolpack", "service")):
            member = self.eco.register(
                manifest(name=f"member-{i}", kind=kind), approved=True)
            self.assertEqual(member.kind, kind)
        self.assertEqual(len(self.eco.list_members()), 4)

    def test_persistence_across_reopen(self):
        self.eco.register(manifest(), approved=True)
        self.eco.register(manifest(name="helper", kind="service",
                                   permissions=["read"]), approved=True)
        self.eco.retire("helper", reason="superseded", approved=True)
        self.core.close()
        self.core = Core(Settings(data_dir=self.tmp.name))
        eco = self.core.ecosystem
        names = [m.name for m in eco.list_members()]
        self.assertEqual(names, ["console-app", "helper"])
        self.assertEqual(eco.inspect("console-app").status, "active")
        retired = eco.inspect("helper")
        self.assertEqual(retired.status, "retired")
        self.assertEqual(retired.retire_reason, "superseded")
        self.assertIsNotNone(retired.retired_ts)
        self.assertTrue(eco.can_act("console-app"))
        self.assertFalse(eco.can_act("helper"))
        self.assertTrue(self.core.ledger.verify())

    def test_duplicate_name_refused(self):
        self.eco.register(manifest(), approved=True)
        with self.assertRaises(ValueError):
            self.eco.register(manifest(), approved=True)
        # A retired member's name is not reusable either.
        self.eco.retire("console-app", approved=True)
        with self.assertRaises(ValueError):
            self.eco.register(manifest(), approved=True)
        self.assertEqual(len(self.eco.list_members()), 1)
        self.assertEqual(self.actions().count("ECOSYSTEM_REGISTER"), 1)

    def test_manifest_validation(self):
        with self.assertRaises(ValueError):
            self.eco.register(manifest(name=""), approved=True)
        with self.assertRaises(ValueError):
            self.eco.register(manifest(kind="widget"), approved=True)
        with self.assertRaises(ValueError):
            self.eco.register(manifest(capabilities=["ok", 3]), approved=True)
        with self.assertRaises(ValueError):
            self.eco.register(manifest(permissions=["fly"]), approved=True)
        with self.assertRaises(ValueError):
            self.eco.register(manifest(metadata=["not", "a", "dict"]),
                              approved=True)
        with self.assertRaises(ValueError):
            self.eco.register("not a manifest", approved=True)
        self.assertEqual(self.eco.list_members(), [])
        self.assertNotIn("ECOSYSTEM_REGISTER", self.actions())

    def test_list_status_filter(self):
        self.eco.register(manifest(), approved=True)
        self.eco.register(manifest(name="helper", kind="service"),
                          approved=True)
        self.eco.retire("helper", approved=True)
        self.assertEqual([m.name for m in self.eco.list_members("active")],
                         ["console-app"])
        self.assertEqual([m.name for m in self.eco.list_members("retired")],
                         ["helper"])
        with self.assertRaises(ValueError):
            self.eco.list_members("bogus")


class TestPermissionCeiling(Base):
    def test_default_ceiling_excludes_forbidden(self):
        self.assertNotIn(Risk.FORBIDDEN, DEFAULT_CEILING)
        self.assertEqual(self.eco.ceiling, DEFAULT_CEILING)
        for risk in (Risk.READ, Risk.WRITE_LOCAL, Risk.EXTERNAL,
                     Risk.DESTRUCTIVE):
            self.assertIn(risk, DEFAULT_CEILING)

    def test_forbidden_refused_outright(self):
        with self.assertRaises(ValueError) as ctx:
            self.eco.register(manifest(permissions=["read", "forbidden"]),
                              approved=True)
        self.assertIn("forbidden", str(ctx.exception).lower())
        with self.assertRaises(ValueError):
            self.eco.register(manifest(permissions=[Risk.FORBIDDEN]),
                              approved=True)
        self.assertEqual(self.eco.list_members(), [])
        self.assertNotIn("ECOSYSTEM_REGISTER", self.actions())

    def test_beyond_engine_ceiling_rejected_not_clamped(self):
        # A bounded host: this engine may only register read-only members.
        bounded = EcosystemEngine(self.core.ledger, self.core.conn,
                                  core=self.core, ceiling={Risk.READ})
        member = bounded.register(manifest(name="reader",
                                           permissions=["read"]),
                                  approved=True)
        self.assertEqual(member.permissions, (Risk.READ,))
        with self.assertRaises(ValueError) as ctx:
            bounded.register(manifest(name="writer",
                                      permissions=["read", "write_local"]),
                             approved=True)
        self.assertIn("ceiling", str(ctx.exception))
        # Rejected, not clamped: no 'writer' member exists at all.
        with self.assertRaises(KeyError):
            bounded.inspect("writer")
        self.assertEqual([m.name for m in bounded.list_members()], ["reader"])

    def test_ceiling_cannot_grant_forbidden(self):
        engine = EcosystemEngine(self.core.ledger, self.core.conn,
                                 core=self.core,
                                 ceiling={Risk.READ, Risk.FORBIDDEN})
        self.assertNotIn(Risk.FORBIDDEN, engine.ceiling)
        with self.assertRaises(ValueError):
            engine.register(manifest(permissions=["forbidden"]), approved=True)

    def test_validate_manifest_subset_directly(self):
        ok = validate_manifest(manifest(permissions=["external"]))
        self.assertEqual(ok["permissions"], (Risk.EXTERNAL,))
        with self.assertRaises(ValueError):
            validate_manifest(manifest(permissions=["destructive"]),
                              ceiling={Risk.READ, Risk.WRITE_LOCAL})


class TestGate(Base):
    def test_register_denied_without_approval_leaves_no_member(self):
        with self.assertRaises(PermissionError):
            self.eco.register(manifest(), approved=False)
        self.assertEqual(self.eco.list_members(), [])
        self.assertNotIn("ECOSYSTEM_REGISTER", self.actions())
        # The gate ledgered its own denial, as it does for every action.
        self.assertIn("GATE_DECISION", self.actions())

    def test_auto_approve_local_writes_allows_register_not_retire(self):
        self.core.close()
        self.core = Core(Settings(data_dir=self.tmp.name,
                                  auto_approve_local_writes=True))
        eco = self.core.ecosystem
        # WRITE_LOCAL registration is auto-approved by the gate flag...
        member = eco.register(manifest(), approved=False)
        self.assertEqual(member.status, "active")
        # ...but retirement is DESTRUCTIVE and still needs the owner.
        with self.assertRaises(PermissionError):
            eco.retire("console-app", approved=False)
        self.assertEqual(eco.inspect("console-app").status, "active")


class TestRetireAndActGuard(Base):
    def test_retire_ledgered_and_member_cannot_act(self):
        self.eco.register(manifest(), approved=True)
        self.assertTrue(self.eco.can_act("console-app"))
        self.assertEqual(self.eco.require_active("console-app").status,
                         "active")
        retired = self.eco.retire("console-app", reason="replaced",
                                  approved=True)
        self.assertEqual(retired.status, "retired")
        self.assertEqual(retired.retire_reason, "replaced")
        self.assertIn("ECOSYSTEM_RETIRE", self.actions())
        entry = [e for e in self.core.ledger.entries(10000)
                 if e.action == "ECOSYSTEM_RETIRE"][0]
        self.assertEqual(entry.payload["name"], "console-app")
        self.assertEqual(entry.payload["reason"], "replaced")
        self.assertTrue(self.core.ledger.verify())
        # A retired member cannot act.
        self.assertFalse(self.eco.can_act("console-app"))
        with self.assertRaises(PermissionError):
            self.eco.require_active("console-app")
        # Retiring twice is refused.
        with self.assertRaises(ValueError):
            self.eco.retire("console-app", approved=True)

    def test_retire_denied_without_approval_keeps_member_active(self):
        self.eco.register(manifest(), approved=True)
        with self.assertRaises(PermissionError):
            self.eco.retire("console-app", approved=False)
        self.assertEqual(self.eco.inspect("console-app").status, "active")
        self.assertTrue(self.eco.can_act("console-app"))
        self.assertNotIn("ECOSYSTEM_RETIRE", self.actions())

    def test_unknown_member_fails_closed(self):
        self.assertFalse(self.eco.can_act("ghost"))
        with self.assertRaises(KeyError):
            self.eco.require_active("ghost")
        with self.assertRaises(KeyError):
            self.eco.inspect("ghost")
        with self.assertRaises(KeyError):
            self.eco.retire("ghost", approved=True)


class TestCli(Base):
    def _run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_main(["--data-dir", self.tmp.name, *argv])
        return code, buf.getvalue()

    def _manifest_file(self, data=None):
        path = os.path.join(self.tmp.name, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data if data is not None else manifest(), f)
        return path

    def test_cli_full_cycle(self):
        path = self._manifest_file()
        # Registration without --approved is refused by the gate.
        code, _ = self._run_cli("ecosystem", "register", path)
        self.assertEqual(code, 1)
        code, out = self._run_cli("ecosystem", "list")
        self.assertEqual(code, 0)
        self.assertIn("no ecosystem members", out)
        code, out = self._run_cli("ecosystem", "register", path, "--approved")
        self.assertEqual(code, 0)
        self.assertIn("member console-app: kind=app status=active", out)
        code, out = self._run_cli("ecosystem", "list")
        self.assertEqual(code, 0)
        self.assertIn("console-app: kind=app status=active", out)
        code, out = self._run_cli("ecosystem", "inspect", "console-app")
        self.assertEqual(code, 0)
        self.assertIn("permissions: ['read', 'write_local']", out)
        # Retirement without --approved is refused; with it, it lands.
        code, _ = self._run_cli("ecosystem", "retire", "console-app")
        self.assertEqual(code, 1)
        code, out = self._run_cli("ecosystem", "retire", "console-app",
                                  "--approved", "--reason", "done")
        self.assertEqual(code, 0)
        self.assertIn("member console-app: retired", out)
        code, out = self._run_cli("ecosystem", "list", "--status", "retired")
        self.assertIn("status=retired", out)

    def test_cli_register_forbidden_manifest_fails(self):
        path = self._manifest_file(manifest(permissions=["forbidden"]))
        code, _ = self._run_cli("ecosystem", "register", path, "--approved")
        self.assertEqual(code, 1)
        code, out = self._run_cli("ecosystem", "list")
        self.assertIn("no ecosystem members", out)

    def test_cli_inspect_unknown_fails(self):
        code, _ = self._run_cli("ecosystem", "inspect", "ghost")
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
