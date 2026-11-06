"""Toolpack SDK: gated, validated, risk-clamped pack loading.

Written against the API as wired in the repo: ``core.toolpacks`` is
the core's ToolpackManager (built in config.py) and
``load_toolpacks(core)`` runs at Core startup, so a pack loaded in
one Core is live again in a reopened Core on the same data dir, and
is an active ecosystem member of kind "toolpack" in both.

Covers: the example pack loading and running through the registry,
the load/unload gate, manifest and static-validation refusals
(nothing partially registered), risk clamping semantics (FORBIDDEN
always refused; above-ceiling risks refused unless the loader allows
elevation *and* the load is approved), persistence across a Core
reopen, unload, the ecosystem hook (real registry and duck-typed
stub), and the CLI surface.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.cli import main as cli_main
from steelbound.gate import Risk
from steelbound.toolpacks import (
    ToolpackError, ToolpackManager, load_toolpack, load_toolpacks,
    unload_toolpack, validate_pack_code)

EXAMPLE_PACK = os.path.join(os.path.dirname(__file__), "..",
                            "examples", "toolpacks", "text_stats")
SAMPLE = "The quick brown fox jumps. The lazy dog sleeps!"

GOOD_CODE = "def shout(text):\n    return text.upper() + '!'\n"


def write_pack(root, name, tools, modules, manifest_overrides=None):
    """Write a pack directory under ``root`` and return its path.

    ``tools`` is the manifest tool list; ``modules`` maps filename ->
    source. ``manifest_overrides`` may replace any top-level manifest
    field (or, with the key '_raw', the whole manifest object)."""
    pack_dir = os.path.join(root, name)
    os.makedirs(pack_dir, exist_ok=True)
    manifest = {"name": name, "version": "1.0.0",
                "description": f"The {name} pack.", "tools": tools}
    if manifest_overrides:
        manifest.update(manifest_overrides)
    if "_raw" in (manifest_overrides or {}):
        manifest = manifest_overrides["_raw"]
    with open(os.path.join(pack_dir, "pack.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f)
    for filename, source in modules.items():
        with open(os.path.join(pack_dir, filename), "w",
                  encoding="utf-8") as f:
            f.write(source)
    return pack_dir


def one_tool_pack(root, name="shout_pack", risk="read", code=GOOD_CODE,
                  tool_name="shout", entrypoint="main.py:shout"):
    return write_pack(
        root, name,
        [{"name": tool_name, "description": "Shout a text.",
          "risk": risk, "entrypoint": entrypoint,
          "schema": {"text": "str"}}],
        {"main.py": code})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = Core(Settings(data_dir=self.tmp.name))

    def tearDown(self):
        self.core.close()
        self.tmp.cleanup()

    def actions(self):
        return [e.action for e in self.core.ledger.entries(10000)]

    def tool_names(self):
        return sorted(t.name for t in self.core.tools.list())


class TestValidatePackCode(unittest.TestCase):
    def test_plain_functions_and_allowlisted_imports_pass(self):
        validate_pack_code(
            "import math\nfrom collections import Counter\n"
            "def area(r):\n    return math.pi * r * r\n")
        validate_pack_code('"""Doc."""\nX = 3\ndef run(x):\n    return x + X\n')

    def test_disallowed_imports_rejected(self):
        for code in ("import os\ndef f():\n    return 1\n",
                     "import sys\ndef f():\n    return 1\n",
                     "from subprocess import run\ndef f():\n    return 1\n",
                     "def f():\n    import os\n    return 1\n",
                     "from . import sibling\ndef f():\n    return 1\n"):
            with self.assertRaises(ToolpackError, msg=code):
                validate_pack_code(code)

    def test_dangerous_names_and_dunder_rejected(self):
        for code in ("def f(p):\n    return open(p).read()\n",
                     "def f(c):\n    return eval(c)\n",
                     "def f(x):\n    return x.__class__\n",
                     "def f(x):\n    return getattr(x, 'y')\n"):
            with self.assertRaises(ToolpackError, msg=code):
                validate_pack_code(code)

    def test_import_time_statements_rejected(self):
        with self.assertRaises(ToolpackError):
            validate_pack_code("print('side effect')\ndef f():\n    return 1\n")
        with self.assertRaises(ToolpackError):
            validate_pack_code("def f():\n    global x\n    return 1\n")

    def test_unparseable_rejected(self):
        with self.assertRaises(ToolpackError):
            validate_pack_code("def f(:\n")


class TestExamplePack(Base):
    def test_example_pack_loads_and_runs_through_registry(self):
        result = load_toolpack(self.core, EXAMPLE_PACK, approved=True)
        self.assertEqual(result.name, "text_stats")
        self.assertEqual(result.pack_name, "text_stats")
        self.assertEqual(result.version, "1.0.0")
        self.assertEqual(sorted(result.tools),
                         ["text_stats", "word_frequency"])
        tool = self.core.tools.get("text_stats")
        self.assertIsNotNone(tool)
        self.assertIs(tool.risk, Risk.READ)
        stats = self.core.tools.run("text_stats", {"text": SAMPLE})
        self.assertEqual(stats["words"], 9)
        self.assertEqual(stats["characters"], len(SAMPLE))
        self.assertEqual(stats["sentences"], 2)
        self.assertEqual(stats["unique_words"], 8)
        self.assertEqual(stats["reading_time_minutes"], 0.045)
        freq = self.core.tools.run(
            "word_frequency", {"text": SAMPLE, "top_n": 3})
        self.assertEqual(freq, {"the": 2, "brown": 1, "dog": 1})
        self.assertIn("TOOLPACK_LOADED", self.actions())
        self.assertTrue(self.core.ledger.verify())

    def test_core_wired_manager_loads(self):
        # config.py wires core.toolpacks as the manager; loading
        # through it must register into the core's registry.
        self.assertIsInstance(self.core.toolpacks, ToolpackManager)
        result = self.core.toolpacks.load_toolpack(EXAMPLE_PACK,
                                                   approved=True)
        self.assertEqual(result.name, "text_stats")
        self.assertIsNotNone(self.core.tools.get("text_stats"))
        self.assertEqual(
            [p.name for p in self.core.toolpacks.loaded_packs()],
            ["text_stats"])

    def test_load_requires_gate_approval(self):
        with self.assertRaises(PermissionError):
            load_toolpack(self.core, EXAMPLE_PACK)
        self.assertIsNone(self.core.tools.get("text_stats"))
        self.assertIn("TOOLPACK_REFUSED", self.actions())

    def test_double_load_refused(self):
        load_toolpack(self.core, EXAMPLE_PACK, approved=True)
        with self.assertRaises(ToolpackError):
            load_toolpack(self.core, EXAMPLE_PACK, approved=True)

    def test_pack_persists_across_core_reopen(self):
        load_toolpack(self.core, EXAMPLE_PACK, approved=True)
        self.core.close()
        reopened = Core(Settings(data_dir=self.tmp.name))
        try:
            # Startup wiring (config.py) already reloaded the pack:
            # tools are live and the ecosystem member is active,
            # with no explicit load call in between.
            self.assertIsNotNone(reopened.tools.get("text_stats"))
            stats = reopened.tools.run("text_stats", {"text": SAMPLE})
            self.assertEqual(stats["words"], 9)
            member = reopened.ecosystem.inspect("text_stats")
            self.assertEqual(member.kind, "toolpack")
            self.assertEqual(member.status, "active")
            self.assertEqual(member.permissions, (Risk.READ,))
            # An explicit startup reload is idempotent.
            self.assertEqual(load_toolpacks(reopened), ["text_stats"])
            self.assertTrue(reopened.ledger.verify())
        finally:
            reopened.close()
        self.core = Core(Settings(data_dir=self.tmp.name))

    def test_changed_manifest_is_not_reloaded(self):
        pack_copy = os.path.join(self.tmp.name, "packs", "text_stats")
        os.makedirs(os.path.dirname(pack_copy))
        shutil.copytree(EXAMPLE_PACK, pack_copy)
        load_toolpack(self.core, pack_copy, approved=True)
        unload_toolpack(self.core, "text_stats", approved=True)
        load_toolpack(self.core, pack_copy, approved=True)
        # Tamper with the on-disk manifest after the gated load.
        manifest_path = os.path.join(pack_copy, "pack.json")
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        data["description"] = "Tampered description."
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        self.core.close()
        reopened = Core(Settings(data_dir=self.tmp.name))
        try:
            # Startup reload refused the changed pack, and an
            # explicit reload agrees.
            self.assertIsNone(reopened.tools.get("text_stats"))
            self.assertEqual(load_toolpacks(reopened), [])
        finally:
            reopened.close()
        self.core = Core(Settings(data_dir=self.tmp.name))


class TestRiskSemantics(Base):
    def test_write_local_tool_runs_only_with_approval(self):
        pack_dir = one_tool_pack(self.tmp.name, risk="write_local")
        load_toolpack(self.core, pack_dir, approved=True)
        self.assertIs(self.core.tools.get("shout").risk, Risk.WRITE_LOCAL)
        with self.assertRaises(PermissionError):
            self.core.tools.run("shout", {"text": "hey"})
        self.assertEqual(
            self.core.tools.run("shout", {"text": "hey"}, approved=True),
            "HEY!")

    def test_external_tool_refused_by_default_loader(self):
        pack_dir = one_tool_pack(self.tmp.name, name="net_pack",
                                 risk="external", tool_name="fetch_thing")
        before = self.tool_names()
        with self.assertRaises(ToolpackError):
            load_toolpack(self.core, pack_dir, approved=True)
        self.assertEqual(self.tool_names(), before)
        self.assertIn("TOOLPACK_REFUSED", self.actions())

    def test_external_tool_loads_with_elevated_loader_and_approval(self):
        pack_dir = one_tool_pack(self.tmp.name, name="net_pack",
                                 risk="external", tool_name="fetch_thing")
        manager = ToolpackManager(self.core.ledger, self.core.conn,
                                  self.core.tools, self.core.gate,
                                  allow_elevated=True, core=self.core)
        result = manager.load_toolpack(pack_dir, approved=True)
        self.assertEqual(result.tools, ["fetch_thing"])
        # The tool keeps its true risk: the gate still applies per call.
        self.assertIs(self.core.tools.get("fetch_thing").risk, Risk.EXTERNAL)
        with self.assertRaises(PermissionError):
            self.core.tools.run("fetch_thing", {"text": "hey"})
        self.assertEqual(
            self.core.tools.run("fetch_thing", {"text": "hey"},
                                approved=True),
            "HEY!")

    def test_elevation_without_approval_still_refused(self):
        pack_dir = one_tool_pack(self.tmp.name, name="net_pack",
                                 risk="external", tool_name="fetch_thing")
        # Per-call elevation flag, but the load itself is not approved:
        # the load gate (WRITE_LOCAL) denies before risk is considered.
        with self.assertRaises(PermissionError):
            load_toolpack(self.core, pack_dir, approved=False,
                          allow_elevated=True)
        self.assertIsNone(self.core.tools.get("fetch_thing"))

    def test_forbidden_tool_refused_even_when_elevated(self):
        pack_dir = one_tool_pack(self.tmp.name, name="bad_pack",
                                 risk="forbidden", tool_name="doom")
        manager = ToolpackManager(self.core.ledger, self.core.conn,
                                  self.core.tools, self.core.gate,
                                  allow_elevated=True, core=self.core)
        with self.assertRaises(ToolpackError):
            manager.load_toolpack(pack_dir, approved=True)
        self.assertIsNone(self.core.tools.get("doom"))

    def test_narrower_ceiling_refuses_write_local(self):
        pack_dir = one_tool_pack(self.tmp.name, risk="write_local")
        manager = ToolpackManager(self.core.ledger, self.core.conn,
                                  self.core.tools, self.core.gate,
                                  ceiling=Risk.READ, core=self.core)
        with self.assertRaises(ToolpackError):
            manager.load_toolpack(pack_dir, approved=True)
        self.assertIsNone(self.core.tools.get("shout"))

    def test_one_bad_risk_refuses_whole_pack(self):
        pack_dir = write_pack(
            self.tmp.name, "mixed_pack",
            [{"name": "fine_tool", "description": "Fine.",
              "risk": "read", "entrypoint": "main.py:shout",
              "schema": {"text": "str"}},
             {"name": "doom", "description": "Not fine.",
              "risk": "forbidden", "entrypoint": "main.py:shout",
              "schema": {"text": "str"}}],
            {"main.py": GOOD_CODE})
        with self.assertRaises(ToolpackError):
            load_toolpack(self.core, pack_dir, approved=True)
        self.assertIsNone(self.core.tools.get("fine_tool"))
        self.assertIsNone(self.core.tools.get("doom"))


class TestMaliciousAndMalformed(Base):
    def test_import_os_refused_nothing_registered(self):
        pack_dir = one_tool_pack(
            self.tmp.name, name="evil_pack",
            code="import os\ndef shout(text):\n    return os.getcwd()\n")
        before = self.tool_names()
        with self.assertRaises(ToolpackError):
            load_toolpack(self.core, pack_dir, approved=True)
        self.assertEqual(self.tool_names(), before)
        self.assertIn("TOOLPACK_REFUSED", self.actions())

    def test_open_call_refused(self):
        pack_dir = one_tool_pack(
            self.tmp.name, name="evil_pack",
            code="def shout(text):\n    return open(text).read()\n")
        with self.assertRaises(ToolpackError):
            load_toolpack(self.core, pack_dir, approved=True)
        self.assertIsNone(self.core.tools.get("shout"))

    def test_unreferenced_malicious_module_refuses_pack(self):
        # Every module is validated, not just entrypoint modules.
        pack_dir = write_pack(
            self.tmp.name, "sneaky_pack",
            [{"name": "shout", "description": "Shout.",
              "risk": "read", "entrypoint": "main.py:shout",
              "schema": {"text": "str"}}],
            {"main.py": GOOD_CODE,
             "helper.py": "import subprocess\ndef x():\n    return 1\n"})
        with self.assertRaises(ToolpackError):
            load_toolpack(self.core, pack_dir, approved=True)
        self.assertIsNone(self.core.tools.get("shout"))

    def test_missing_entrypoint_function_refused(self):
        pack_dir = one_tool_pack(self.tmp.name, name="broken_pack",
                                 entrypoint="main.py:nonexistent")
        with self.assertRaises(ToolpackError):
            load_toolpack(self.core, pack_dir, approved=True)
        self.assertIsNone(self.core.tools.get("shout"))

    def test_tool_name_collision_with_builtin_refused(self):
        pack_dir = one_tool_pack(self.tmp.name, name="collide_pack",
                                 tool_name="read_file")
        with self.assertRaises(ToolpackError):
            load_toolpack(self.core, pack_dir, approved=True)

    def test_malformed_manifests_refused(self):
        root = self.tmp.name
        cases = []
        # No pack.json at all.
        empty_dir = os.path.join(root, "empty_pack")
        os.makedirs(empty_dir)
        cases.append(empty_dir)
        # Invalid JSON.
        bad_json = os.path.join(root, "bad_json")
        os.makedirs(bad_json)
        with open(os.path.join(bad_json, "pack.json"), "w") as f:
            f.write("{not json")
        cases.append(bad_json)
        # Manifest without tools / with an unknown risk / bad entrypoint.
        cases.append(write_pack(root, "no_tools", [],
                                {"main.py": GOOD_CODE}))
        cases.append(one_tool_pack(root, name="bad_risk", risk="fly"))
        cases.append(one_tool_pack(root, name="bad_entry",
                                   entrypoint="main.py"))
        cases.append(one_tool_pack(root, name="missing_mod",
                                   entrypoint="nope.py:shout"))
        before = self.tool_names()
        for pack_dir in cases:
            with self.assertRaises(ToolpackError, msg=pack_dir):
                load_toolpack(self.core, pack_dir, approved=True)
        self.assertEqual(self.tool_names(), before)
        self.assertTrue(self.core.ledger.verify())


class TestUnload(Base):
    def test_unload_removes_tools_and_persistence(self):
        load_toolpack(self.core, EXAMPLE_PACK, approved=True)
        record = unload_toolpack(self.core, "text_stats", approved=True)
        self.assertEqual(record.name, "text_stats")
        self.assertIsNone(self.core.tools.get("text_stats"))
        self.assertIsNone(self.core.tools.get("word_frequency"))
        with self.assertRaises(KeyError):
            self.core.tools.run("text_stats", {"text": "x"})
        self.assertIn("TOOLPACK_UNLOADED", self.actions())
        # The persistence row is gone: a startup reload finds nothing,
        # and the pack can be loaded fresh again afterwards.
        self.assertEqual(load_toolpacks(self.core), [])
        load_toolpack(self.core, EXAMPLE_PACK, approved=True)
        self.assertIsNotNone(self.core.tools.get("text_stats"))

    def test_unload_requires_approval(self):
        load_toolpack(self.core, EXAMPLE_PACK, approved=True)
        with self.assertRaises(PermissionError):
            unload_toolpack(self.core, "text_stats")
        self.assertIsNotNone(self.core.tools.get("text_stats"))

    def test_unload_unknown_pack_raises(self):
        with self.assertRaises(KeyError):
            unload_toolpack(self.core, "ghost_pack", approved=True)


class FakeMember:
    def __init__(self, name):
        self.name = name
        self.status = "active"


class FakeEcosystem:
    """Duck-typed stand-in for the ecosystem registry: records calls,
    mimics the real engine's register/retire/inspect surface."""

    def __init__(self, explode=False):
        self.registered = []
        self.retired = []
        self.members = {}
        self.explode = explode

    def inspect(self, name):
        if name not in self.members:
            raise KeyError(name)
        return self.members[name]

    def register(self, manifest, approved=False):
        if self.explode:
            raise RuntimeError("registry unavailable")
        self.registered.append((manifest, approved))
        member = FakeMember(manifest["name"])
        self.members[manifest["name"]] = member
        return member

    def retire(self, name, reason=None, approved=False):
        self.retired.append((name, reason, approved))
        if name in self.members:
            self.members[name].status = "retired"


class TestEcosystemHook(Base):
    def test_real_registry_member_lifecycle(self):
        # The real Core carries an ecosystem registry; loading
        # registers the pack as a "toolpack" member and unloading
        # retires it.
        load_toolpack(self.core, EXAMPLE_PACK, approved=True)
        member = self.core.ecosystem.inspect("text_stats")
        self.assertEqual(member.kind, "toolpack")
        self.assertEqual(member.status, "active")
        self.assertEqual(member.permissions, (Risk.READ,))
        self.assertEqual(sorted(member.capabilities),
                         ["text_stats", "word_frequency"])
        self.assertTrue(self.core.ecosystem.can_act("text_stats"))
        unload_toolpack(self.core, "text_stats", approved=True)
        member = self.core.ecosystem.inspect("text_stats")
        self.assertEqual(member.status, "retired")
        self.assertFalse(self.core.ecosystem.can_act("text_stats"))
        self.assertTrue(self.core.ledger.verify())

    def test_stub_registry_records_register_and_retire(self):
        fake = FakeEcosystem()
        self.core.ecosystem = fake
        load_toolpack(self.core, EXAMPLE_PACK, approved=True)
        self.assertEqual(len(fake.registered), 1)
        manifest, approved = fake.registered[0]
        self.assertTrue(approved)
        self.assertEqual(manifest["name"], "text_stats")
        self.assertEqual(manifest["kind"], "toolpack")
        self.assertEqual(manifest["permissions"], ["read"])
        self.assertEqual(sorted(manifest["capabilities"]),
                         ["text_stats", "word_frequency"])
        self.assertEqual(manifest["metadata"]["version"], "1.0.0")
        unload_toolpack(self.core, "text_stats", approved=True)
        self.assertEqual(fake.retired,
                         [("text_stats", "toolpack unloaded", True)])
        self.assertEqual(fake.members["text_stats"].status, "retired")

    def test_broken_ecosystem_never_breaks_loading(self):
        self.core.ecosystem = FakeEcosystem(explode=True)
        result = load_toolpack(self.core, EXAMPLE_PACK, approved=True)
        self.assertEqual(result.name, "text_stats")
        self.assertIsNotNone(self.core.tools.get("text_stats"))
        unload_toolpack(self.core, "text_stats", approved=True)
        self.assertIsNone(self.core.tools.get("text_stats"))


class TestToolpackCLI(Base):
    def _run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_main(["--data-dir", self.tmp.name, *argv])
        return code, buf.getvalue()

    def test_cli_load_list_run_unload(self):
        # Loading without --approved is refused by the gate.
        code, _ = self._run_cli("toolpack", "load", EXAMPLE_PACK)
        self.assertEqual(code, 1)
        code, out = self._run_cli("toolpack", "load", EXAMPLE_PACK,
                                  "--approved")
        self.assertEqual(code, 0)
        self.assertIn("toolpack text_stats loaded", out)
        # A fresh CLI invocation reopens the Core: the pack was
        # reloaded at startup and shows in the list.
        code, out = self._run_cli("toolpack", "list")
        self.assertEqual(code, 0)
        self.assertIn("text_stats v1.0.0", out)
        code, _ = self._run_cli(
            "tool", "text_stats", "--args",
            '{"text": "One two three. Four."}')
        self.assertEqual(code, 0)
        code, out = self._run_cli("toolpack", "unload", "text_stats",
                                  "--approved")
        self.assertEqual(code, 0)
        self.assertIn("toolpack text_stats unloaded", out)
        code, out = self._run_cli("toolpack", "list")
        self.assertEqual(code, 0)
        self.assertIn("no toolpacks loaded", out)


if __name__ == "__main__":
    unittest.main()
