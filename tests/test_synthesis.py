import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator
from steelbound.cli import main as cli_main
from steelbound.synthesis import (
    SynthesisEngine, validate_code, validate_cases, evaluate_in_sandbox,
    make_synthesized_fn)

ADD_CODE = "def run(a, b):\n    return a + b\n"
ADD_CASES = [{"args": {"a": 2, "b": 3}, "expected": 5},
             {"args": {"a": -1, "b": 1}, "expected": 0}]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = Core(Settings(data_dir=self.tmp.name))
        self.orch = Orchestrator(self.core)
        self.syn = self.core.synthesis

    def tearDown(self):
        self.core.close()
        self.tmp.cleanup()


class TestValidation(unittest.TestCase):
    def test_valid_code_passes(self):
        validate_code(ADD_CODE)
        validate_code('"""Docstring."""\ndef run(x):\n    return x * 2\n')

    def test_import_rejected(self):
        with self.assertRaises(ValueError):
            validate_code("import os\ndef run(x):\n    return x\n")

    def test_dunder_access_rejected(self):
        with self.assertRaises(ValueError):
            validate_code("def run(x):\n    return x.__class__\n")

    def test_banned_builtin_rejected(self):
        with self.assertRaises(ValueError):
            validate_code("def run(path):\n    return open(path).read()\n")
        with self.assertRaises(ValueError):
            validate_code("def run(code):\n    return eval(code)\n")

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            validate_code("def other(x):\n    return x\n")
        with self.assertRaises(ValueError):
            validate_code("x = 1\ndef run():\n    return x\n")
        with self.assertRaises(ValueError):
            validate_code("def run(x):\n    return x +\n")
        with self.assertRaises(ValueError):
            validate_code("class C:\n    pass\ndef run():\n    return 1\n")

    def test_cases_validation(self):
        with self.assertRaises(ValueError):
            validate_cases([])
        with self.assertRaises(ValueError):
            validate_cases([{"args": {"a": 1}}])
        with self.assertRaises(ValueError):
            validate_cases([{"args": [1], "expected": 1}])
        # expected=None is legitimate and must not be treated as missing.
        self.assertEqual(validate_cases([{"args": {}, "expected": None}]),
                         [{"args": {}, "expected": None}])


class TestPropose(Base):
    def test_accepted_tool_runs_and_persists(self):
        proposal = self.syn.propose("add_numbers", "Add two numbers.",
                                    ADD_CODE, ADD_CASES)
        self.assertEqual(proposal.verdict, "accepted")
        self.assertEqual(self.core.tools.run("add_numbers", {"a": 4, "b": 5}), 9)
        self.assertEqual(len(self.syn.synthesized()), 1)
        self.assertTrue(self.core.ledger.verify())
        # A kept tool survives a restart: re-registered from the table.
        self.core.close()
        reopened = Core(Settings(data_dir=self.tmp.name))
        try:
            self.assertEqual(
                reopened.tools.run("add_numbers", {"a": 1, "b": 2}), 3)
        finally:
            reopened.close()
        self.core = Core(Settings(data_dir=self.tmp.name))

    def test_failing_cases_rejected_and_nothing_kept(self):
        proposal = self.syn.propose(
            "add_numbers", "Add two numbers.", ADD_CODE,
            [{"args": {"a": 2, "b": 3}, "expected": 6}])
        self.assertEqual(proposal.verdict, "rejected")
        self.assertEqual(self.syn.synthesized(), [])
        self.assertIsNone(self.core.tools.get("add_numbers"))
        with self.assertRaises(KeyError):
            self.core.tools.run("add_numbers", {"a": 1, "b": 2})

    def test_runtime_error_case_rejected_with_reason(self):
        proposal = self.syn.propose(
            "divide", "Divide a by b.",
            "def run(a, b):\n    return a / b\n",
            [{"args": {"a": 1, "b": 0}, "expected": 0}])
        self.assertEqual(proposal.verdict, "rejected")
        self.assertIn("ZeroDivisionError", proposal.detail)
        self.assertEqual(self.syn.synthesized(), [])

    def test_timeout_rejected(self):
        proposal = self.syn.propose(
            "loop_forever", "Never returns.",
            "def run():\n    while True:\n        pass\n",
            [{"args": {}, "expected": None}],
            timeout_seconds=2.0)
        self.assertEqual(proposal.verdict, "error")
        self.assertIn("timeout", proposal.detail)
        self.assertEqual(self.syn.synthesized(), [])

    def test_name_and_collision_rules(self):
        with self.assertRaises(ValueError):
            self.syn.propose("Bad Name", "x", ADD_CODE, ADD_CASES)
        with self.assertRaises(ValueError):
            self.syn.propose("read_file", "collision", ADD_CODE, ADD_CASES)
        self.syn.propose("add_numbers", "Add two numbers.", ADD_CODE, ADD_CASES)
        with self.assertRaises(ValueError):
            self.syn.propose("add_numbers", "again", ADD_CODE, ADD_CASES)
        with self.assertRaises(ValueError):
            self.syn.propose("add_numbers2", "", ADD_CODE, ADD_CASES)

    def test_malformed_candidate_raises_not_ledgered_as_proposal(self):
        before = self.core.ledger.count()
        with self.assertRaises(ValueError):
            self.syn.propose("bad_tool", "x", "import os\ndef run():\n    return 1\n",
                             ADD_CASES)
        # No SYNTHESIS_PROPOSED entry for a candidate that failed
        # validation: only a well-formed proposal reaches the ledger.
        actions = [e.action for e in self.core.ledger.entries(10000)]
        self.assertNotIn("SYNTHESIS_PROPOSED", actions)
        self.assertEqual(self.core.ledger.count(), before)

    def test_accept_and_reject_both_ledgered(self):
        self.syn.propose("add_numbers", "Add.", ADD_CODE, ADD_CASES)
        self.syn.propose("bad_add", "Add, wrongly.", ADD_CODE,
                         [{"args": {"a": 1, "b": 1}, "expected": 3}])
        actions = [e.action for e in self.core.ledger.entries(10000)]
        self.assertIn("SYNTHESIS_ACCEPTED", actions)
        self.assertIn("SYNTHESIS_REJECTED", actions)

    def test_remove_unregisters_and_deletes(self):
        self.syn.propose("add_numbers", "Add.", ADD_CODE, ADD_CASES)
        removed = self.syn.remove("add_numbers")
        self.assertEqual(removed.name, "add_numbers")
        self.assertEqual(self.syn.synthesized(), [])
        self.assertIsNone(self.core.tools.get("add_numbers"))
        with self.assertRaises(KeyError):
            self.syn.remove("add_numbers")

    def test_sandbox_cannot_reach_filesystem(self):
        # `open` is a banned name, so propose() refuses the candidate
        # at validation; and even if a candidate reached the sandbox
        # runner directly, its namespace has no open — the case fails
        # with NameError instead of reading anything.
        with self.assertRaises(ValueError):
            self.syn.propose(
                "sneaky", "Tries to read a file.",
                "def run():\n    return open('/etc/hostname').read()\n",
                [{"args": {}, "expected": "x"}])
        result = evaluate_in_sandbox(
            "def run():\n    return open('/etc/hostname').read()\n",
            [{"args": {}, "expected": "x"}])
        self.assertEqual(result["verdict"], "rejected")
        self.assertIn("NameError", result["detail"])


class TestSandboxDirect(unittest.TestCase):
    def test_evaluate_accepts_all_pass(self):
        result = evaluate_in_sandbox(ADD_CODE, ADD_CASES)
        self.assertEqual(result["verdict"], "accepted")

    def test_evaluate_reports_failing_case(self):
        result = evaluate_in_sandbox(
            ADD_CODE, [{"args": {"a": 1, "b": 2}, "expected": 4}])
        self.assertEqual(result["verdict"], "rejected")
        self.assertIn("case 0 failed", result["detail"])

    def test_live_fn_sandboxed_call(self):
        fn = make_synthesized_fn(ADD_CODE)
        self.assertEqual(fn(a=10, b=20), 30)
        with self.assertRaises(RuntimeError):
            make_synthesized_fn("def run(a, b):\n    return a / b\n")(a=1, b=0)


class TestSynthesisCLI(Base):
    def test_cli_propose_list_tool_remove(self):
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as code_file:
            code_file.write(ADD_CODE)
            code_path = code_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as cases_file:
            import json
            json.dump(ADD_CASES, cases_file)
            cases_path = cases_file.name
        try:
            rc = cli_main(["--data-dir", self.tmp.name, "synthesize", "propose",
                           "add_numbers", "--description", "Add two numbers.",
                           "--code-file", code_path,
                           "--cases-file", cases_path])
            self.assertEqual(rc, 0)
            rc = cli_main(["--data-dir", self.tmp.name, "synthesize", "list"])
            self.assertEqual(rc, 0)
            rc = cli_main(["--data-dir", self.tmp.name, "tool", "add_numbers",
                           "--args", '{"a": 3, "b": 4}'])
            self.assertEqual(rc, 0)
            rc = cli_main(["--data-dir", self.tmp.name, "synthesize", "remove",
                           "add_numbers"])
            self.assertEqual(rc, 0)
            rc = cli_main(["--data-dir", self.tmp.name, "synthesize", "remove",
                           "add_numbers"])
            self.assertEqual(rc, 1)
        finally:
            os.unlink(code_path)
            os.unlink(cases_path)

    def test_cli_rejected_proposal_exits_nonzero(self):
        rc = cli_main(["--data-dir", self.tmp.name, "synthesize", "propose",
                       "bad_add", "--description", "Wrong add.",
                       "--code", ADD_CODE,
                       "--cases", '[{"args": {"a": 1, "b": 1}, "expected": 3}]'])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
