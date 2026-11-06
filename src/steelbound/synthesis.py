"""Program synthesis — Synthesis S3 (docs/SYNTHESIS.md, decision D-016,
Gen 3: "synthesize").

The model generates text, not programs — until now. Here a *tool* can
be proposed as code, and it is kept only when its tests pass in a
sandbox. The pipeline, enforced in code:

1. ``propose(name, description, code, cases)`` — a candidate is a
   single pure function ``run(...)`` plus a list of test cases
   (``{"args": {...}, "expected": ...}``). The name must be a fresh,
   valid tool name (no collision with a built-in or already-kept
   synthesized tool).
2. Static validation (``validate_code``): the source must parse, must
   define exactly one top-level function named ``run`` (an optional
   module docstring is allowed), and must contain no import, no
   dunder access, and no dangerous builtin (``open``, ``eval``,
   ``exec``, ``__import__``, ``getattr``/``setattr``/``input`` and
   friends). A synthesized tool is pure computation by construction:
   it cannot touch the filesystem, the network, or the core.
3. Sandbox evaluation (``evaluate_in_sandbox``): the candidate and
   its cases run in a *separate process* with a restricted builtins
   namespace, under a timeout. A timeout, a crash, a missing argument,
   or any failing case rejects the candidate — nothing is kept on a
   partial pass, and the rejection is ledgered with the failing cases.
4. Acceptance: only an all-pass candidate is persisted (in the
   ``synthesized_tools`` table, with its code and cases) and
   registered live as a READ-risk tool, so it runs through the same
   gate and ledger as every other tool. Kept tools are re-registered
   on Core startup from the table (``load_synthesized_tools``), so a
   kept tool survives a restart; removal deletes the row and
   unregisters it live (``remove``).

A kept tool's live calls also run in the sandbox subprocess, not in
the core's process: the same restriction that judged it guards every
call, so a kept tool can never grow side effects after acceptance.
Its return value must be JSON-serializable (anything else is returned
as its ``repr`` string, so a tool call can never crash the caller on
an exotic return type).

Bounds: synthesis never touches the ledger, the gate, the eval
suites, or the clean-room policy — it only adds READ-risk pure
tools, and a tool that fails its own tests is rejected, never kept.
No network is used anywhere in this module.
"""
from __future__ import annotations

import ast
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from .gate import Risk
from .tools import Tool

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# Builtins a synthesized tool may never name: anything that reaches
# outside pure computation (I/O, evaluation, introspection escapes).
_BANNED_NAMES = frozenset({
    "open", "eval", "exec", "compile", "__import__", "input", "help",
    "breakpoint", "exit", "quit", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "memoryview", "object", "type",
})
# The only builtins available inside the sandbox. Deliberately small:
# arithmetic, comparison, and sequence/string helpers — no I/O.
_SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "len", "list", "max", "min", "range", "round", "sorted", "str",
    "sum", "tuple", "zip", "reversed", "isinstance",
)

# The sandbox runner: a small script written to a temp dir and run as
# `python3 runner.py candidate.py request.json`. It execs the
# candidate under restricted builtins and either runs its test cases
# ("mode": "cases") or makes one live call ("mode": "call"), then
# prints exactly one JSON object on stdout. It never imports the
# candidate's surroundings and never touches the parent core.
_RUNNER_SOURCE = '''
import json, sys
SAFE = ("abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "max", "min", "range", "round", "sorted", "str",
        "sum", "tuple", "zip", "reversed", "isinstance")
import builtins as _b
namespace = {"__builtins__": {n: getattr(_b, n) for n in SAFE}}
with open(sys.argv[1], encoding="utf-8") as f:
    source = f.read()
with open(sys.argv[2], encoding="utf-8") as f:
    request = json.load(f)
exec(compile(source, "synthesized_tool", "exec"), namespace)
run = namespace["run"]
if request["mode"] == "cases":
    results = []
    for case in request["cases"]:
        try:
            got = run(**case["args"])
            results.append({"passed": got == case["expected"],
                            "got": repr(got), "error": None})
        except Exception as exc:
            results.append({"passed": False, "got": None,
                            "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps({"results": results}))
else:
    try:
        value = run(**request["args"])
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
    else:
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            value = repr(value)
        print(json.dumps({"value": value}))
'''


def validate_code(code: str) -> None:
    """Static gate for candidate source. Raises ValueError on any
    shape a pure tool must not have; passing here does not mean the
    tool is kept — its cases must still pass in the sandbox."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("synthesized tool code must be a non-empty string")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"synthesized tool code does not parse: {exc}")
    defs = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue  # module docstring
        if isinstance(node, ast.FunctionDef):
            defs.append(node)
        else:
            raise ValueError(
                "synthesized tool code must define exactly one function"
                " 'run' and nothing else at module level")
    if len(defs) != 1 or defs[0].name != "run":
        raise ValueError(
            "synthesized tool code must define exactly one function 'run'")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("synthesized tool code may not import anything")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("synthesized tool code may not use dunder attributes")
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise ValueError(f"synthesized tool code may not use {node.id!r}")
        if isinstance(node, ast.ClassDef):
            raise ValueError("synthesized tool code may not define classes")


def validate_cases(cases) -> list[dict]:
    """Cases are [{"args": {...}, "expected": ...}]; at least one,
    args a dict, expected present (it may legitimately be None, so
    presence — not truthiness — is checked)."""
    if not isinstance(cases, list) or not cases:
        raise ValueError("synthesized tool cases must be a non-empty list")
    cleaned = []
    for i, case in enumerate(cases):
        if not isinstance(case, dict) or "args" not in case \
                or "expected" not in case:
            raise ValueError(f"case {i} needs 'args' and 'expected'")
        if not isinstance(case["args"], dict):
            raise ValueError(f"case {i} 'args' must be an object")
        try:
            json.dumps(case)
        except (TypeError, ValueError):
            raise ValueError(f"case {i} is not JSON-serializable")
        cleaned.append({"args": case["args"], "expected": case["expected"]})
    return cleaned


def _run_sandbox(code: str, request: dict, timeout_seconds: float) -> dict:
    """Run the sandbox runner in a separate process. Returns its JSON
    output, or {"error": ...} on timeout / crash / bad output."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    with tempfile.TemporaryDirectory() as tmp:
        candidate = f"{tmp}/candidate.py"
        request_path = f"{tmp}/request.json"
        runner = f"{tmp}/runner.py"
        with open(candidate, "w", encoding="utf-8") as f:
            f.write(code)
        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(request, f)
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_RUNNER_SOURCE)
        try:
            proc = subprocess.run(
                [sys.executable, runner, candidate, request_path],
                capture_output=True, text=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return {"error": "sandbox timeout"}
        if proc.returncode != 0:
            return {"error": f"sandbox runner failed: {proc.stderr.strip()[-200:]}"}
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {"error": "sandbox runner returned unreadable output"}


def evaluate_in_sandbox(code: str, cases: list[dict],
                        timeout_seconds: float = 10.0) -> dict:
    """Score a candidate against its cases in the sandbox. Returns
    {"verdict": "accepted"|"rejected"|"error", "results": [...],
    "detail": str}. A verdict of accepted requires every case to pass;
    anything else keeps nothing."""
    results = _run_sandbox(
        code, {"mode": "cases", "cases": cases}, timeout_seconds)
    if "error" in results:
        return {"verdict": "error", "results": [],
                "detail": results["error"]}
    case_results = results["results"]
    failed = [i for i, r in enumerate(case_results) if not r["passed"]]
    if not failed:
        return {"verdict": "accepted", "results": case_results,
                "detail": f"all {len(case_results)} case(s) passed in the sandbox"}
    detail = "; ".join(
        f"case {i} failed"
        + (f" ({case_results[i]['error']})" if case_results[i]["error"]
           else f" (got {case_results[i]['got']})")
        for i in failed)
    return {"verdict": "rejected", "results": case_results, "detail": detail}


def make_synthesized_fn(code: str, timeout_seconds: float = 10.0):
    """The live callable for a kept tool: every call runs in the
    sandbox subprocess, under the same restrictions as evaluation."""
    def _call(**args):
        result = _run_sandbox(
            code, {"mode": "call", "args": args}, timeout_seconds)
        if "error" in result:
            raise RuntimeError(f"synthesized tool failed: {result['error']}")
        return result["value"]
    return _call


@dataclass(frozen=True)
class SynthesizedTool:
    name: str
    description: str
    code: str
    cases: list[dict]
    ts: float


@dataclass(frozen=True)
class SynthesisProposal:
    name: str
    verdict: str  # accepted | rejected | error
    detail: str
    results: list[dict]


class SynthesisEngine:
    def __init__(self, ledger, conn: sqlite3.Connection, tools=None):
        self.ledger = ledger
        self.conn = conn
        self.tools = tools
        conn.execute("""CREATE TABLE IF NOT EXISTS synthesized_tools(
            name TEXT PRIMARY KEY, description TEXT NOT NULL,
            code TEXT NOT NULL, cases TEXT NOT NULL, ts REAL NOT NULL)""")
        conn.commit()

    # -- kept tools ----------------------------------------------------------
    def synthesized(self) -> list[SynthesizedTool]:
        rows = self.conn.execute(
            "SELECT name,description,code,cases,ts FROM synthesized_tools"
            " ORDER BY name ASC").fetchall()
        return [SynthesizedTool(r[0], r[1], r[2], json.loads(r[3]), r[4])
                for r in rows]

    def _register_live(self, tool: SynthesizedTool) -> None:
        # Schema is informational for pure tools: the runner passes
        # whatever arguments the caller supplies to run(**args).
        if self.tools is not None:
            self.tools.register(Tool(
                tool.name, tool.description, Risk.READ,
                make_synthesized_fn(tool.code), {}))

    # -- propose / evaluate / keep ----------------------------------------------
    def propose(self, name: str, description: str, code: str, cases,
                timeout_seconds: float = 10.0) -> SynthesisProposal:
        """Validate, sandbox-test, and keep-or-reject a candidate tool.
        Validation failures raise before anything is ledgered as a
        proposal (a malformed candidate is not a proposal); a
        well-formed candidate that fails its cases is ledgered as
        SYNTHESIS_REJECTED and kept nowhere."""
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ValueError(
                f"tool name {name!r} must match ^[a-z][a-z0-9_]*$")
        if self.tools is not None and self.tools.get(name) is not None:
            raise ValueError(f"a tool named {name!r} already exists")
        if self.conn.execute(
                "SELECT 1 FROM synthesized_tools WHERE name=?",
                (name,)).fetchone() is not None:
            raise ValueError(f"a synthesized tool named {name!r} already exists")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("synthesized tool needs a non-empty description")
        validate_code(code)
        cleaned_cases = validate_cases(cases)
        if self.ledger:
            self.ledger.append("synthesis", "SYNTHESIS_PROPOSED", {
                "name": name, "description": description,
                "cases": len(cleaned_cases)})
        scored = evaluate_in_sandbox(code, cleaned_cases,
                                     timeout_seconds=timeout_seconds)
        proposal = SynthesisProposal(name, scored["verdict"],
                                     scored["detail"], scored["results"])
        if scored["verdict"] != "accepted":
            if self.ledger:
                self.ledger.append("synthesis", "SYNTHESIS_REJECTED", {
                    "name": name, "verdict": scored["verdict"],
                    "detail": scored["detail"]})
            return proposal
        tool = SynthesizedTool(name, description.strip(), code,
                               cleaned_cases, time.time())
        self.conn.execute(
            "INSERT INTO synthesized_tools(name,description,code,cases,ts)"
            " VALUES(?,?,?,?,?)",
            (tool.name, tool.description, tool.code,
             json.dumps(tool.cases), tool.ts))
        self.conn.commit()
        self._register_live(tool)
        if self.ledger:
            self.ledger.append("synthesis", "SYNTHESIS_ACCEPTED", {
                "name": name, "cases": len(cleaned_cases)})
        return proposal

    def remove(self, name: str) -> SynthesizedTool:
        """Delete a kept tool: its row and its live registration both
        go, so it cannot run again until re-proposed and re-tested."""
        rows = self.conn.execute(
            "SELECT name,description,code,cases,ts FROM synthesized_tools"
            " WHERE name=?", (name,)).fetchall()
        if not rows:
            raise KeyError(f"no synthesized tool {name!r}")
        r = rows[0]
        tool = SynthesizedTool(r[0], r[1], r[2], json.loads(r[3]), r[4])
        self.conn.execute("DELETE FROM synthesized_tools WHERE name=?", (name,))
        self.conn.commit()
        if self.tools is not None and hasattr(self.tools, "unregister"):
            self.tools.unregister(name)
        if self.ledger:
            self.ledger.append("synthesis", "SYNTHESIS_REMOVED", {"name": name})
        return tool


def load_synthesized_tools(core) -> list[str]:
    """Re-register every kept synthesized tool on a Core, from the
    table. Called at Core construction, after the built-in tools, so
    a kept tool survives a restart exactly as it was accepted."""
    names: list[str] = []
    try:
        rows = core.conn.execute(
            "SELECT name,description,code,cases,ts FROM synthesized_tools"
            " ORDER BY name ASC").fetchall()
    except Exception:
        # A core whose database predates synthesis has no table yet;
        # creating it here would write during construction, so the
        # engine creates it on first propose instead.
        return names
    for r in rows:
        tool = SynthesizedTool(r[0], r[1], r[2], json.loads(r[3]), r[4])
        core.tools.register(Tool(
            tool.name, tool.description, Risk.READ,
            make_synthesized_fn(tool.code), {}))
        names.append(tool.name)
    return names
