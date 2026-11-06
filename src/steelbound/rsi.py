"""Bounded recursive self-improvement, stages R0 + R1 + R2 + R3.

R0 (proposal engine) + R1 (sandbox evaluation), per docs/SELF_IMPROVEMENT.md.
Owner approval via ``decide(..., "approved")`` records the owner's
decision (ledgered ``RSI_ADOPTED``) but deliberately does NOT apply any
patch to the live repo or live data. R2 adds bounded auto-adoption on
top of that, without weakening the owner gate:

- Only a proposal with a ``config`` patch whose sandbox verdict is
  ``improvement`` (no regression on any suite, targeted metric up) may
  auto-adopt. The patch is applied to the live consolidator and
  persisted, so it survives a Core restart, and is ledgered
  ``RSI_AUTO_ADOPTED`` with the previous value recorded for rollback.
- Code patches (``replace``), descriptive proposals with no patch, and
  prompt proposals (which carry no patch to apply) stay owner-gated:
  ``auto_adopt()`` refuses them, and owner ``decide()`` still never
  applies anything by itself.
- Auto-adoption is capped per day (default 4, the same bound as the
  cycle cap) and blocked while the loop is paused after 3 consecutive
  rejections, so a rejected loop cannot adopt its way out of pause.
- Post-adoption regression triggers automatic rollback: ``rollback()``
  restores the previous value from the ``RSI_AUTO_ADOPTED`` ledger
  entry and ledgers ``RSI_ROLLBACK``; ``rollback_if_regressed()``
  runs the same checks as the scheduler's audit (evals all pass, chain
  verifies, every memory row has provenance) and rolls back every
  live auto-adoption, newest first, when any check fails.
  ``run_cycle()`` now auto-adopts eligible proposals itself; a
  proposal it cannot auto-adopt stays pending for the owner.

R3 adds scheduled recursion on top of that, without new bounds to
weaken: ``scheduled_cycle()`` is the body of the scheduler's built-in
``rsi_cycle`` job (6-hour default interval — four cycles a day is
exactly the daily cycle cap). It first runs the R2
``rollback_if_regressed()`` check, so a scheduled sweep repairs a
regressed auto-adoption before it proposes anything new, then runs
one ordinary ``run_cycle()`` with all its existing caps (daily cycle
cap, pause after consecutive rejections, no-regression sandbox rule,
owner gate for code). ``history()`` assembles the full recursion
history — every proposal with its sandbox score and status, plus
every ledgered cycle — from the proposal table and the ledger, so
the v1.0 target state (scheduled recursion with readable history)
is one CLI call away.

What the loop may never do, enforced here as well as in the design doc:

- It never touches the files that bind it. A ``replace`` patch whose file
  is ``ledger.py``, ``gate.py``, ``evals.py``, any file under ``tests/``,
  ``WORK_LEDGER.md``, ``docs/SELF_IMPROVEMENT.md``, or ``LICENSE`` is
  rejected with ValueError before anything is copied or run.
- It never lowers a risk class, approves its own external/destructive
  action, or weakens an eval: the sandbox verdict treats ANY eval
  regression, or any failing test run, as a regression, whatever the
  targeted improvement is.
- It never uses the network. Observation reads the local ledger, local
  memory, local scheduler state, and the local eval suites only.
  Proposals are deterministic heuristics over those measurements, not
  invented busywork: a cycle with no measurable weakness ends with no
  proposal (see the hard caps in docs/SELF_IMPROVEMENT.md).

The cycle, in code:

1. ``observe()`` — eval scores, ledger action counts, gate denials,
   failed scheduler runs, memory statistics. The weakest eval suite
   (lowest pass ratio) is the first improvement target.
2. ``propose()`` — one small, single-purpose proposal per measured
   weakness, stored in ``rsi_proposals`` and ledgered ``RSI_PROPOSAL``.
   Re-proposing an already-pending proposal for the same kind+target
   is a no-op, so cycles are idempotent.
3. ``evaluate()`` — sandbox scoring, ledgered ``RSI_EVALUATED``:
   - a ``config`` patch (today: consolidation thresholds) is measured
     in fresh temporary cores seeded with a snapshot of the live
     episodic items, baseline vs patched, so the live core is untouched;
   - a ``replace`` patch (one exact string replacement in one file) is
     applied to a temporary copy of the repo and the full test suite
     plus all five eval suites are run there in a subprocess, so the
     live repo is untouched;
   - a proposal with no patch is descriptive only: its report is the
     baseline eval scores, and its verdict is ``descriptive`` — it can
     never claim a measured improvement it did not earn.
   Verdicts: ``improvement`` (no regression + targeted metric up),
   ``no_improvement`` (no regression, nothing gained), ``regression``
   (any eval dropped or the sandbox test run failed), ``descriptive``
   (no patch to score), ``error`` (the sandbox itself could not run).
4. ``decide()`` — the owner approves or rejects; ledgered
   ``RSI_ADOPTED`` / ``RSI_REJECTED``. ``run_cycle()`` additionally
   enforces the daily cycle cap (default 4) and pauses after 3
   consecutive rejections, both read from the ledger itself.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from .consolidation import Consolidator


def _run_evals():
    # Imported lazily: evals imports config, and config imports this
    # module, so a top-level import here would be circular.
    from . import evals as evals_mod
    return evals_mod.run_all()

DEFAULT_MAX_CYCLES_PER_DAY = 4
DEFAULT_MAX_AUTO_ADOPTIONS_PER_DAY = 4
DEFAULT_MAX_CONSECUTIVE_REJECTIONS = 3
GATE_DENIAL_PROPOSAL_THRESHOLD = 3
BACKLOG_PROPOSAL_THRESHOLD = 3
CONFIG_THRESHOLD_FLOOR = 0.7
CONFIG_THRESHOLD_STEP = 0.05

# Files the loop may never patch, whatever the proposal claims. The
# ledger, the gate, and the evals are the rules that bind the loop;
# tests are excluded too, because patching a test to pass is not an
# improvement. WORK_LEDGER.md and the design doc are history, not code.
FORBIDDEN_PATCH_FILES = frozenset({
    "src/steelbound/ledger.py",
    "src/steelbound/gate.py",
    "src/steelbound/evals.py",
    "WORK_LEDGER.md",
    "docs/SELF_IMPROVEMENT.md",
    "docs/BUILD_QUEUE.md",
    "LICENSE",
})
FORBIDDEN_PATCH_PREFIXES = ("tests/",)

# Patch shapes the sandbox knows how to apply and score.
#   {"kind": "config", "setting": "consolidation_auto_threshold",
#    "value": 0.85}
#   {"kind": "replace", "file": "src/steelbound/tools.py",
#    "old": "...", "new": "..."}   (exactly one occurrence required)
CONFIG_SETTINGS = ("consolidation_auto_threshold", "consolidation_min_confidence")
_CHAT_PREFIXES = ("user:", "assistant:")


def _normalize(content: str) -> str:
    return " ".join(content.strip().lower().split())


def validate_patch(patch: dict | None) -> None:
    """Reject a patch that the loop must never apply. Raises ValueError."""
    if patch is None:
        return
    if not isinstance(patch, dict):
        raise ValueError("patch must be a JSON object or null")
    kind = patch.get("kind")
    if kind == "config":
        setting, value = patch.get("setting"), patch.get("value")
        if setting not in CONFIG_SETTINGS:
            raise ValueError(f"unsupported config setting {setting!r}")
        if not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
            raise ValueError(f"config value for {setting!r} must be in (0, 1]")
        return
    if kind == "replace":
        rel = patch.get("file", "")
        if not isinstance(rel, str) or not rel or os.path.isabs(rel) or ".." in rel.split("/"):
            raise ValueError(f"unsafe patch file {rel!r}")
        if rel in FORBIDDEN_PATCH_FILES or rel.startswith(FORBIDDEN_PATCH_PREFIXES):
            raise ValueError(f"patch file {rel!r} is outside the loop's reach")
        if not isinstance(patch.get("old"), str) or not patch["old"]:
            raise ValueError("replace patch needs a non-empty 'old' string")
        if not isinstance(patch.get("new"), str):
            raise ValueError("replace patch needs a 'new' string")
        return
    raise ValueError(f"unknown patch kind {kind!r}")


@dataclass(frozen=True)
class Observation:
    eval_scores: dict  # suite name -> {"passed": int, "total": int}
    failing_suites: tuple[str, ...]
    weakest_suite: str | None
    ledger_counts: dict  # action -> count
    gate_denials: int
    failed_scheduler_runs: int
    episodic_active: int
    semantic_active: int
    pending_proposals: int
    backlog: int  # episodic items consolidation would queue

    @property
    def all_evals_pass(self) -> bool:
        return not self.failing_suites


@dataclass(frozen=True)
class Proposal:
    id: int
    kind: str  # "config" | "code" | "prompt"
    target: str
    rationale: str
    patch: dict | None
    status: str  # pending | approved | rejected | auto_adopted | rolled_back
    ts: float
    score: dict | None = None


@dataclass(frozen=True)
class SandboxResult:
    baseline_evals: dict
    sandbox_evals: dict
    baseline_tests: dict | None  # {"ran": int, "ok": bool} for replace patches
    sandbox_tests: dict | None
    targeted_before: float | None
    targeted_after: float | None
    verdict: str  # improvement | no_improvement | regression | descriptive | error
    detail: str

    def to_dict(self) -> dict:
        return {
            "baseline_evals": self.baseline_evals,
            "sandbox_evals": self.sandbox_evals,
            "baseline_tests": self.baseline_tests,
            "sandbox_tests": self.sandbox_tests,
            "targeted_before": self.targeted_before,
            "targeted_after": self.targeted_after,
            "verdict": self.verdict,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CycleResult:
    status: str  # completed | capped | paused | no_proposal
    proposals: tuple[int, ...]  # newly created proposal ids
    evaluations: dict  # proposal id -> verdict
    detail: str
    auto_adopted: tuple[int, ...] = ()  # proposals this cycle auto-adopted


class RSIEngine:
    def __init__(self, ledger, conn: sqlite3.Connection, memory,
                 consolidation, scheduler=None, repo_root: str | None = None,
                 max_cycles_per_day: int = DEFAULT_MAX_CYCLES_PER_DAY,
                 max_consecutive_rejections: int = DEFAULT_MAX_CONSECUTIVE_REJECTIONS,
                 max_auto_adoptions_per_day: int = DEFAULT_MAX_AUTO_ADOPTIONS_PER_DAY):
        if max_cycles_per_day < 1:
            raise ValueError("max_cycles_per_day must be positive")
        if max_consecutive_rejections < 1:
            raise ValueError("max_consecutive_rejections must be positive")
        if max_auto_adoptions_per_day < 1:
            raise ValueError("max_auto_adoptions_per_day must be positive")
        self.ledger = ledger
        self.conn = conn
        self.memory = memory
        self.consolidation = consolidation
        self.scheduler = scheduler
        self.repo_root = repo_root
        self.max_cycles_per_day = max_cycles_per_day
        self.max_consecutive_rejections = max_consecutive_rejections
        self.max_auto_adoptions_per_day = max_auto_adoptions_per_day
        conn.execute("""CREATE TABLE IF NOT EXISTS rsi_proposals(
            id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
            target TEXT NOT NULL, rationale TEXT NOT NULL,
            patch TEXT, status TEXT NOT NULL DEFAULT 'pending',
            ts REAL NOT NULL, score TEXT)""")
        conn.commit()

    # -- proposal storage ---------------------------------------------
    def pending(self) -> list[Proposal]:
        return self.proposals("pending")

    def proposals(self, status: str | None = None) -> list[Proposal]:
        if status is None:
            rows = self.conn.execute(
                "SELECT id,kind,target,rationale,patch,status,ts,score "
                "FROM rsi_proposals ORDER BY id ASC").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id,kind,target,rationale,patch,status,ts,score "
                "FROM rsi_proposals WHERE status=? ORDER BY id ASC", (status,)).fetchall()
        out = []
        for r in rows:
            out.append(Proposal(r[0], r[1], r[2], r[3],
                                json.loads(r[4]) if r[4] else None,
                                r[5], r[6],
                                json.loads(r[7]) if r[7] else None))
        return out

    def _get(self, proposal_id: int) -> Proposal:
        rows = [p for p in self.proposals() if p.id == proposal_id]
        if not rows:
            raise KeyError(f"no RSI proposal {proposal_id}")
        return rows[0]

    def _insert(self, kind: str, target: str, rationale: str,
                patch: dict | None) -> Proposal:
        validate_patch(patch)
        cur = self.conn.execute(
            "INSERT INTO rsi_proposals(kind,target,rationale,patch,ts)"
            " VALUES(?,?,?,?,?)",
            (kind, target, rationale,
             json.dumps(patch) if patch is not None else None, time.time()))
        self.conn.commit()
        proposal = Proposal(cur.lastrowid, kind, target, rationale, patch,
                            "pending", time.time())
        if self.ledger:
            self.ledger.append("rsi", "RSI_PROPOSAL", {
                "proposal_id": proposal.id, "kind": kind, "target": target,
                "patch_kind": patch["kind"] if patch else None})
        return proposal

    def submit(self, kind: str, target: str, rationale: str,
               patch: dict | None = None) -> Proposal:
        """Owner/author-submitted proposal (e.g. a code patch to sandbox).

        Unlike propose(), this does not deduplicate: an explicitly
        submitted patch is a deliberate act, and its sandbox report is
        what the owner asked for. Forbidden patches are still rejected.
        """
        if kind not in ("config", "code", "prompt"):
            raise ValueError(f"unknown proposal kind {kind!r}")
        if not target or not rationale:
            raise ValueError("proposal needs a target and a rationale")
        return self._insert(kind, target, rationale, patch)

    # -- 1. observe ------------------------------------------------------
    def _backlog_items(self) -> list:
        """Active episodic items consolidation would queue, not promote."""
        semantic = {_normalize(i.content)
                    for i in self.memory.all_active(kind="semantic")}
        open_ids = {p.episodic_id for p in self.consolidation.proposals()
                    if p.status in ("pending", "promoted")}
        out = []
        for item in self.memory.all_active(kind="episodic"):
            if item.content.strip().lower().startswith(_CHAT_PREFIXES):
                continue
            if item.id in open_ids or _normalize(item.content) in semantic:
                continue
            if self.consolidation.min_confidence <= item.confidence < self.consolidation.auto_threshold:
                out.append(item)
        return out

    def observe(self) -> Observation:
        results = _run_evals()
        scores = {name: {"passed": p, "total": t}
                  for name, p, t, _ in results}
        failing = tuple(name for name, p, t, _ in results if p < t)
        weakest = None
        worst = None
        for name, p, t, _ in results:
            ratio = p / t if t else 1.0
            if worst is None or ratio < worst:
                worst, weakest = ratio, name
        counts: dict[str, int] = {}
        gate_denials = 0
        failed_scheduler = 0
        for entry in self.ledger.entries(10000):
            counts[entry.action] = counts.get(entry.action, 0) + 1
            if entry.action == "GATE_DECISION" and not entry.payload.get("allowed", True):
                gate_denials += 1
            if entry.action == "TOOL_DENIED":
                gate_denials += 1
            if entry.action == "SCHEDULER_RUN" and entry.payload.get("status") == "failed":
                failed_scheduler += 1
        backlog = self._backlog_items()
        return Observation(
            eval_scores=scores, failing_suites=failing, weakest_suite=weakest,
            ledger_counts=counts, gate_denials=gate_denials,
            failed_scheduler_runs=failed_scheduler,
            episodic_active=len(self.memory.all_active(kind="episodic")),
            semantic_active=len(self.memory.all_active(kind="semantic")),
            pending_proposals=len(self.consolidation.pending()),
            backlog=len(backlog))

    # -- 2. propose ------------------------------------------------------
    def propose(self, observation: Observation | None = None) -> list[Proposal]:
        """Generate one proposal per measured weakness; no weakness, no
        proposal. Re-proposing a pending kind+target is a no-op."""
        obs = observation or self.observe()
        pending_keys = {(p.kind, p.target) for p in self.pending()}
        candidates: list[tuple[str, str, str, dict | None]] = []
        for suite in obs.failing_suites:
            s = obs.eval_scores[suite]
            candidates.append(("code", f"eval:{suite}",
                               f"Eval suite {suite} scores {s['passed']}/{s['total']}. "
                               "Fix the measured failure; the sandbox must show "
                               "no regression on any suite and an improvement here.",
                               None))
        if obs.failed_scheduler_runs:
            candidates.append(("code", "scheduler",
                               f"{obs.failed_scheduler_runs} failed scheduler run(s) "
                               "are ledgered. Fix the failing job's root cause; "
                               "do not silence the audit.",
                               None))
        if obs.gate_denials >= GATE_DENIAL_PROPOSAL_THRESHOLD:
            candidates.append(("prompt", "gate_denials",
                               f"{obs.gate_denials} gate denials are ledgered. "
                               "Clarify the prompt/docs so legitimate local work "
                               "is requested in an approvable form; never lower "
                               "a risk class or widen permissions.",
                               None))
        if obs.backlog >= BACKLOG_PROPOSAL_THRESHOLD:
            # Only propose the step if it actually promotes something:
            # the items that sit just under the current threshold, in
            # [lowered, current), are the ones the step would promote.
            # Counting the whole queued range would propose steps that
            # measure as no_improvement in the sandbox.
            current = float(self.consolidation.auto_threshold)
            if current > CONFIG_THRESHOLD_FLOOR:
                lowered = max(CONFIG_THRESHOLD_FLOOR,
                              round(current - CONFIG_THRESHOLD_STEP, 2))
                near = [i for i in self._backlog_items()
                        if lowered <= i.confidence < current]
                if len(near) >= BACKLOG_PROPOSAL_THRESHOLD:
                    candidates.append(("config", "consolidation_backlog",
                                       f"{len(near)} episodic items sit just under "
                                       "the auto-promote threshold (confidence "
                                       f"in [{lowered}, {current})). Lower the "
                                       "auto-promote threshold from "
                                       f"{current} to {lowered} so they "
                                       "promote on the next consolidation run.",
                                       {"kind": "config",
                                        "setting": "consolidation_auto_threshold",
                                        "value": lowered}))
        created = []
        for kind, target, rationale, patch in candidates:
            if (kind, target) in pending_keys:
                continue
            created.append(self._insert(kind, target, rationale, patch))
            pending_keys.add((kind, target))
        return created

    # -- 3. sandbox evaluation -------------------------------------------
    def evaluate(self, proposal_id: int, repo_root: str | None = None,
                 seed_items: list[dict] | None = None) -> SandboxResult:
        proposal = self._get(proposal_id)
        if proposal.status != "pending":
            raise ValueError(f"proposal {proposal_id} is already {proposal.status}")
        validate_patch(proposal.patch)
        baseline = {name: {"passed": p, "total": t}
                    for name, p, t, _ in _run_evals()}
        if proposal.patch is None:
            result = SandboxResult(baseline, baseline, None, None, None, None,
                                   "descriptive",
                                   "No patch: descriptive proposal, scored against "
                                   "baseline evals only. Adoption is an owner decision.")
        elif proposal.patch["kind"] == "config":
            result = self._evaluate_config(proposal, baseline, seed_items)
        else:
            root = repo_root or self.repo_root
            if not root:
                raise ValueError("replace patch evaluation needs a repo root")
            result = self._evaluate_replace(proposal, baseline, root)
        self.conn.execute("UPDATE rsi_proposals SET score=? WHERE id=?",
                          (json.dumps(result.to_dict()), proposal_id))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("rsi", "RSI_EVALUATED", {
                "proposal_id": proposal_id, "kind": proposal.kind,
                "target": proposal.target, "verdict": result.verdict,
                "baseline_evals": baseline, "sandbox_evals": result.sandbox_evals})
        return result

    def _seeded_core(self, tmp: str, seed_items: list[dict]):
        """A throwaway core holding the given episodic items, nothing else."""
        from .config import Settings, Core
        core = Core(Settings(data_dir=tmp))
        for item in seed_items:
            core.memory.episode(item["content"], source=item.get("source", "rsi-seed"),
                                confidence=float(item.get("confidence", 0.8)))
        return core

    def _snapshot_seed(self) -> list[dict]:
        return [{"content": i.content, "source": i.source, "confidence": i.confidence}
                for i in self.memory.all_active(kind="episodic")]

    def _evaluate_config(self, proposal: Proposal, baseline: dict,
                         seed_items: list[dict] | None) -> SandboxResult:
        patch = proposal.patch or {}
        seed = seed_items if seed_items is not None else self._snapshot_seed()
        setting, value = patch["setting"], float(patch["value"])
        con = self.consolidation
        kwargs_base = {"auto_threshold": con.auto_threshold,
                       "min_confidence": con.min_confidence}
        kwargs_new = dict(kwargs_base)
        if setting == "consolidation_auto_threshold":
            kwargs_new["auto_threshold"] = value
        else:
            kwargs_new["min_confidence"] = value
        # Sanity: a patched threshold pair must stay ordered, or the
        # sandbox result is an error, not a silent pass.
        if not 0.0 <= kwargs_new["min_confidence"] <= kwargs_new["auto_threshold"] <= 1.0:
            return SandboxResult(baseline, baseline, None, None, None, None,
                                 "error",
                                 "patched thresholds violate "
                                 "min_confidence <= auto_threshold")
        with tempfile.TemporaryDirectory() as tmp:
            base_core = self._seeded_core(os.path.join(tmp, "base"), seed)
            base_promoted = len(base_core.consolidation.run().promoted)
            base_core.close()
            new_core = self._seeded_core(os.path.join(tmp, "new"), seed)
            new_core.consolidation = Consolidator(
                new_core.memory, new_core.ledger, **kwargs_new)
            new_promoted = len(new_core.consolidation.run().promoted)
            new_core.close()
        # Thresholds do not touch the eval suites, but the sandbox still
        # re-runs them so a "no regression" claim is measured, not assumed.
        sandbox = {name: {"passed": p, "total": t}
                   for name, p, t, _ in _run_evals()}
        regression = any(sandbox[n]["passed"] < baseline[n]["passed"]
                         for n in baseline)
        if regression:
            verdict, detail = "regression", "sandbox evals dropped below baseline"
        elif new_promoted > base_promoted:
            verdict = "improvement"
            detail = (f"promoted {base_promoted} -> {new_promoted} episodic "
                      f"items on the seeded snapshot, no eval regression")
        else:
            verdict, detail = "no_improvement", (
                f"promoted {base_promoted} -> {new_promoted} on the seeded "
                "snapshot, no eval regression")
        return SandboxResult(baseline, sandbox, None, None,
                             float(base_promoted), float(new_promoted),
                             verdict, detail)

    def _run_subprocess(self, cmd: list[str], cwd: str, env: dict) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                              text=True, timeout=600)

    def _evaluate_replace(self, proposal: Proposal, baseline: dict,
                          repo_root: str) -> SandboxResult:
        patch = proposal.patch or {}
        rel, old, new = patch["file"], patch["old"], patch["new"]
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = os.path.join(tmp, "repo")
            shutil.copytree(repo_root, sandbox, ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", ".steelbound"))
            target = os.path.join(sandbox, rel)
            if not os.path.isfile(target):
                return SandboxResult(baseline, baseline, None, None, None, None,
                                     "error", f"patch file {rel!r} not found in repo copy")
            with open(target, encoding="utf-8") as f:
                text = f.read()
            if text.count(old) != 1:
                return SandboxResult(baseline, baseline, None, None, None, None,
                                     "error",
                                     f"patch 'old' string occurs {text.count(old)} "
                                     "times in the repo copy; exactly one required")
            with open(target, "w", encoding="utf-8") as f:
                f.write(text.replace(old, new))
            env = dict(os.environ)
            env["PYTHONPATH"] = os.path.join(sandbox, "src")
            # The sandbox runs the full suite, including this module's
            # own tests; those tests must not spawn a second sandbox
            # inside the first one (see tests/test_rsi.py).
            env["STEELBOUND_RSI_SANDBOX"] = "1"
            try:
                tests = self._run_subprocess(
                    [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                    sandbox, env)
                test_out = tests.stdout + tests.stderr
                m = re.search(r"Ran (\d+) tests", test_out)
                sandbox_tests = {"ran": int(m.group(1)) if m else 0,
                                 "ok": "OK" in test_out.splitlines()[-1]
                                 if test_out.splitlines() else False}
                data_dir = os.path.join(tmp, "data")
                ev = self._run_subprocess(
                    [sys.executable, "-m", "steelbound", "--data-dir", data_dir, "evals"],
                    sandbox, env)
                sandbox_evals = {}
                for line in (ev.stdout + ev.stderr).splitlines():
                    mm = re.match(r"(\w+): (\d+)/(\d+)", line)
                    if mm:
                        sandbox_evals[mm.group(1)] = {
                            "passed": int(mm.group(2)), "total": int(mm.group(3))}
                # Note: the eval CLI exits 1 when a suite fails — that
                # is a scored result, not a run failure. Only missing
                # output is an error.
                if not sandbox_evals:
                    return SandboxResult(baseline, baseline,
                                         None, sandbox_tests, None, None,
                                         "error",
                                         "sandbox eval run produced no output: "
                                         + (ev.stderr.strip()[-300:] if ev.stderr.strip() else "empty"))
            except subprocess.TimeoutExpired:
                return SandboxResult(baseline, baseline, None, None, None, None,
                                     "error", "sandbox run timed out")
        regression = (not sandbox_tests["ok"]) or any(
            sandbox_evals.get(n, {"passed": 0})["passed"] < baseline[n]["passed"]
            for n in baseline)
        target_suite = None
        if proposal.target.startswith("eval:"):
            target_suite = proposal.target[len("eval:"):]
        if regression:
            verdict, detail = "regression", (
                f"sandbox tests ok={sandbox_tests['ok']}; "
                f"sandbox evals {sandbox_evals} vs baseline {baseline}")
            t_before = t_after = None
        elif (target_suite and target_suite in baseline
              and sandbox_evals.get(target_suite, {}).get("passed", 0)
              > baseline[target_suite]["passed"]):
            verdict = "improvement"
            detail = f"eval suite {target_suite} improved in the sandbox, no regression"
            t_before = float(baseline[target_suite]["passed"])
            t_after = float(sandbox_evals[target_suite]["passed"])
        else:
            verdict, detail = "no_improvement", (
                "sandbox tests pass and no eval regressed, but the targeted "
                "metric did not improve")
            t_before = t_after = None
        return SandboxResult(baseline, sandbox_evals, None, sandbox_tests,
                             t_before, t_after, verdict, detail)

    # -- 4. owner gate -----------------------------------------------------
    def decide(self, proposal_id: int, decision: str) -> Proposal:
        """Owner decision. Records adoption/rejection; it never applies
        the patch — applying a config patch is the separate R2
        ``auto_adopt()`` step (or an owner config change), and applying
        a code patch is an owner git commit, so approval here cannot
        silently change the core."""
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be 'approved' or 'rejected'")
        proposal = self._get(proposal_id)
        if proposal.status != "pending":
            raise ValueError(f"proposal {proposal_id} is already {proposal.status}")
        new_status = "approved" if decision == "approved" else "rejected"
        self.conn.execute("UPDATE rsi_proposals SET status=? WHERE id=?",
                          (new_status, proposal_id))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("rsi",
                               "RSI_ADOPTED" if decision == "approved" else "RSI_REJECTED",
                               {"proposal_id": proposal_id, "kind": proposal.kind,
                                "target": proposal.target})
        return Proposal(proposal.id, proposal.kind, proposal.target,
                        proposal.rationale, proposal.patch, new_status,
                        proposal.ts, proposal.score)

    # -- 5. R2 bounded auto-adoption ---------------------------------------
    def _entries_today(self, action: str) -> int:
        day_start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        return sum(1 for e in self.ledger.entries(10000)
                   if e.action == action and e.ts >= day_start)

    def _auto_adoptions_today(self) -> int:
        return self._entries_today("RSI_AUTO_ADOPTED")

    def _adoption_record(self, proposal_id: int) -> dict | None:
        """The RSI_AUTO_ADOPTED payload for a proposal, or None. The
        ledger is the record of the previous value, so rollback never
        depends on a side table that could drift from the ledger."""
        for entry in reversed(self.ledger.entries(10000)):
            if (entry.action == "RSI_AUTO_ADOPTED"
                    and entry.payload.get("proposal_id") == proposal_id):
                return entry.payload
        return None

    def _already_rolled_back(self, proposal_id: int) -> bool:
        return any(entry.action == "RSI_ROLLBACK"
                   and entry.payload.get("proposal_id") == proposal_id
                   for entry in self.ledger.entries(10000))

    def auto_adopt(self, proposal_id: int) -> Proposal:
        """R2: apply a config patch without owner approval, but only
        when every bound holds. Anything else raises ValueError and
        the proposal stays pending (or keeps its existing status):
        - the proposal must be pending;
        - it must carry a ``config`` patch — code (``replace``) patches
          and patch-less prompt/descriptive proposals stay owner-gated;
        - it must have been sandbox-evaluated with verdict
          ``improvement`` (all-green, no regression, target metric up);
        - the daily auto-adoption cap must not be reached, and the loop
          must not be paused after consecutive rejections.
        On success the new value is applied to the live consolidator
        AND persisted (see Consolidator.set_thresholds), the proposal
        becomes ``auto_adopted``, and the previous value is recorded
        in the RSI_AUTO_ADOPTED ledger entry for rollback."""
        proposal = self._get(proposal_id)
        if proposal.status != "pending":
            raise ValueError(f"proposal {proposal_id} is already {proposal.status}")
        if proposal.patch is None:
            raise ValueError(
                f"proposal {proposal_id} has no patch to apply; "
                "descriptive and prompt proposals stay owner-gated")
        if proposal.patch.get("kind") != "config":
            raise ValueError(
                f"proposal {proposal_id} is a {proposal.patch.get('kind')!r} patch; "
                "only config patches may auto-adopt — code stays owner-gated")
        validate_patch(proposal.patch)
        if self._consecutive_rejections() >= self.max_consecutive_rejections:
            raise ValueError(
                f"{self.max_consecutive_rejections} consecutive rejections: "
                "loop paused for the owner; auto-adoption refused")
        if self._auto_adoptions_today() >= self.max_auto_adoptions_per_day:
            raise ValueError(
                f"daily auto-adoption cap ({self.max_auto_adoptions_per_day}) reached")
        score = proposal.score or {}
        if score.get("verdict") != "improvement":
            raise ValueError(
                f"proposal {proposal_id} has no 'improvement' sandbox verdict "
                f"(verdict: {score.get('verdict')!r}); evaluate it first and "
                "only all-green improvements may auto-adopt")
        if not hasattr(self.consolidation, "set_thresholds"):
            raise ValueError("consolidator cannot persist threshold changes")
        setting = proposal.patch["setting"]
        value = float(proposal.patch["value"])
        # Patch setting names differ from consolidator attributes:
        # consolidation_auto_threshold -> auto_threshold.
        previous = float(self.consolidation.auto_threshold
                         if setting == "consolidation_auto_threshold"
                         else self.consolidation.min_confidence)
        if setting == "consolidation_auto_threshold":
            self.consolidation.set_thresholds(auto_threshold=value)
        else:
            self.consolidation.set_thresholds(min_confidence=value)
        self.conn.execute("UPDATE rsi_proposals SET status='auto_adopted' WHERE id=?",
                          (proposal_id,))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("rsi", "RSI_AUTO_ADOPTED", {
                "proposal_id": proposal_id, "kind": proposal.kind,
                "target": proposal.target, "setting": setting,
                "previous_value": previous, "new_value": value,
                "verdict": "improvement"})
        return Proposal(proposal.id, proposal.kind, proposal.target,
                        proposal.rationale, proposal.patch, "auto_adopted",
                        proposal.ts, proposal.score)

    def rollback(self, proposal_id: int, reason: str = "manual") -> Proposal:
        """Restore the value an auto-adoption changed, from the
        RSI_AUTO_ADOPTED ledger entry, and ledger RSI_ROLLBACK.
        Only an ``auto_adopted`` proposal can be rolled back, and only
        once; anything else raises ValueError (fail closed)."""
        proposal = self._get(proposal_id)
        if proposal.status != "auto_adopted":
            raise ValueError(
                f"proposal {proposal_id} is {proposal.status}, not auto_adopted")
        if self._already_rolled_back(proposal_id):
            raise ValueError(f"proposal {proposal_id} was already rolled back")
        record = self._adoption_record(proposal_id)
        if record is None:
            raise ValueError(
                f"no RSI_AUTO_ADOPTED record for proposal {proposal_id}; "
                "cannot roll back without a recorded previous value")
        setting, previous = record["setting"], float(record["previous_value"])
        if setting == "consolidation_auto_threshold":
            self.consolidation.set_thresholds(auto_threshold=previous)
        else:
            self.consolidation.set_thresholds(min_confidence=previous)
        self.conn.execute("UPDATE rsi_proposals SET status='rolled_back' WHERE id=?",
                          (proposal_id,))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("rsi", "RSI_ROLLBACK", {
                "proposal_id": proposal_id, "setting": setting,
                "restored_value": previous,
                "adopted_value": record.get("new_value"), "reason": reason})
        return Proposal(proposal.id, proposal.kind, proposal.target,
                        proposal.rationale, proposal.patch, "rolled_back",
                        proposal.ts, proposal.score)

    def _regression_detail(self) -> dict | None:
        """The scheduler audit's checks, applied to post-adoption
        state: every eval suite must pass, the ledger chain must
        verify, and every memory row must carry provenance. Returns
        None when healthy, else a diagnostic dict."""
        results = _run_evals()
        failing = [name for name, p, t, _ in results if p < t]
        valid = self.ledger.verify() if self.ledger else False
        missing = 0
        try:
            rows = self.conn.execute("SELECT source FROM memory_items").fetchall()
            missing = sum(1 for (s,) in rows if s is None or not str(s).strip())
        except sqlite3.Error:
            missing = 0
        if not failing and valid and not missing:
            return None
        return {"failing_suites": failing, "chain_valid": valid,
                "missing_provenance": missing}

    def rollback_if_regressed(self) -> list[Proposal]:
        """Automatic rollback (SELF_IMPROVEMENT.md step 6): if the
        audit checks fail while any auto-adoption is live, roll every
        live auto-adoption back, newest first, and return them. When
        healthy, or when nothing is adopted, returns [] and changes
        nothing — a healthy core is never rolled back on speculation."""
        live = [p for p in self.proposals("auto_adopted")]
        if not live:
            return []
        detail = self._regression_detail()
        if detail is None:
            return []
        reason = f"post-adoption regression: {detail}"
        rolled = []
        for proposal in reversed(live):
            rolled.append(self.rollback(proposal.id, reason=reason))
        return rolled

    # -- full cycle, with the hard caps ------------------------------------
    def _cycles_today(self) -> int:
        return self._entries_today("RSI_CYCLE")

    def _consecutive_rejections(self) -> int:
        count = 0
        for entry in reversed(self.ledger.entries(10000)):
            if entry.action == "RSI_REJECTED":
                count += 1
            elif entry.action in ("RSI_ADOPTED", "RSI_AUTO_ADOPTED"):
                break
        return count

    def run_cycle(self, repo_root: str | None = None) -> CycleResult:
        if self._cycles_today() >= self.max_cycles_per_day:
            return CycleResult("capped", (), {},
                               f"daily cycle cap ({self.max_cycles_per_day}) reached")
        if self._consecutive_rejections() >= self.max_consecutive_rejections:
            return CycleResult("paused", (), {},
                               f"{self.max_consecutive_rejections} consecutive "
                               "rejections: paused for the owner")
        created = self.propose()
        evaluations: dict[int, str] = {}
        auto_adopted: list[int] = []
        for proposal in created:
            if proposal.patch and proposal.patch.get("kind") == "replace" \
                    and not (repo_root or self.repo_root):
                continue  # replace patches need a repo root to sandbox
            result = self.evaluate(proposal.id, repo_root=repo_root)
            evaluations[proposal.id] = result.verdict
            # R2: an all-green config improvement adopts itself here;
            # anything else (code, descriptive, no_improvement, or a
            # reached cap) stays pending for the owner, and a failed
            # auto-adoption must never fail the whole cycle.
            if result.verdict == "improvement":
                try:
                    self.auto_adopt(proposal.id)
                    auto_adopted.append(proposal.id)
                except ValueError:
                    pass
        status = "completed" if created else "no_proposal"
        detail = (f"{len(created)} proposal(s), "
                  f"{len(evaluations)} evaluated, "
                  f"{len(auto_adopted)} auto-adopted" if created
                  else "no measurable weakness found; no proposal made")
        if self.ledger:
            self.ledger.append("rsi", "RSI_CYCLE", {
                "status": status, "proposals": [p.id for p in created],
                "evaluations": evaluations, "auto_adopted": auto_adopted})
        return CycleResult(status, tuple(p.id for p in created), evaluations,
                           detail, tuple(auto_adopted))

    # -- R3: scheduled recursion ------------------------------------------
    def scheduled_cycle(self, repo_root: str | None = None) -> dict:
        """One scheduled recursion sweep: the body of the scheduler's
        built-in ``rsi_cycle`` job.

        Rollback-first: a regressed live auto-adoption is repaired
        before anything new is proposed. Then one ordinary
        ``run_cycle()`` — same caps, same sandbox rule, same owner
        gate — so the schedule adds timing, not new permissions.
        Returns a JSON-able detail dict for the SCHEDULER_RUN entry.
        ``generation`` is the count of ledgered RSI_CYCLE entries
        including this one when it ran (a capped or paused sweep
        ledgers no cycle, so its generation is the count so far)."""
        rolled = self.rollback_if_regressed()
        cycle = self.run_cycle(repo_root=repo_root or self.repo_root)
        generation = sum(1 for e in self.ledger.entries(10000)
                         if e.action == "RSI_CYCLE")
        return {"generation": generation, "cycle_status": cycle.status,
                "proposals": list(cycle.proposals),
                "evaluations": {str(k): v for k, v in cycle.evaluations.items()},
                "auto_adopted": list(cycle.auto_adopted),
                "rolled_back": [p.id for p in rolled],
                "detail": cycle.detail}

    def history(self) -> dict:
        """The full recursion history, readable in one place.

        Every proposal (with its sandbox score and current status),
        each paired with the ledger events that mention it
        (RSI_PROPOSAL, RSI_EVALUATED, RSI_ADOPTED, RSI_REJECTED,
        RSI_AUTO_ADOPTED, RSI_ROLLBACK), plus every ledgered cycle.
        The ledger itself remains the authoritative record; this is
        just the joined view of it."""
        events: dict[int, list[dict]] = {}
        cycles: list[dict] = []
        for entry in self.ledger.entries(10000):
            if entry.action == "RSI_CYCLE":
                cycles.append({"ts": entry.ts, "payload": entry.payload})
            elif (entry.action.startswith("RSI_")
                    and isinstance(entry.payload.get("proposal_id"), int)):
                events.setdefault(entry.payload["proposal_id"], []).append(
                    {"action": entry.action, "ts": entry.ts,
                     "payload": entry.payload})
        proposals = []
        for p in self.proposals():
            proposals.append({"proposal_id": p.id, "kind": p.kind,
                              "target": p.target, "rationale": p.rationale,
                              "patch": p.patch, "status": p.status,
                              "score": p.score, "events": events.get(p.id, [])})
        return {"proposals": proposals, "cycles": cycles}
