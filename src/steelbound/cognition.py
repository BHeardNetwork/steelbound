"""Cognition: the core's verifiable account of itself.

Implements docs/COGNITION.md sections 1-3 (the daily episode is a
separate build-queue item and lives elsewhere):

1. **Six-question self-query.** Before any significant action, the
   core answers who / what / why / where / how / when from its own
   ledger, memory, and eval records — never invented. The six answers
   are ledgered as one ``SELF_QUERY`` entry. An action with no
   answerable "why" is not taken: the query is still ledgered, with
   ``why`` null and ``blocked`` true, so the gap itself is on the
   record and any caller can refuse to proceed by checking
   ``SelfQuery.blocked``.
2. **Metacognition.** Per-capability eval scores, confidence
   calibration (stated confidence of semantic items vs how often
   high-confidence items later needed correction), and known
   unknowns — questions asked that could not be grounded in memory,
   ledger, or tools, recorded explicitly and resolved explicitly
   against a memory item.
3. **Reasoning / self-dialogue.** Significant decisions produce a
   deliberation record — options, evidence, chosen option, deciding
   reason — stored in its own table and ledgered as ``DELIBERATION``.
   For external/destructive actions the approval path goes through
   ``request_approval``: with no deliberation record for that action
   and risk, the request is refused *before* it reaches the gate, so
   approval can never be granted on an undeliberated action. The
   gate itself is unchanged — it still decides, and still fails
   closed — this layer only gates access to it.

Nothing here invents facts: every answer is derived from ledger
entries, memory rows, eval results, or caller-supplied arguments
that are stored verbatim. The layer binds itself to the same
invariants as everything else: the ledger stays append-only, the
gate's risk classes are never lowered, and no network is used.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

from . import __version__
from .gate import GateDecision, Risk


def _normalize(content: str) -> str:
    return " ".join(content.strip().lower().split())


def _risk_value(risk) -> str:
    if isinstance(risk, Risk):
        return risk.value
    value = str(risk)
    if value in {r.value for r in Risk}:
        return value
    raise ValueError(f"unknown risk class {risk!r}")


def _risk_enum(risk) -> Risk:
    if isinstance(risk, Risk):
        return risk
    for r in Risk:
        if r.value == str(risk):
            return r
    raise ValueError(f"unknown risk class {risk!r}")


@dataclass(frozen=True)
class SelfQuery:
    """The six answers for one action, plus whether it may proceed."""
    entry_id: int
    who: dict
    what: str
    why: str | None
    where: str
    how: dict
    when: dict
    blocked: bool  # True exactly when why is unanswerable


@dataclass(frozen=True)
class Deliberation:
    id: int
    action: str
    risk: str
    options: tuple[str, ...]
    evidence: dict
    chosen: str
    reason: str
    ts: float


@dataclass(frozen=True)
class Unknown:
    id: int
    question: str
    context: str
    status: str  # open | resolved
    ts: float
    resolution_id: int | None = None


@dataclass(frozen=True)
class MetacognitionReport:
    eval_scores: dict
    failing_suites: tuple[str, ...]
    semantic_count: int
    mean_semantic_confidence: float | None
    corrected_semantic: int
    overconfident_corrections: int
    calibration_gap: float | None
    gate_denials: int
    open_unknowns: int
    assessment: str


@dataclass(frozen=True)
class ComprehensionMismatch:
    episodic_id: int
    semantic_id: int
    detail: str


@dataclass(frozen=True)
class ComprehensionReport:
    checked: int
    mismatches: tuple[ComprehensionMismatch, ...]


class CognitionEngine:
    """Self-query, metacognition, and deliberation records for a Core."""

    def __init__(self, ledger, conn: sqlite3.Connection, memory=None,
                 gate=None, provider_name: str = "offline",
                 data_dir: str = ""):
        self.ledger = ledger
        self.conn = conn
        self.memory = memory
        self.gate = gate
        self.provider_name = provider_name
        self.data_dir = data_dir
        conn.execute("""CREATE TABLE IF NOT EXISTS cognition_deliberations(
            id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL,
            risk TEXT NOT NULL, options TEXT NOT NULL,
            evidence TEXT NOT NULL, chosen TEXT NOT NULL,
            reason TEXT NOT NULL, ts REAL NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS cognition_unknowns(
            id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'open',
            ts REAL NOT NULL, resolution_id INTEGER)""")
        conn.commit()

    # -- identity --------------------------------------------------------
    def _own_generation(self) -> int:
        """This core's generation: 0 for a root core, else the
        generation recorded in its own OFFSPRING_GENESIS entry — the
        same rule the offspring engine uses."""
        if self.ledger is None:
            return 0
        for entry in self.ledger.entries(10000):
            if entry.action == "OFFSPRING_GENESIS":
                return int(entry.payload.get("generation", 0))
        return 0

    def identity(self, authority: str = "owner") -> dict:
        generation = self._own_generation()
        return {
            "name": "Steelbound",
            "version": __version__,
            "provider": self.provider_name,
            "generation": generation,
            "lineage": "root" if generation == 0 else f"offspring generation {generation}",
            "authority": authority,
        }

    # -- 1. six-question self-query --------------------------------------
    def self_query(self, action: str, goal: str | None = None,
                   where: str | None = None, risk=Risk.READ,
                   approved: bool = False, authority: str = "owner",
                   schedule: str | None = None) -> SelfQuery:
        """Answer who/what/why/where/how/when for ``action`` and ledger
        the answers as SELF_QUERY. ``goal`` traces the why to a goal,
        queue item, or owner instruction; without one the query is
        blocked and the gap is ledgered, not hidden."""
        if not action or not str(action).strip():
            raise ValueError("self_query requires a non-empty action")
        if goal is not None and not str(goal).strip():
            goal = None
        risk_value = _risk_value(risk)
        now = time.time()
        who = self.identity(authority=authority)
        what = str(action).strip()
        why = str(goal).strip() if goal is not None else None
        where_answer = str(where) if where else f"local: {self.data_dir or 'this machine'}"
        how = {"risk": risk_value, "approved": bool(approved)}
        when = {"at": now, "schedule": schedule}
        blocked = why is None
        payload = {"who": who, "what": what, "why": why, "where": where_answer,
                   "how": how, "when": when, "blocked": blocked}
        entry_id = None
        if self.ledger:
            entry_id = self.ledger.append("cognition", "SELF_QUERY", payload).id
        return SelfQuery(entry_id, who, what, why, where_answer, how, when, blocked)

    def scheduled_self_query(self) -> dict:
        """Zero-argument self-check suitable for a scheduler custom
        job: it answers the six questions for the scheduled check
        itself, so a core on a schedule keeps an auditable account
        of itself without any new built-in job."""
        query = self.self_query(
            "scheduled self-check",
            goal="Maintain an auditable account of identity, state, and "
                 "capability, per docs/COGNITION.md",
            authority="scheduler",
            schedule="scheduler custom job")
        return {"self_query_id": query.entry_id, "blocked": query.blocked,
                "generation": query.who["generation"],
                "provider": query.who["provider"]}

    # -- 3. deliberation / self-dialogue ----------------------------------
    def deliberate(self, action: str, options, evidence: dict,
                   chosen: str, reason: str, risk=Risk.EXTERNAL) -> Deliberation:
        """Record a deliberation: the options considered, the evidence
        for each, the chosen option, and the deciding reason. The
        record is stored in full and ledgered as DELIBERATION; the
        ledger payload carries the decision, the table carries the
        whole dialogue."""
        if not action or not str(action).strip():
            raise ValueError("deliberate requires a non-empty action")
        options = tuple(str(o) for o in (options or []))
        if not options:
            raise ValueError("deliberate requires at least one option")
        if chosen not in options:
            raise ValueError(f"chosen option {chosen!r} is not among the options")
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be a dict mapping option to evidence")
        if not reason or not str(reason).strip():
            raise ValueError("deliberate requires a non-empty deciding reason")
        risk_value = _risk_value(risk)
        ts = time.time()
        cur = self.conn.execute(
            "INSERT INTO cognition_deliberations(action,risk,options,evidence,chosen,reason,ts)"
            " VALUES(?,?,?,?,?,?,?)",
            (str(action), risk_value, json.dumps(list(options)),
             json.dumps(evidence), str(chosen), str(reason), ts))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("cognition", "DELIBERATION", {
                "deliberation_id": cur.lastrowid, "action": str(action),
                "risk": risk_value, "chosen": str(chosen),
                "reason": str(reason)})
        return Deliberation(cur.lastrowid, str(action), risk_value, options,
                            dict(evidence), str(chosen), str(reason), ts)

    def deliberations(self, action: str | None = None) -> list[Deliberation]:
        if action is None:
            rows = self.conn.execute(
                "SELECT id,action,risk,options,evidence,chosen,reason,ts"
                " FROM cognition_deliberations ORDER BY id ASC").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id,action,risk,options,evidence,chosen,reason,ts"
                " FROM cognition_deliberations WHERE action=? ORDER BY id ASC",
                (action,)).fetchall()
        return [Deliberation(r[0], r[1], r[2], tuple(json.loads(r[3])),
                             json.loads(r[4]), r[5], r[6], r[7]) for r in rows]

    def _deliberation_for(self, action: str, risk_value: str):
        row = self.conn.execute(
            "SELECT id FROM cognition_deliberations WHERE action=? AND risk=?"
            " ORDER BY id DESC LIMIT 1", (action, risk_value)).fetchone()
        return row[0] if row else None

    def request_approval(self, action: str, risk, deliberation_id: int | None = None,
                         approved: bool = False) -> GateDecision:
        """The approval path for external/destructive actions: no
        deliberation record, no approval request. Without a matching
        record the request is refused before the gate is consulted,
        and the refusal is ledgered as DELIBERATION_REQUIRED; with
        one, the gate decides exactly as it always has (so an
        unapproved request is still denied, and FORBIDDEN stays
        forbidden). Non-external risks do not need a deliberation
        and go straight to the gate."""
        risk_enum = _risk_enum(risk)
        if risk_enum in (Risk.EXTERNAL, Risk.DESTRUCTIVE):
            record_id = None
            if deliberation_id is not None:
                row = self.conn.execute(
                    "SELECT action,risk FROM cognition_deliberations WHERE id=?",
                    (deliberation_id,)).fetchone()
                if row and row[0] == action and row[1] == risk_enum.value:
                    record_id = deliberation_id
            if record_id is None:
                if self.ledger:
                    self.ledger.append("cognition", "DELIBERATION_REQUIRED", {
                        "action": action, "risk": risk_enum.value,
                        "deliberation_id": deliberation_id})
                return GateDecision(
                    False, risk_enum,
                    "No deliberation record for this action and risk: "
                    "approval request refused before the gate.", True)
        if self.gate is None:
            raise RuntimeError("cognition engine has no gate to decide with")
        return self.gate.evaluate(action, risk_enum, approved=approved)

    # -- 2. metacognition: known unknowns ----------------------------------
    def record_unknown(self, question: str, context: str = "") -> Unknown:
        """Record a question that could not be grounded in memory,
        ledger, or tools. The gap is named, not papered over."""
        if not question or not str(question).strip():
            raise ValueError("record_unknown requires a non-empty question")
        ts = time.time()
        cur = self.conn.execute(
            "INSERT INTO cognition_unknowns(question,context,status,ts)"
            " VALUES(?,?, 'open', ?)", (str(question).strip(), str(context), ts))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("cognition", "UNKNOWN_RECORDED", {
                "unknown_id": cur.lastrowid, "question": str(question).strip()})
        return Unknown(cur.lastrowid, str(question).strip(), str(context),
                       "open", ts)

    def unknowns(self, status: str | None = None) -> list[Unknown]:
        if status is None:
            rows = self.conn.execute(
                "SELECT id,question,context,status,ts,resolution_id"
                " FROM cognition_unknowns ORDER BY id ASC").fetchall()
        else:
            if status not in ("open", "resolved"):
                raise ValueError(f"unknown status {status!r}")
            rows = self.conn.execute(
                "SELECT id,question,context,status,ts,resolution_id"
                " FROM cognition_unknowns WHERE status=? ORDER BY id ASC",
                (status,)).fetchall()
        return [Unknown(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]

    def resolve_unknown(self, unknown_id: int, resolution_memory_id: int) -> Unknown:
        """Resolve an open unknown against a memory item that grounds
        it. The memory item must exist; resolving twice is refused."""
        row = self.conn.execute(
            "SELECT question,context,status,ts FROM cognition_unknowns WHERE id=?",
            (unknown_id,)).fetchone()
        if row is None:
            raise KeyError(f"no unknown {unknown_id}")
        if row[2] != "open":
            raise ValueError(f"unknown {unknown_id} is already {row[2]}")
        mem = self.conn.execute(
            "SELECT 1 FROM memory_items WHERE id=?", (resolution_memory_id,)).fetchone()
        if mem is None:
            raise KeyError(f"no memory item {resolution_memory_id}")
        self.conn.execute(
            "UPDATE cognition_unknowns SET status='resolved', resolution_id=?"
            " WHERE id=?", (resolution_memory_id, unknown_id))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("cognition", "UNKNOWN_RESOLVED", {
                "unknown_id": unknown_id,
                "resolution_memory_id": resolution_memory_id})
        return Unknown(unknown_id, row[0], row[1], "resolved", row[3],
                       resolution_memory_id)

    # -- 2. metacognition: capability + calibration report -----------------
    def metacognition(self) -> MetacognitionReport:
        """What the core knows about its own knowing: per-capability
        eval scores (measured, by running the suites), confidence
        calibration over semantic memory, gate denials, and open
        unknowns. A read, not a ledger entry — the underlying records
        are already ledgered; this derives the view from them."""
        from . import evals as _evals  # lazy: evals imports config
        results = _evals.run_all()
        scores = {name: {"passed": p, "total": t} for name, p, t, _ in results}
        failing = tuple(name for name, p, t, _ in results if p < t)
        semantic_rows = self.conn.execute(
            "SELECT confidence, superseded_by FROM memory_items"
            " WHERE kind='semantic'").fetchall()
        confidences = [float(r[0]) for r in semantic_rows]
        corrected = [float(r[0]) for r in semantic_rows if r[1] is not None]
        overconfident = sum(1 for c in corrected if c >= 0.9)
        gap = (overconfident / len(corrected)) if corrected else None
        gate_denials = 0
        if self.ledger:
            for entry in self.ledger.entries(10000):
                if entry.action == "GATE_DECISION" and not entry.payload.get("allowed", True):
                    gate_denials += 1
                elif entry.action == "TOOL_DENIED":
                    gate_denials += 1
        open_unknowns = len(self.unknowns(status="open"))
        parts = []
        if failing:
            parts.append(f"{len(failing)} eval suite(s) failing: {', '.join(failing)}")
        else:
            parts.append("all eval suites pass")
        if gap is None:
            parts.append("no corrected semantic items to calibrate against")
        elif gap > 0:
            parts.append(f"calibration gap: {overconfident}/{len(corrected)} "
                         "corrected items were stated at confidence >= 0.9")
        else:
            parts.append("no high-confidence item has needed correction")
        if open_unknowns:
            parts.append(f"{open_unknowns} known unknown(s) still open")
        return MetacognitionReport(
            eval_scores=scores, failing_suites=failing,
            semantic_count=len(semantic_rows),
            mean_semantic_confidence=(sum(confidences) / len(confidences)
                                       if confidences else None),
            corrected_semantic=len(corrected),
            overconfident_corrections=overconfident,
            calibration_gap=gap, gate_denials=gate_denials,
            open_unknowns=open_unknowns, assessment="; ".join(parts))

    # -- 2. comprehension checks --------------------------------------------
    def comprehension_check(self) -> ComprehensionReport:
        """Re-derive every promoted fact from its source: for each
        still-active semantic item carrying a promoted_from_episodic
        tag, compare its content (normalized) with the episodic item
        it was promoted from. Every mismatch is ledgered as
        COMPREHENSION_MISMATCH — and therefore visible to the next
        RSI observation, which reads the ledger — and returned in
        the report. Superseded semantic items are skipped: a later
        correction is a new fact, not a comprehension failure."""
        rows = self.conn.execute(
            "SELECT id,content,tags,superseded_by FROM memory_items"
            " WHERE kind='semantic'").fetchall()
        checked = 0
        mismatches: list[ComprehensionMismatch] = []
        for semantic_id, content, tags_json, superseded_by in rows:
            if superseded_by is not None:
                continue
            episodic_id = None
            for tag in json.loads(tags_json or "[]"):
                if isinstance(tag, str) and tag.startswith("promoted_from_episodic:"):
                    try:
                        episodic_id = int(tag.split(":", 1)[1])
                    except ValueError:
                        episodic_id = None
            if episodic_id is None:
                continue
            checked += 1
            src = self.conn.execute(
                "SELECT content FROM memory_items WHERE id=?",
                (episodic_id,)).fetchone()
            if src is None:
                detail = "source episodic item is missing"
            elif _normalize(src[0]) != _normalize(content):
                detail = "promoted content no longer matches its source episodic item"
            else:
                continue
            mismatches.append(ComprehensionMismatch(episodic_id, semantic_id, detail))
            if self.ledger:
                self.ledger.append("cognition", "COMPREHENSION_MISMATCH", {
                    "episodic_id": episodic_id, "semantic_id": semantic_id,
                    "detail": detail})
        return ComprehensionReport(checked, tuple(mismatches))
