"""Memory consolidation: episodic -> semantic promotion.

A consolidation run scans active episodic items and proposes promotions:

- Candidates are active episodic items that are not verbatim chat
  transcripts (``user: ...`` / ``assistant: ...`` are events, not facts),
  not daily-episode narratives (tagged ``daily-episode``: a narrative
  of the day is a record, never a fact to promote), not already
  covered by an active semantic item with the same content, and not
  already the subject of a pending or completed proposal.
- A candidate with confidence >= ``auto_threshold`` (default 0.9) is
  auto-promoted: a semantic item is created carrying the original
  provenance, the episodic item is marked superseded by it (history is
  preserved, never erased), and the promotion is written to the ledger.
- A candidate below the auto threshold but at or above ``min_confidence``
  (default 0.5) is queued as a pending proposal for owner approval.
  ``approve()`` promotes it; ``reject()`` declines it and leaves the
  episodic item active. Both decisions are ledgered.
- Candidates below ``min_confidence`` are too weak even to queue and are
  skipped; a later run may reconsider them if their confidence rises.

Runs are idempotent: promoted items are no longer active episodic items,
and queued items already have a pending proposal, so a second run
proposes nothing new for them.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

DEFAULT_AUTO_THRESHOLD = 0.9
DEFAULT_MIN_CONFIDENCE = 0.5

_CHAT_PREFIXES = ("user:", "assistant:")


def _normalize(content: str) -> str:
    return " ".join(content.strip().lower().split())


@dataclass(frozen=True)
class Proposal:
    id: int
    episodic_id: int
    content: str
    source: str
    confidence: float
    status: str  # pending | promoted | rejected
    ts: float
    semantic_id: int | None = None


@dataclass(frozen=True)
class ConsolidationResult:
    promoted: tuple[int, ...]  # new semantic item ids
    queued: tuple[int, ...]  # new proposal ids
    skipped: tuple[int, ...]  # episodic ids skipped (too weak / duplicate / transcript)


class Consolidator:
    def __init__(self, memory, ledger=None,
                 auto_threshold: float = DEFAULT_AUTO_THRESHOLD,
                 min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        if not 0.0 <= min_confidence <= auto_threshold <= 1.0:
            raise ValueError("require 0 <= min_confidence <= auto_threshold <= 1")
        self.memory = memory
        self.ledger = ledger
        self.conn: sqlite3.Connection = memory.conn
        self.auto_threshold = auto_threshold
        self.min_confidence = min_confidence
        self.conn.execute("""CREATE TABLE IF NOT EXISTS consolidation_proposals(
            id INTEGER PRIMARY KEY AUTOINCREMENT, episodic_id INTEGER NOT NULL,
            content TEXT NOT NULL, source TEXT NOT NULL, confidence REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', ts REAL NOT NULL,
            semantic_id INTEGER)""")
        # RSI R2 persistence: an auto-adopted threshold change must
        # survive a Core restart, or "adoption" would be a session-local
        # fiction. One row, id 1, holds the live thresholds; it is
        # written only by set_thresholds(), never by __init__.
        self.conn.execute("""CREATE TABLE IF NOT EXISTS consolidation_settings(
            id INTEGER PRIMARY KEY CHECK(id = 1),
            auto_threshold REAL NOT NULL, min_confidence REAL NOT NULL)""")
        self.conn.commit()

    def load_persisted(self) -> bool:
        """Apply previously adopted thresholds, if any. Returns True
        when a persisted row was found. Core calls this on startup;
        a directly-constructed Consolidator keeps its constructor
        arguments unless the caller asks for the persisted values,
        so sandbox cores seeded with explicit thresholds are unaffected."""
        row = self.conn.execute(
            "SELECT auto_threshold, min_confidence FROM consolidation_settings"
            " WHERE id=1").fetchone()
        if row is None:
            return False
        auto_threshold, min_confidence = float(row[0]), float(row[1])
        if not 0.0 <= min_confidence <= auto_threshold <= 1.0:
            raise ValueError("persisted consolidation thresholds are invalid")
        self.auto_threshold = auto_threshold
        self.min_confidence = min_confidence
        return True

    def set_thresholds(self, auto_threshold: float | None = None,
                       min_confidence: float | None = None) -> None:
        """Change thresholds and persist them (RSI R2 auto-adoption and
        rollback use this; a plain in-memory change would not survive
        a restart). Validation is the same ordering rule as __init__."""
        new_auto = self.auto_threshold if auto_threshold is None else float(auto_threshold)
        new_min = self.min_confidence if min_confidence is None else float(min_confidence)
        if not 0.0 <= new_min <= new_auto <= 1.0:
            raise ValueError("require 0 <= min_confidence <= auto_threshold <= 1")
        self.auto_threshold = new_auto
        self.min_confidence = new_min
        self.conn.execute(
            "INSERT INTO consolidation_settings(id, auto_threshold, min_confidence)"
            " VALUES(1, ?, ?) ON CONFLICT(id) DO UPDATE SET"
            " auto_threshold=excluded.auto_threshold,"
            " min_confidence=excluded.min_confidence",
            (new_auto, new_min))
        self.conn.commit()

    # -- queries ---------------------------------------------------------
    def pending(self) -> list[Proposal]:
        return self._proposals("WHERE status='pending' ORDER BY id ASC")

    def proposals(self, status: str | None = None) -> list[Proposal]:
        if status is None:
            return self._proposals("ORDER BY id ASC")
        return self._proposals("WHERE status=? ORDER BY id ASC", (status,))

    def _proposals(self, clause: str, args: tuple = ()) -> list[Proposal]:
        rows = self.conn.execute(
            f"SELECT id,episodic_id,content,source,confidence,status,ts,semantic_id "
            f"FROM consolidation_proposals {clause}", args).fetchall()
        return [Proposal(*r) for r in rows]

    def _has_open_or_done_proposal(self, episodic_id: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM consolidation_proposals WHERE episodic_id=? AND status IN ('pending','promoted')",
            (episodic_id,)).fetchone()
        return row is not None

    def _semantic_covers(self, content: str) -> bool:
        wanted = _normalize(content)
        return any(_normalize(i.content) == wanted
                   for i in self.memory.all_active(kind="semantic"))

    # -- promotion -------------------------------------------------------
    def _promote(self, episodic_id: int, content: str, source: str,
                 confidence: float) -> int:
        """Create the semantic item, consume the episodic item, ledger it."""
        semantic_id = self.memory.fact(
            content, source=source, confidence=float(confidence),
            tags=[f"promoted_from_episodic:{episodic_id}"])
        self.conn.execute(
            "UPDATE memory_items SET superseded_by=? WHERE id=?",
            (semantic_id, episodic_id))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("consolidation", "MEMORY_PROMOTE", {
                "episodic_id": episodic_id, "semantic_id": semantic_id,
                "source": source, "confidence": float(confidence)})
        return semantic_id

    # -- the job ---------------------------------------------------------
    def run(self) -> ConsolidationResult:
        promoted: list[int] = []
        queued: list[int] = []
        skipped: list[int] = []
        for item in self.memory.all_active(kind="episodic"):
            if item.content.strip().lower().startswith(_CHAT_PREFIXES):
                skipped.append(item.id)
                continue
            if "daily-episode" in (item.tags or []):
                skipped.append(item.id)
                continue
            if self._has_open_or_done_proposal(item.id) or self._semantic_covers(item.content):
                skipped.append(item.id)
                continue
            if item.confidence < self.min_confidence:
                skipped.append(item.id)
                continue
            if item.confidence >= self.auto_threshold:
                promoted.append(self._promote(item.id, item.content, item.source, item.confidence))
            else:
                cur = self.conn.execute(
                    "INSERT INTO consolidation_proposals(episodic_id,content,source,confidence,ts)"
                    " VALUES(?,?,?,?,?)",
                    (item.id, item.content, item.source, float(item.confidence), time.time()))
                self.conn.commit()
                queued.append(cur.lastrowid)
                if self.ledger:
                    self.ledger.append("consolidation", "CONSOLIDATION_PROPOSAL", {
                        "proposal_id": cur.lastrowid, "episodic_id": item.id,
                        "confidence": float(item.confidence)})
        return ConsolidationResult(tuple(promoted), tuple(queued), tuple(skipped))

    # -- owner decisions -------------------------------------------------
    def approve(self, proposal_id: int) -> int:
        """Promote a pending proposal. Returns the new semantic item id."""
        rows = self._proposals("WHERE id=?", (proposal_id,))
        if not rows:
            raise KeyError(f"no consolidation proposal {proposal_id}")
        proposal = rows[0]
        if proposal.status != "pending":
            raise ValueError(f"proposal {proposal_id} is already {proposal.status}")
        semantic_id = self._promote(
            proposal.episodic_id, proposal.content, proposal.source, proposal.confidence)
        self.conn.execute(
            "UPDATE consolidation_proposals SET status='promoted', semantic_id=? WHERE id=?",
            (semantic_id, proposal_id))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("consolidation", "CONSOLIDATION_DECISION", {
                "proposal_id": proposal_id, "decision": "approved",
                "episodic_id": proposal.episodic_id, "semantic_id": semantic_id})
        return semantic_id

    def reject(self, proposal_id: int) -> Proposal:
        rows = self._proposals("WHERE id=?", (proposal_id,))
        if not rows:
            raise KeyError(f"no consolidation proposal {proposal_id}")
        proposal = rows[0]
        if proposal.status != "pending":
            raise ValueError(f"proposal {proposal_id} is already {proposal.status}")
        self.conn.execute(
            "UPDATE consolidation_proposals SET status='rejected' WHERE id=?", (proposal_id,))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("consolidation", "CONSOLIDATION_DECISION", {
                "proposal_id": proposal_id, "decision": "rejected",
                "episodic_id": proposal.episodic_id})
        return Proposal(proposal.id, proposal.episodic_id, proposal.content,
                        proposal.source, proposal.confidence, "rejected",
                        proposal.ts, proposal.semantic_id)
