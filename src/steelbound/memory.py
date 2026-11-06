"""Layered memory: working (session), episodic (events), semantic (durable facts).
Every stored item carries provenance. Corrections supersede; they never erase history.
"""
from __future__ import annotations
import json, sqlite3, time
from dataclasses import dataclass, field

@dataclass
class MemoryItem:
    id: int; kind: str; content: str; source: str; confidence: float; ts: float
    superseded_by: int | None = None; tags: list = field(default_factory=list)

class Memory:
    def __init__(self, conn: sqlite3.Connection, ledger=None):
        self.conn = conn; self.ledger = ledger
        conn.execute("""CREATE TABLE IF NOT EXISTS memory_items(
            id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, content TEXT NOT NULL,
            source TEXT NOT NULL, confidence REAL NOT NULL, ts REAL NOT NULL,
            superseded_by INTEGER, tags TEXT NOT NULL DEFAULT '[]')""")
        conn.commit()
        self._working: list[str] = []

    # working memory (in-session only)
    def work_add(self, text: str) -> None: self._working.append(text)
    def work(self) -> list[str]: return list(self._working)
    def work_clear(self) -> None: self._working.clear()

    def _add(self, kind, content, source, confidence, tags):
        cur = self.conn.execute(
            "INSERT INTO memory_items(kind,content,source,confidence,ts,tags) VALUES(?,?,?,?,?,?)",
            (kind, content, source, float(confidence), time.time(), json.dumps(tags or [])))
        self.conn.commit()
        if self.ledger: self.ledger.append("memory", "MEMORY_ADD", {"kind": kind, "item_id": cur.lastrowid, "source": source})
        return cur.lastrowid

    def episode(self, content: str, source: str = "conversation", confidence: float = 1.0, tags=None) -> int:
        return self._add("episodic", content, source, confidence, tags)
    def fact(self, content: str, source: str, confidence: float = 0.9, tags=None) -> int:
        return self._add("semantic", content, source, confidence, tags)

    def correct(self, item_id: int, new_content: str, source: str) -> int:
        new_id = self._add("semantic", new_content, source, 0.95, [])
        self.conn.execute("UPDATE memory_items SET superseded_by=? WHERE id=?", (new_id, item_id)); self.conn.commit()
        if self.ledger: self.ledger.append("memory", "MEMORY_CORRECT", {"item_id": item_id, "superseded_by": new_id})
        return new_id

    def recall(self, query: str, kind: str | None = None, limit: int = 8) -> list[MemoryItem]:
        q = "SELECT id,kind,content,source,confidence,ts,superseded_by,tags FROM memory_items WHERE superseded_by IS NULL"
        args: list = []
        if kind: q += " AND kind=?"; args.append(kind)
        rows = self.conn.execute(q, args).fetchall()
        words = {w.lower() for w in query.split() if len(w) > 2}
        def score(r): return sum(1 for w in words if w in r[2].lower())
        items = [MemoryItem(r[0], r[1], r[2], r[3], r[4], r[5], r[6], json.loads(r[7])) for r in rows]
        def score_for(i): return sum(1 for w in words if w in i.content.lower())
        items.sort(key=lambda i: (score_for(i), i.ts), reverse=True)
        return items[:limit] if not words else [i for i in items if score_for(i) > 0][:limit] or items[:limit]

    def all_active(self, kind: str | None = None) -> list[MemoryItem]:
        return self.recall("", kind=kind, limit=10000)

    def delete(self, item_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM memory_items WHERE id=?", (item_id,)); self.conn.commit()
        if cur.rowcount and self.ledger:
            self.ledger.append("memory", "MEMORY_DELETE", {"item_id": item_id})  # records the act, never the content
        return cur.rowcount > 0

    def export(self) -> list[dict]:
        return [vars(i) for i in self.all_active()]
