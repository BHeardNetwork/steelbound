"""Append-only, hash-chained ledger.

Append-only is enforced in the database itself: SQLite triggers abort any
UPDATE or DELETE on ledger_entries, so even code with raw DB access cannot
rewrite history through normal SQL. Every entry hashes the previous entry;
any tampering breaks the chain and is detected by verify().
"""
from __future__ import annotations
import hashlib, json, sqlite3, time
from dataclasses import dataclass

GENESIS = "0" * 64

def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _hash(prev: str, ts: float, actor: str, action: str, payload: dict) -> str:
    h = hashlib.sha256()
    h.update(prev.encode()); h.update(str(ts).encode()); h.update(actor.encode())
    h.update(action.encode()); h.update(_canonical(payload).encode())
    return h.hexdigest()

@dataclass(frozen=True)
class LedgerEntry:
    id: int; ts: float; actor: str; action: str; payload: dict; prev_hash: str; entry_hash: str

class Ledger:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        conn.execute("""CREATE TABLE IF NOT EXISTS ledger_entries(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, actor TEXT NOT NULL,
            action TEXT NOT NULL, payload TEXT NOT NULL, prev_hash TEXT NOT NULL, entry_hash TEXT NOT NULL)""")
        conn.execute("""CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger_entries
            BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END""")
        conn.execute("""CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger_entries
            BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END""")
        conn.commit()

    def append(self, actor: str, action: str, payload: dict | None = None) -> LedgerEntry:
        payload = payload or {}
        row = self.conn.execute("SELECT entry_hash FROM ledger_entries ORDER BY id DESC LIMIT 1").fetchone()
        prev = row[0] if row else GENESIS
        ts = time.time()
        eh = _hash(prev, ts, actor, action, payload)
        cur = self.conn.execute(
            "INSERT INTO ledger_entries(ts,actor,action,payload,prev_hash,entry_hash) VALUES(?,?,?,?,?,?)",
            (ts, actor, action, _canonical(payload), prev, eh))
        self.conn.commit()
        return LedgerEntry(cur.lastrowid, ts, actor, action, payload, prev, eh)

    def entries(self, limit: int = 100) -> list[LedgerEntry]:
        rows = self.conn.execute(
            "SELECT id,ts,actor,action,payload,prev_hash,entry_hash FROM ledger_entries ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [LedgerEntry(r[0], r[1], r[2], r[3], json.loads(r[4]), r[5], r[6]) for r in reversed(rows)]

    def verify(self) -> bool:
        prev = GENESIS
        for (ts, actor, action, payload, ph, eh) in self.conn.execute(
                "SELECT ts,actor,action,payload,prev_hash,entry_hash FROM ledger_entries ORDER BY id ASC"):
            if ph != prev: return False
            if _hash(prev, ts, actor, action, json.loads(payload)) != eh: return False
            prev = eh
        return True

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM ledger_entries").fetchone()[0]
