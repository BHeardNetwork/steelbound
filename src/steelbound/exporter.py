"""Export/import: memory + ledger as a single JSON bundle.

Bundle format (version 1)::

    {
      "format": "steelbound-bundle",
      "version": 1,
      "exported_at": <unix timestamp>,
      "ledger": [{"ts", "actor", "action", "payload",
                  "prev_hash", "entry_hash"}, ...],   # oldest first
      "memory": [{"id", "kind", "content", "source", "confidence",
                  "ts", "superseded_by", "tags"}, ...],  # ALL items, incl. superseded
      "proposals": [{"id", "episodic_id", "content", "source",
                     "confidence", "status", "ts", "semantic_id"}, ...],
      "integrity": {"memory_sha256": <hex>}  # over the canonical memory+proposals lists
    }

Why the ledger is re-chained on import instead of copied verbatim: a
bundle's entries form a chain rooted at the genesis hash, but a target
ledger already has its own chain (Core writes a CORE_INIT entry at
startup, so a target is never empty). Splicing two chains would break
both. So import does two verifications:

1. Before writing anything, the bundle's own chain is re-verified from
   the genesis hash, and the memory digest is recomputed. Any mismatch
   raises ValueError and nothing is written.
2. Imported entries are re-chained onto the target ledger: their
   ts/actor/action/payload are preserved exactly and their hashes are
   recomputed against the target's current tip. After import the whole
   target chain is verified again, and the import itself is ledgered as
   BUNDLE_IMPORT.

Memory ids are remapped (the target may already hold items), and
superseded_by / proposal references are remapped through the same map,
so correction history and pending consolidation proposals survive the
round trip. Importing the same bundle twice duplicates content — it is
a merge, not a sync — so restores should target a fresh data dir.
"""
from __future__ import annotations

import hashlib
import json
import time

from .ledger import GENESIS, _hash

BUNDLE_FORMAT = "steelbound-bundle"
BUNDLE_VERSION = 1

_LEDGER_KEYS = ("ts", "actor", "action", "payload", "prev_hash", "entry_hash")
_MEMORY_KEYS = ("id", "kind", "content", "source", "confidence", "ts",
                "superseded_by", "tags")
_PROPOSAL_KEYS = ("id", "episodic_id", "content", "source", "confidence",
                  "status", "ts", "semantic_id")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _memory_digest(memory: list[dict], proposals: list[dict]) -> str:
    return hashlib.sha256(
        _canonical({"memory": memory, "proposals": proposals}).encode()
    ).hexdigest()


# -- export ---------------------------------------------------------------

def export_bundle(core) -> dict:
    """Snapshot a core's ledger, memory and proposals into one dict."""
    ledger_rows = core.conn.execute(
        "SELECT ts,actor,action,payload,prev_hash,entry_hash "
        "FROM ledger_entries ORDER BY id ASC").fetchall()
    ledger = [{"ts": r[0], "actor": r[1], "action": r[2],
               "payload": json.loads(r[3]), "prev_hash": r[4],
               "entry_hash": r[5]} for r in ledger_rows]
    mem_rows = core.conn.execute(
        "SELECT id,kind,content,source,confidence,ts,superseded_by,tags "
        "FROM memory_items ORDER BY id ASC").fetchall()
    memory = [{"id": r[0], "kind": r[1], "content": r[2], "source": r[3],
               "confidence": r[4], "ts": r[5], "superseded_by": r[6],
               "tags": json.loads(r[7])} for r in mem_rows]
    proposals = []
    try:
        rows = core.conn.execute(
            "SELECT id,episodic_id,content,source,confidence,status,ts,semantic_id "
            "FROM consolidation_proposals ORDER BY id ASC").fetchall()
        proposals = [dict(zip(_PROPOSAL_KEYS, r)) for r in rows]
    except Exception:
        # Cores without a proposals table (older data dirs) export fine.
        proposals = []
    return {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "exported_at": time.time(),
        "ledger": ledger,
        "memory": memory,
        "proposals": proposals,
        "integrity": {"memory_sha256": _memory_digest(memory, proposals)},
    }


# -- verification -----------------------------------------------------------

def _require_keys(entry: dict, keys, where: str) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: entry is not an object")
    for key in keys:
        if key not in entry:
            raise ValueError(f"{where}: missing key {key!r}")


def verify_bundle(bundle: dict) -> None:
    """Validate a bundle's shape, ledger chain and memory digest.

    Raises ValueError on the first problem found. Writes nothing.
    """
    if not isinstance(bundle, dict):
        raise ValueError("bundle is not an object")
    if bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"unknown bundle format: {bundle.get('format')!r}")
    if bundle.get("version") != BUNDLE_VERSION:
        raise ValueError(f"unsupported bundle version: {bundle.get('version')!r}")
    ledger = bundle.get("ledger")
    memory = bundle.get("memory")
    proposals = bundle.get("proposals")
    if not isinstance(ledger, list) or not isinstance(memory, list) \
            or not isinstance(proposals, list):
        raise ValueError("bundle ledger/memory/proposals must be lists")

    # Ledger chain, recomputed from genesis exactly as Ledger.verify does.
    prev = GENESIS
    for i, entry in enumerate(ledger):
        _require_keys(entry, _LEDGER_KEYS, f"ledger[{i}]")
        if entry["prev_hash"] != prev:
            raise ValueError(f"ledger[{i}]: prev_hash does not link to prior entry")
        if _hash(prev, entry["ts"], entry["actor"], entry["action"],
                 entry["payload"]) != entry["entry_hash"]:
            raise ValueError(f"ledger[{i}]: entry_hash mismatch (tampered entry?)")
        prev = entry["entry_hash"]

    # Memory items: shape, confidence range, and digest over the lists.
    for i, item in enumerate(memory):
        _require_keys(item, _MEMORY_KEYS, f"memory[{i}]")
        if item["kind"] not in ("episodic", "semantic"):
            raise ValueError(f"memory[{i}]: unknown kind {item['kind']!r}")
        if not isinstance(item["content"], str) or not isinstance(item["source"], str):
            raise ValueError(f"memory[{i}]: content/source must be strings")
        if not isinstance(item["confidence"], (int, float)) \
                or not 0.0 <= float(item["confidence"]) <= 1.0:
            raise ValueError(f"memory[{i}]: confidence out of range")
        if not isinstance(item["tags"], list) \
                or not all(isinstance(t, str) for t in item["tags"]):
            raise ValueError(f"memory[{i}]: tags must be a list of strings")
    for i, pr in enumerate(proposals):
        _require_keys(pr, _PROPOSAL_KEYS, f"proposals[{i}]")
        if pr["status"] not in ("pending", "promoted", "rejected"):
            raise ValueError(f"proposals[{i}]: unknown status {pr['status']!r}")
    integrity = bundle.get("integrity") or {}
    if integrity.get("memory_sha256") != _memory_digest(memory, proposals):
        raise ValueError("memory digest mismatch (tampered memory/proposals?)")


# -- import ---------------------------------------------------------------

def import_bundle(core, bundle: dict) -> dict:
    """Verify then merge a bundle into a core. Returns a summary dict.

    Raises ValueError before writing anything if verification fails.
    """
    verify_bundle(bundle)
    conn = core.conn

    # Memory first, with id remapping; superseded_by resolved in a pass
    # afterwards so forward references within the bundle still resolve.
    id_map: dict[int, int] = {}
    for item in bundle["memory"]:
        cur = conn.execute(
            "INSERT INTO memory_items(kind,content,source,confidence,ts,tags) "
            "VALUES(?,?,?,?,?,?)",
            (item["kind"], item["content"], item["source"],
             float(item["confidence"]), item["ts"],
             json.dumps(item["tags"])))
        id_map[item["id"]] = cur.lastrowid
    for item in bundle["memory"]:
        if item["superseded_by"] is not None \
                and item["superseded_by"] in id_map:
            conn.execute(
                "UPDATE memory_items SET superseded_by=? WHERE id=?",
                (id_map[item["superseded_by"]], id_map[item["id"]]))

    # Proposals, remapped; a reference outside the bundle is dropped
    # (None) rather than inventing an id.
    proposals_imported = 0
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS consolidation_proposals(
            id INTEGER PRIMARY KEY AUTOINCREMENT, episodic_id INTEGER NOT NULL,
            content TEXT NOT NULL, source TEXT NOT NULL, confidence REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', ts REAL NOT NULL,
            semantic_id INTEGER)""")
        for pr in bundle["proposals"]:
            if pr["episodic_id"] not in id_map:
                continue
            conn.execute(
                "INSERT INTO consolidation_proposals"
                "(episodic_id,content,source,confidence,status,ts,semantic_id) "
                "VALUES(?,?,?,?,?,?,?)",
                (id_map[pr["episodic_id"]], pr["content"], pr["source"],
                 float(pr["confidence"]), pr["status"], pr["ts"],
                 id_map.get(pr["semantic_id"])))
            proposals_imported += 1
    except Exception:
        conn.rollback()
        raise

    # Ledger entries, re-chained onto the target's current tip. Payloads,
    # actors, actions and timestamps are preserved exactly.
    row = conn.execute(
        "SELECT entry_hash FROM ledger_entries ORDER BY id DESC LIMIT 1").fetchone()
    prev = row[0] if row else GENESIS
    for entry in bundle["ledger"]:
        eh = _hash(prev, entry["ts"], entry["actor"], entry["action"],
                   entry["payload"])
        conn.execute(
            "INSERT INTO ledger_entries(ts,actor,action,payload,prev_hash,entry_hash) "
            "VALUES(?,?,?,?,?,?)",
            (entry["ts"], entry["actor"], entry["action"],
             _canonical(entry["payload"]), prev, eh))
        prev = eh
    conn.commit()

    core.ledger.append("import", "BUNDLE_IMPORT", {
        "ledger_imported": len(bundle["ledger"]),
        "memory_imported": len(bundle["memory"]),
        "proposals_imported": proposals_imported})
    if not core.ledger.verify():
        raise ValueError("target ledger chain invalid after import")
    return {"ledger_imported": len(bundle["ledger"]),
            "memory_imported": len(bundle["memory"]),
            "proposals_imported": proposals_imported,
            "ledger_valid": True}
