"""Backup & restore: a one-command snapshot of a data dir.

A backup is a single zip file::

    steelbound.db          # a consistent copy of the SQLite database
    workspace/<relpath>    # every file under the data dir's workspace/
    manifest.json          # format, version, hashes, counts

Why the database copy goes through SQLite's online backup API instead
of a plain file copy: a Core may hold the database open, and a raw copy
of a live SQLite file can be torn. ``conn.backup()`` produces a
transactionally consistent snapshot even while the source connection
is open, so ``backup`` is safe to run against a live core.

Verification happens before restore writes anything (``verify_backup``):
the manifest's format/version, the SHA-256 of the database and of every
workspace file, the ledger hash chain inside the backed-up database,
and a digest over all memory rows are all recomputed. A backup whose
database was altered — even one whose manifest hashes were rewritten
to match the altered file — still fails, because the ledger chain
itself no longer verifies. A failed verification writes nothing to
the target.

Restore is an exact copy: the restored database is byte-identical to
the backed-up one, so the restored ledger chain verifies with the same
entries (no new entry is appended by the restore itself; the next Core
open appends its usual CORE_INIT, as always). Restore refuses a
non-empty target unless ``force=True``, so it cannot silently clobber
a live data dir. Restoring the same backup twice into the same target
without ``force`` is therefore an error, not a merge — unlike import,
which is a merge; use import for merging, backup/restore for
disaster recovery.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
import zipfile

from .ledger import Ledger

BACKUP_FORMAT = "steelbound-backup"
BACKUP_VERSION = 1

_DB_MEMBER = "steelbound.db"
_MANIFEST_MEMBER = "manifest.json"
_WORKSPACE_PREFIX = "workspace/"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _db_path(data_dir: str) -> str:
    return os.path.join(data_dir, "steelbound.db")


def _workspace_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "workspace")


def _memory_digest(conn: sqlite3.Connection) -> str:
    """Digest over ALL memory rows, mirroring the exporter's integrity check."""
    try:
        rows = conn.execute(
            "SELECT id,kind,content,source,confidence,ts,superseded_by,tags "
            "FROM memory_items ORDER BY id ASC").fetchall()
    except sqlite3.Error as exc:
        raise ValueError(f"backed-up database has no readable memory table: {exc}")
    memory = [{"id": r[0], "kind": r[1], "content": r[2], "source": r[3],
               "confidence": r[4], "ts": r[5], "superseded_by": r[6],
               "tags": json.loads(r[7])} for r in rows]
    return hashlib.sha256(_canonical({"memory": memory}).encode()).hexdigest()


def _check_member_name(name: str) -> None:
    """Reject zip-slip names: absolute paths, drive paths, or '..' parts."""
    if not name or name.startswith("/") or name.startswith("\\"):
        raise ValueError(f"unsafe backup member name: {name!r}")
    if len(name) >= 2 and name[1] == ":":
        raise ValueError(f"unsafe backup member name: {name!r}")
    parts = name.replace("\\", "/").split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"unsafe backup member name: {name!r}")


def _ledger_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM ledger_entries").fetchone()[0]


# -- creation ---------------------------------------------------------------

def create_backup(data_dir: str, out_path: str, conn=None) -> dict:
    """Snapshot ``data_dir`` into a zip file at ``out_path``.

    ``conn``, when given, should be the live connection of a Core on
    this data dir; its backup API produces the database snapshot. When
    omitted, a fresh connection is opened for the same purpose. Returns
    a summary dict. Overwrites an existing ``out_path`` atomically
    (write to a temp file, then replace).
    """
    db = _db_path(data_dir)
    if not os.path.isfile(db):
        raise FileNotFoundError(f"no Steelbound database at {db!r}")
    own_conn = None
    if conn is None:
        own_conn = sqlite3.connect(db)
        conn = own_conn
    tmpdir = tempfile.mkdtemp(prefix="steelbound-backup-")
    try:
        # Consistent database snapshot via the online backup API.
        snap_path = os.path.join(tmpdir, "steelbound.db")
        target = sqlite3.connect(snap_path)
        try:
            conn.backup(target)
        finally:
            target.close()
        db_sha = _sha256_file(snap_path)
        snap = sqlite3.connect(snap_path)
        try:
            ledger_entries = _ledger_count(snap)
            memory_sha = _memory_digest(snap)
        finally:
            snap.close()

        # Workspace files, sorted for a deterministic manifest.
        ws = _workspace_dir(data_dir)
        files: list[dict] = []
        if os.path.isdir(ws):
            for root, _dirs, names in os.walk(ws):
                for name in sorted(names):
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, ws).replace(os.sep, "/")
                    files.append({"path": rel,
                                  "sha256": _sha256_file(full),
                                  "size": os.path.getsize(full)})
        files.sort(key=lambda f: f["path"])

        manifest = {
            "format": BACKUP_FORMAT,
            "version": BACKUP_VERSION,
            "created_at": time.time(),
            "db_sha256": db_sha,
            "db_size": os.path.getsize(snap_path),
            "ledger_entries": ledger_entries,
            "memory_sha256": memory_sha,
            "files": files,
        }

        tmp_zip = os.path.join(tmpdir, "backup.zip")
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(snap_path, _DB_MEMBER)
            for entry in files:
                zf.write(os.path.join(ws, entry["path"].replace("/", os.sep)),
                         _WORKSPACE_PREFIX + entry["path"])
            zf.writestr(_MANIFEST_MEMBER,
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        os.replace(tmp_zip, out_path)
        return {"path": out_path, "db_sha256": db_sha,
                "ledger_entries": ledger_entries,
                "files": len(files),
                "size": os.path.getsize(out_path)}
    finally:
        if own_conn is not None:
            own_conn.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


# -- verification -----------------------------------------------------------

def verify_backup(path: str) -> dict:
    """Verify a backup file without writing anything outside temp files.

    Recomputes every hash in the manifest, re-verifies the ledger chain
    inside the backed-up database, and recomputes the memory digest.
    Returns the manifest. Raises ValueError on the first problem found.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no backup file at {path!r}")
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a valid backup zip: {exc}")
    with zf:
        names = zf.namelist()
        for name in names:
            _check_member_name(name)
        if _DB_MEMBER not in names or _MANIFEST_MEMBER not in names:
            raise ValueError("backup is missing steelbound.db or manifest.json")
        try:
            manifest = json.loads(zf.read(_MANIFEST_MEMBER).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"backup manifest is not valid JSON: {exc}")
        if not isinstance(manifest, dict):
            raise ValueError("backup manifest is not an object")
        if manifest.get("format") != BACKUP_FORMAT:
            raise ValueError(f"unknown backup format: {manifest.get('format')!r}")
        if manifest.get("version") != BACKUP_VERSION:
            raise ValueError(
                f"unsupported backup version: {manifest.get('version')!r}")
        listed = manifest.get("files")
        if not isinstance(listed, list):
            raise ValueError("backup manifest files must be a list")
        expected = {_DB_MEMBER, _MANIFEST_MEMBER}
        for entry in listed:
            if not isinstance(entry, dict) or "path" not in entry \
                    or "sha256" not in entry:
                raise ValueError("backup manifest has a malformed file entry")
            _check_member_name(entry["path"])
            expected.add(_WORKSPACE_PREFIX + entry["path"])
        if set(names) != expected:
            raise ValueError("backup members do not match its manifest")

        # File hashes.
        db_bytes = zf.read(_DB_MEMBER)
        if _sha256_bytes(db_bytes) != manifest.get("db_sha256"):
            raise ValueError("database hash mismatch (tampered or corrupt backup?)")
        for entry in listed:
            data = zf.read(_WORKSPACE_PREFIX + entry["path"])
            if _sha256_bytes(data) != entry["sha256"]:
                raise ValueError(
                    f"workspace file hash mismatch: {entry['path']!r}")
            if len(data) != entry.get("size"):
                raise ValueError(f"workspace file size mismatch: {entry['path']!r}")

        # Ledger chain + memory digest inside the backed-up database.
        # Work on a temp copy so Ledger's trigger setup touches nothing real.
        tmpdir = tempfile.mkdtemp(prefix="steelbound-verify-")
        try:
            tmp_db = os.path.join(tmpdir, "steelbound.db")
            with open(tmp_db, "wb") as f:
                f.write(db_bytes)
            conn = sqlite3.connect(tmp_db)
            try:
                ledger = Ledger(conn)
                if not ledger.verify():
                    raise ValueError(
                        "ledger chain in backup does not verify "
                        "(tampered ledger?)")
                if ledger.count() != manifest.get("ledger_entries"):
                    raise ValueError("ledger entry count does not match manifest")
                if _memory_digest(conn) != manifest.get("memory_sha256"):
                    raise ValueError(
                        "memory digest mismatch (tampered memory?)")
            finally:
                conn.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return manifest


# -- restore ----------------------------------------------------------------

def restore_backup(path: str, data_dir: str, force: bool = False) -> dict:
    """Verify then restore a backup into ``data_dir`` (exact copy).

    Raises ValueError before writing anything if verification fails,
    and if the target is non-empty without ``force=True``. With force,
    an existing database and workspace tree are replaced. Returns a
    summary dict; the restored chain is verified once more after
    extraction, so a successful return always means a valid chain.
    """
    manifest = verify_backup(path)
    if os.path.exists(data_dir) and os.listdir(data_dir) and not force:
        raise ValueError(
            f"target data dir {data_dir!r} is not empty; "
            "restore into a fresh dir or pass force=True")
    with zipfile.ZipFile(path) as zf:
        db_bytes = zf.read(_DB_MEMBER)
        ws_files = [(entry["path"], zf.read(_WORKSPACE_PREFIX + entry["path"]))
                    for entry in manifest["files"]]
    if force and os.path.isdir(data_dir):
        # Replace only what a backup owns: the database and the workspace
        # tree. Anything else in the dir is left alone.
        db_target = _db_path(data_dir)
        if os.path.isfile(db_target):
            os.remove(db_target)
        ws_target = _workspace_dir(data_dir)
        if os.path.isdir(ws_target):
            shutil.rmtree(ws_target)
    os.makedirs(_workspace_dir(data_dir), exist_ok=True)
    with open(_db_path(data_dir), "wb") as f:
        f.write(db_bytes)
    for rel, data in ws_files:
        dest = os.path.join(_workspace_dir(data_dir), rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
    # Post-restore chain check on the restored database itself.
    conn = sqlite3.connect(_db_path(data_dir))
    try:
        if not Ledger(conn).verify():
            raise ValueError("restored ledger chain does not verify")
    finally:
        conn.close()
    return {"data_dir": data_dir,
            "ledger_entries": manifest.get("ledger_entries"),
            "files_restored": len(ws_files),
            "chain_valid": True}
