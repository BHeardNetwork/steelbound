"""Self-healing weights: redundant, hash-verified checkpoint storage.

This is the Gen 3 item recorded in docs/SYNTHESIS.md: the model's
weights previously had no redundancy or repair — a corrupted
checkpoint was refused by ``SteelboundModel.load_checkpoint``, never
healed. Healing is an explicit layer *above* that stock loader: the
stock loader is untouched and still refuses corruption. Nothing in
this module ever weakens its hash checks.

Storage scheme (correctness over cleverness — mirrored shards)
---------------------------------------------------------------
``store_checkpoint_healed`` writes a completely ordinary checkpoint
directory (``weights.npz`` / ``vocab.json`` / ``manifest.json``,
byte-identical to what ``save_checkpoint`` or a verbatim copy of an
existing verified checkpoint produced, so the stock loader works on
it unchanged) plus a ``healing/`` subdirectory:

    healing/healing.json          healing manifest (see below)
    healing/shards/shard-0000.copy0.bin
    healing/shards/shard-0000.copy1.bin
    ...

The exact bytes of ``weights.npz`` are split into K contiguous
shards. Every shard is stored ``redundancy`` times (default 2) as
independent files, and the healing manifest records each shard's
offset, size and SHA-256, plus the whole-file SHA-256 of
``weights.npz`` and ``vocab.json``. Concatenating one verified copy
of every shard reproduces ``weights.npz`` byte-for-byte, so a repair
is proven correct by re-checking the reconstructed file's hash
against the checkpoint manifest — the same hash the stock loader
checks. (A slice of a surviving ``weights.npz`` whose per-shard hash
verifies is also accepted as a reconstruction source for that shard;
it is just as hash-proven as a shard copy.)

What healing can and cannot promise
-----------------------------------
* A manifest whose recorded hashes do not match the files is
  corruption, and is handled here. A manifest whose hashes were
  *edited to match tampered weights* is undetectable by hashing
  alone — this module does not claim otherwise.
* Only ``weights.npz`` is sharded. ``vocab.json`` and
  ``manifest.json`` are small, are verified against the manifest /
  healing manifest, and if they are themselves corrupt the local
  weights repair cannot produce a trustworthy checkpoint; the
  ancestor fallback below is the path for that case.

Repair semantics
----------------
``load_checkpoint_healed(directory, ancestor_dirs, ledger, repair)``:

1. Healthy checkpoint (stock hashes verify): loaded normally. If
   individual shard copies are degraded but the weights file itself
   is healthy, ``repair=True`` additionally rewrites the bad copies
   from verified bytes and ledgers ``CHECKPOINT_HEALED``; a fully
   healthy checkpoint produces no ledger entries at all.
2. Corrupt weights, local redundancy sufficient: the weights are
   reconstructed shard by shard from verified copies, the
   reconstruction's hash is checked against the manifest, and only
   then is ``weights.npz`` rewritten in place (atomically) and the
   model loaded. Ledgers ``CHECKPOINT_HEALED`` with the repaired
   shards and the repair source.
3. Local redundancy insufficient: the NEAREST healthy bloodline
   ancestor is loaded instead — ancestors are tried in the order
   supplied (nearest-first), each is used only if its *own*
   manifest hashes verify, and a corrupt ancestor is skipped, never
   trusted. The returned model *is the ancestor*: its generation and
   manifest are the ancestor's, and ``model.healing_report`` (plus a
   ``CHECKPOINT_HEALED_FROM_ANCESTOR`` ledger entry naming both
   checkpoints and both weights hashes) says so explicitly. The
   corrupted directory is NOT modified by a load-time fallback —
   silently overwriting a generation with its ancestor is exactly
   what this design refuses to do.
4. Nothing can repair: ledgers ``CHECKPOINT_HEAL_FAILED`` and raises
   :class:`CheckpointUnhealableError`.

``repair_checkpoint_dir(directory, ancestor_dirs, ledger,
acknowledge_ancestor=False)`` performs the same repair but returns a
report dict instead of a model, leaving the directory loadable by the
stock loader afterwards. Local repair rewrites only a hash-verified
reconstruction. Ancestor fallback in this function is stricter than
in ``load_checkpoint_healed``: because it would *write the ancestor's
weights, vocab and manifest into the corrupted directory*, it refuses
unless ``acknowledge_ancestor=True``; without the flag it reports
``ancestor_fallback_requires_acknowledgement`` and writes nothing.

Lineage
-------
Ancestors are, first-class, an explicit ordered list the caller
supplies (nearest-first). :func:`discover_ancestor_dirs` is a helper
for the standard bloodline layout: under a ``model_data`` root,
generation N's ancestors are ``genN-1`` ... ``gen0``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile

from .model import SteelboundModel

__all__ = [
    "CheckpointUnhealableError",
    "HEALING_FORMAT",
    "HEALING_VERSION",
    "discover_ancestor_dirs",
    "find_ancestor_dirs",
    "load_checkpoint_healed",
    "repair_checkpoint_dir",
    "store_checkpoint_healed",
    "verify_checkpoint",
]

HEALING_FORMAT = "steelbound-healing"
HEALING_VERSION = 1

_WEIGHTS = "weights.npz"
_VOCAB = "vocab.json"
_MANIFEST = "manifest.json"
_HEALING_DIR = "healing"
_HEALING_MANIFEST = "healing.json"
_SHARDS_DIR = "shards"

_ACTOR = "healing"
_GEN_DIR_RE = re.compile(r"^gen(\d+)$")


class CheckpointUnhealableError(ValueError):
    """A checkpoint is corrupt and neither its local redundancy nor
    any supplied healthy ancestor could repair it. Subclasses
    ValueError so existing corruption handling keeps working, but the
    message names exactly what was tried and what failed."""


# -- small helpers ------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _atomic_write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                               prefix=".heal-tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _ledger(ledger, action: str, payload: dict) -> None:
    if ledger is not None:
        ledger.append(_ACTOR, action, payload)


def _load_json_file(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _manifest_of(directory: str) -> dict | None:
    """The checkpoint's manifest dict, or None if missing/unusable."""
    path = os.path.join(directory, _MANIFEST)
    if not os.path.isfile(path):
        return None
    try:
        doc = _load_json_file(path)
    except (ValueError, UnicodeDecodeError, OSError):
        return None
    if not isinstance(doc, dict) or doc.get("format") != "steelbound-model":
        return None
    if not isinstance(doc.get("weights_sha256"), str) \
            or not isinstance(doc.get("vocab_sha256"), str):
        return None
    return doc


def _healing_manifest_of(directory: str) -> dict | None:
    path = os.path.join(directory, _HEALING_DIR, _HEALING_MANIFEST)
    if not os.path.isfile(path):
        return None
    try:
        doc = _load_json_file(path)
    except (ValueError, UnicodeDecodeError, OSError):
        return None
    if not isinstance(doc, dict) or doc.get("format") != HEALING_FORMAT \
            or doc.get("version") != HEALING_VERSION \
            or not isinstance(doc.get("shards"), list) \
            or not isinstance(doc.get("weights_sha256"), str):
        return None
    return doc


def _identity(directory: str) -> dict:
    """Best-effort checkpoint identity for reports/ledger payloads."""
    manifest = _manifest_of(directory)
    weights_path = os.path.join(directory, _WEIGHTS)
    return {
        "directory": str(directory),
        "generation": (manifest or {}).get("generation"),
        "weights_sha256_expected": (manifest or {}).get("weights_sha256"),
        "weights_sha256_actual": (_sha256_file(weights_path)
                                  if os.path.isfile(weights_path) else None),
    }


# -- storage ------------------------------------------------------------------

def store_checkpoint_healed(model_or_dir, dest_dir, shards: int = 8,
                            redundancy: int = 2, generation: int | None = None,
                            training: dict | None = None,
                            ledger=None) -> dict:
    """Store a checkpoint at ``dest_dir`` with sharded redundancy.

    ``model_or_dir`` is either a :class:`SteelboundModel` (saved via
    its own ``save_checkpoint``) or the path of an existing checkpoint
    whose manifest hashes verify — its three files are copied
    verbatim, so the stored checkpoint is byte-identical to the
    source. Refuses (ValueError) to build a healing store on top of a
    source checkpoint that does not verify: redundancy recorded over
    already-corrupt bytes would be a lie.

    ``shards`` is the number of contiguous shards the weights payload
    is split into (fewer if the payload is smaller than that);
    ``redundancy`` is the number of independent copies stored per
    shard (at least 1; 2 by default). Returns the healing manifest.
    """
    shards = int(shards)
    redundancy = int(redundancy)
    if shards < 1:
        raise ValueError("shards must be >= 1")
    if redundancy < 1:
        raise ValueError("redundancy must be >= 1")

    tmp_src = None
    if isinstance(model_or_dir, SteelboundModel):
        model = model_or_dir
        tmp_src = tempfile.mkdtemp(prefix="steelbound-heal-src-")
        gen = generation if generation is not None \
            else getattr(model, "generation", 0)
        train = training if training is not None \
            else (getattr(model, "manifest", None) or {}).get("training")
        model.save_checkpoint(tmp_src, training=train, generation=int(gen))
        source_dir = tmp_src
    else:
        source_dir = str(model_or_dir)
    try:
        manifest = _manifest_of(source_dir)
        if manifest is None:
            raise ValueError(
                f"source {source_dir!r} is not a readable steelbound-model "
                "checkpoint; refusing to store healing data for it")
        weights_path = os.path.join(source_dir, _WEIGHTS)
        vocab_path = os.path.join(source_dir, _VOCAB)
        if not os.path.isfile(weights_path) or not os.path.isfile(vocab_path):
            raise ValueError(
                f"source checkpoint {source_dir!r} is missing weights or vocab")
        if _sha256_file(weights_path) != manifest["weights_sha256"] \
                or _sha256_file(vocab_path) != manifest["vocab_sha256"]:
            raise ValueError(
                f"source checkpoint {source_dir!r} does not verify against "
                "its manifest; refusing to store healing data for corrupt "
                "bytes")
        # Read everything into memory first so dest_dir may equal (or
        # replace) the source directory safely.
        weights = _read_bytes(weights_path)
        vocab_bytes = _read_bytes(vocab_path)
        manifest_bytes = _read_bytes(os.path.join(source_dir, _MANIFEST))
    finally:
        if tmp_src is not None:
            shutil.rmtree(tmp_src, ignore_errors=True)

    dest_dir = str(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    _atomic_write(os.path.join(dest_dir, _WEIGHTS), weights)
    _atomic_write(os.path.join(dest_dir, _VOCAB), vocab_bytes)
    _atomic_write(os.path.join(dest_dir, _MANIFEST), manifest_bytes)

    healing_root = os.path.join(dest_dir, _HEALING_DIR)
    if os.path.isdir(healing_root):
        shutil.rmtree(healing_root)
    shards_root = os.path.join(healing_root, _SHARDS_DIR)
    os.makedirs(shards_root)

    total = len(weights)
    count = max(1, min(shards, total)) if total else 1
    chunk = (total + count - 1) // count if total else 0
    shard_docs = []
    offset = 0
    for index in range(count):
        size = min(chunk, total - offset) if total else 0
        piece = weights[offset:offset + size]
        copies = []
        for copy_no in range(redundancy):
            rel = f"{_HEALING_DIR}/{_SHARDS_DIR}/shard-{index:04d}.copy{copy_no}.bin"
            with open(os.path.join(dest_dir, rel.replace("/", os.sep)), "wb") as f:
                f.write(piece)
            copies.append(rel)
        shard_docs.append({
            "index": index,
            "offset": offset,
            "size": size,
            "sha256": _sha256_bytes(piece),
            "copies": copies,
        })
        offset += size

    healing_manifest = {
        "format": HEALING_FORMAT,
        "version": HEALING_VERSION,
        "weights_sha256": manifest["weights_sha256"],
        "vocab_sha256": manifest["vocab_sha256"],
        "weights_size": total,
        "generation": manifest.get("generation"),
        "shard_count": count,
        "redundancy": redundancy,
        "shards": shard_docs,
    }
    _atomic_write(os.path.join(healing_root, _HEALING_MANIFEST),
                  (json.dumps(healing_manifest, indent=2, sort_keys=True)
                   + "\n").encode("utf-8"))
    _ledger(ledger, "CHECKPOINT_HEALING_STORED", {
        "directory": dest_dir,
        "generation": manifest.get("generation"),
        "weights_sha256": manifest["weights_sha256"],
        "shard_count": count,
        "redundancy": redundancy,
    })
    return healing_manifest


# -- verification ---------------------------------------------------------------

def _file_status(directory: str, name: str, expected_sha: str | None) -> dict:
    path = os.path.join(directory, name)
    if not os.path.isfile(path):
        return {"path": name, "status": "missing", "sha256": None,
                "expected_sha256": expected_sha, "size": None}
    actual = _sha256_file(path)
    if expected_sha is None:
        status = "unknown"  # no trusted expectation to check against
    elif actual == expected_sha:
        status = "ok"
    else:
        status = "corrupt"
    return {"path": name, "status": status, "sha256": actual,
            "expected_sha256": expected_sha,
            "size": os.path.getsize(path)}


def verify_checkpoint(directory) -> dict:
    """Verify a checkpoint directory without raising, ever.

    Returns a report dict: per-file status for the standard checkpoint
    files (``ok`` / ``corrupt`` / ``missing`` / ``invalid`` /
    ``unknown``), per-shard and per-copy status for the healing
    store when present, whether the directory is healthy (stock
    hashes verify), whether it is degraded (healthy weights but
    damaged redundancy), and whether local shard repair could rebuild
    the weights. A tampered manifest shows up here as hash mismatches
    on the files; a manifest edited to match tampered weights cannot
    be detected by hashing alone and this report does not claim to.
    """
    directory = str(directory)
    problems: list[str] = []
    report: dict = {
        "directory": directory,
        "healthy": False,
        "degraded": False,
        "generation": None,
        "files": {},
        "healing": {"present": False, "manifest_status": "missing",
                    "shards": [], "repairable_locally": False},
        "repairable_locally": False,
        "problems": problems,
    }
    if not os.path.isdir(directory):
        problems.append(f"checkpoint directory {directory!r} does not exist")
        report["files"] = {name: {"path": name, "status": "missing",
                                  "sha256": None, "expected_sha256": None,
                                  "size": None}
                           for name in (_WEIGHTS, _VOCAB, _MANIFEST)}
        return report

    # Manifest first: its hashes are the expectations for the rest.
    manifest_path = os.path.join(directory, _MANIFEST)
    manifest = None
    if not os.path.isfile(manifest_path):
        manifest_status = "missing"
        problems.append("manifest.json is missing")
    else:
        manifest = _manifest_of(directory)
        if manifest is None:
            manifest_status = "invalid"
            problems.append(
                "manifest.json is unreadable or not a steelbound-model "
                "manifest with recorded hashes")
        else:
            manifest_status = "ok"
            report["generation"] = manifest.get("generation")
    report["files"][_MANIFEST] = {
        "path": _MANIFEST, "status": manifest_status,
        "sha256": (_sha256_file(manifest_path)
                   if os.path.isfile(manifest_path) else None),
        "expected_sha256": None,
        "size": (os.path.getsize(manifest_path)
                 if os.path.isfile(manifest_path) else None),
    }

    expected_w = manifest.get("weights_sha256") if manifest else None
    expected_v = manifest.get("vocab_sha256") if manifest else None
    # If the model manifest is unusable, the healing manifest is an
    # independent record of the expected hashes for status purposes.
    healing_manifest = _healing_manifest_of(directory)
    if expected_w is None and healing_manifest is not None:
        expected_w = healing_manifest.get("weights_sha256")
        expected_v = healing_manifest.get("vocab_sha256")

    report["files"][_WEIGHTS] = _file_status(directory, _WEIGHTS, expected_w)
    report["files"][_VOCAB] = _file_status(directory, _VOCAB, expected_v)
    for name in (_WEIGHTS, _VOCAB):
        status = report["files"][name]["status"]
        if status != "ok":
            problems.append(f"{name} is {status}")

    weights_ok = report["files"][_WEIGHTS]["status"] == "ok"
    vocab_ok = report["files"][_VOCAB]["status"] == "ok"
    # Healthy means the stock loader would accept it: manifest usable
    # AND both file hashes matching that manifest.
    report["healthy"] = bool(
        manifest is not None
        and _sha256_file(os.path.join(directory, _WEIGHTS)) == manifest["weights_sha256"]
        and _sha256_file(os.path.join(directory, _VOCAB)) == manifest["vocab_sha256"]
    ) if os.path.isfile(os.path.join(directory, _WEIGHTS)) \
        and os.path.isfile(os.path.join(directory, _VOCAB)) else False

    # Healing store.
    healing_path = os.path.join(directory, _HEALING_DIR, _HEALING_MANIFEST)
    if os.path.isfile(healing_path):
        report["healing"]["present"] = True
        if healing_manifest is None:
            report["healing"]["manifest_status"] = "invalid"
            problems.append("healing manifest is unreadable or malformed")
        else:
            report["healing"]["manifest_status"] = "ok"
            if healing_manifest["weights_sha256"] != expected_w:
                report["healing"]["manifest_status"] = "corrupt"
                problems.append(
                    "healing manifest weights hash disagrees with the "
                    "checkpoint manifest")
            weights_bytes = None
            weights_file = os.path.join(directory, _WEIGHTS)
            if os.path.isfile(weights_file):
                weights_bytes = _read_bytes(weights_file)
            all_shards_have_source = True
            any_copy_bad = False
            for shard in healing_manifest["shards"]:
                copies = []
                good = 0
                for rel in shard.get("copies", []):
                    cpath = os.path.join(directory, rel.replace("/", os.sep))
                    if not os.path.isfile(cpath):
                        copies.append({"path": rel, "status": "missing",
                                       "sha256": None})
                        any_copy_bad = True
                        continue
                    data = _read_bytes(cpath)
                    ok = len(data) == shard["size"] \
                        and _sha256_bytes(data) == shard["sha256"]
                    copies.append({"path": rel,
                                   "status": "ok" if ok else "corrupt",
                                   "sha256": _sha256_bytes(data)})
                    good += 1 if ok else 0
                    any_copy_bad = any_copy_bad or not ok
                slice_ok = False
                if weights_bytes is not None:
                    piece = weights_bytes[shard["offset"]:
                                          shard["offset"] + shard["size"]]
                    slice_ok = len(piece) == shard["size"] \
                        and _sha256_bytes(piece) == shard["sha256"]
                has_source = good > 0 or slice_ok
                all_shards_have_source = all_shards_have_source and has_source
                if not copies:
                    shard_status = "missing"
                elif good == len(copies) and not any(
                        c["status"] != "ok" for c in copies):
                    shard_status = "ok"
                elif has_source:
                    shard_status = "degraded"
                elif all(c["status"] == "missing" for c in copies) \
                        and not slice_ok:
                    shard_status = "missing"
                else:
                    shard_status = "corrupt"
                report["healing"]["shards"].append({
                    "index": shard["index"],
                    "offset": shard["offset"],
                    "size": shard["size"],
                    "expected_sha256": shard["sha256"],
                    "status": shard_status,
                    "good_copies": good,
                    "weights_slice_ok": slice_ok,
                    "copies": copies,
                })
            report["degraded"] = bool(any_copy_bad) and weights_ok
            # Local repair can rebuild the weights iff every shard has
            # a verified source AND the rest of the checkpoint (vocab
            # + a manifest to check the reconstruction against) is
            # intact.
            report["healing"]["repairable_locally"] = bool(
                all_shards_have_source and manifest is not None and vocab_ok)
            report["repairable_locally"] = \
                report["healing"]["repairable_locally"]
    return report


# -- lineage --------------------------------------------------------------------

def discover_ancestor_dirs(checkpoint_dir, model_data_root=None) -> list[str]:
    """Discover bloodline ancestors of ``checkpoint_dir``, nearest-first.

    Standard layout: the checkpoint lives at ``<root>/genN`` (or its
    manifest records generation N and ``model_data_root`` names the
    root); its ancestors are ``<root>/genN-1`` ... ``<root>/gen0``
    that exist on disk, nearest first. Returns [] when the generation
    cannot be determined or no ancestor directories exist. Existence
    is all this claims — callers must still verify an ancestor's own
    hashes before trusting it (the repair functions do).
    """
    checkpoint_dir = str(checkpoint_dir)
    generation = None
    match = _GEN_DIR_RE.match(os.path.basename(os.path.normpath(checkpoint_dir)))
    if match:
        generation = int(match.group(1))
        root = model_data_root or os.path.dirname(
            os.path.normpath(checkpoint_dir))
    else:
        manifest = _manifest_of(checkpoint_dir)
        if manifest is None or not isinstance(manifest.get("generation"), int):
            return []
        generation = int(manifest["generation"])
        root = model_data_root or os.path.dirname(
            os.path.normpath(checkpoint_dir))
    root = str(root)
    ancestors = []
    for gen in range(generation - 1, -1, -1):
        candidate = os.path.join(root, f"gen{gen}")
        if os.path.isdir(candidate) and os.path.normpath(candidate) != \
                os.path.normpath(checkpoint_dir):
            ancestors.append(candidate)
    return ancestors


# Backwards/forwards-friendly alias.
find_ancestor_dirs = discover_ancestor_dirs


# -- reconstruction ---------------------------------------------------------------

def _verified_shard_bytes(directory: str, healing_manifest: dict,
                          shard: dict, weights_bytes: bytes | None):
    """One hash-verified copy of a shard's bytes, or None.

    Sources, in order: each stored copy file, then the matching slice
    of the current weights file. Every candidate is accepted only if
    its length and SHA-256 match the healing manifest.
    """
    for rel in shard.get("copies", []):
        cpath = os.path.join(directory, rel.replace("/", os.sep))
        if os.path.isfile(cpath):
            data = _read_bytes(cpath)
            if len(data) == shard["size"] \
                    and _sha256_bytes(data) == shard["sha256"]:
                return data
    if weights_bytes is not None:
        piece = weights_bytes[shard["offset"]:shard["offset"] + shard["size"]]
        if len(piece) == shard["size"] \
                and _sha256_bytes(piece) == shard["sha256"]:
            return piece
    return None


def _reconstruct_weights(directory: str, expected_sha: str):
    """Rebuild weights bytes from verified shards.

    Returns (data, repaired_shard_indices) or (None, failed_indices).
    ``repaired`` lists shards whose slice of the current weights file
    did NOT verify — i.e. shards the reconstruction actually heals.
    The caller must still check the whole-file hash; this function
    already does and refuses (None) on mismatch.
    """
    healing_manifest = _healing_manifest_of(directory)
    if healing_manifest is None:
        return None, []
    weights_file = os.path.join(directory, _WEIGHTS)
    weights_bytes = _read_bytes(weights_file) \
        if os.path.isfile(weights_file) else None
    parts = []
    repaired, failed = [], []
    for shard in healing_manifest["shards"]:
        piece = _verified_shard_bytes(directory, healing_manifest, shard,
                                      weights_bytes)
        if piece is None:
            failed.append(shard["index"])
            continue
        slice_ok = False
        if weights_bytes is not None:
            current = weights_bytes[shard["offset"]:
                                    shard["offset"] + shard["size"]]
            slice_ok = len(current) == shard["size"] \
                and _sha256_bytes(current) == shard["sha256"]
        if not slice_ok:
            repaired.append(shard["index"])
        parts.append(piece)
    if failed:
        return None, failed
    data = b"".join(parts)
    if len(data) != healing_manifest.get("weights_size") \
            or _sha256_bytes(data) != expected_sha:
        return None, [-1]
    return data, repaired


def _repair_shard_copies(directory: str, weights_bytes: bytes) -> list[str]:
    """Rewrite every missing/corrupt shard copy from verified bytes
    (slices of ``weights_bytes``, each re-checked per shard). Returns
    the relative paths rewritten."""
    healing_manifest = _healing_manifest_of(directory)
    if healing_manifest is None:
        return []
    rewritten = []
    for shard in healing_manifest["shards"]:
        piece = weights_bytes[shard["offset"]:shard["offset"] + shard["size"]]
        if len(piece) != shard["size"] \
                or _sha256_bytes(piece) != shard["sha256"]:
            continue  # no verified source for this shard; leave copies alone
        for rel in shard.get("copies", []):
            cpath = os.path.join(directory, rel.replace("/", os.sep))
            current = _read_bytes(cpath) if os.path.isfile(cpath) else None
            if current is None or _sha256_bytes(current) != shard["sha256"]:
                _atomic_write(cpath, piece)
                rewritten.append(rel)
    return rewritten


def _try_local_repair(directory: str, report: dict):
    """Attempt shard reconstruction. Returns (weights_bytes,
    repaired_shards) or (None, reason). Never writes anything."""
    manifest = _manifest_of(directory)
    if manifest is None:
        return None, "checkpoint manifest is missing or unusable, so there " \
                     "is no trusted hash to verify a reconstruction against"
    if report["files"][_VOCAB]["status"] != "ok":
        return None, "vocab.json does not verify; weights shards cannot " \
                     "repair the vocabulary"
    if _healing_manifest_of(directory) is None:
        return None, "no usable healing store (healing/healing.json) present"
    data, info = _reconstruct_weights(directory, manifest["weights_sha256"])
    if data is None:
        return None, f"local redundancy insufficient (shards unrecoverable: " \
                     f"{info})"
    return (data, info), None


def _commit_local_repair(directory: str, data: bytes) -> list[str]:
    """Write a hash-verified reconstruction into the directory and
    refresh degraded shard copies from it. Returns repaired copies."""
    _atomic_write(os.path.join(directory, _WEIGHTS), data)
    return _repair_shard_copies(directory, data)


def _load_model_from_weights_bytes(directory: str, data: bytes):
    """Load a model whose (reconstructed) weights bytes are trusted,
    without depending on the on-disk weights file (used if writing
    the repair back failed, e.g. a read-only directory)."""
    tmp = tempfile.mkdtemp(prefix="steelbound-healed-load-")
    try:
        for name in (_VOCAB, _MANIFEST):
            shutil.copyfile(os.path.join(directory, name),
                            os.path.join(tmp, name))
        with open(os.path.join(tmp, _WEIGHTS), "wb") as f:
            f.write(data)
        return SteelboundModel.load_checkpoint(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _first_healthy_ancestor(ancestor_dirs):
    """The first ancestor (in supplied nearest-first order) whose own
    manifest hashes verify, as (dir, verify_report); else (None,
    tried_reports). Corrupt ancestors are skipped, never trusted."""
    tried = []
    for candidate in ancestor_dirs or ():
        candidate = str(candidate)
        rep = verify_checkpoint(candidate)
        tried.append(rep)
        if rep["healthy"]:
            return candidate, tried
    return None, tried


# -- healed loading ---------------------------------------------------------------

def load_checkpoint_healed(directory, ancestor_dirs=(), ledger=None,
                           repair: bool = True) -> SteelboundModel:
    """Load a checkpoint, healing it first if it is corrupt.

    See the module docstring for the full semantics. The returned
    model always carries a ``healing_report`` dict; when the result
    came from an ancestor, ``model.healed_from_ancestor`` is True,
    ``model.healing_source_dir`` is the ancestor directory, and the
    model's own generation/manifest are the ancestor's — the ancestor
    substitution is explicit, never silent. Raises
    :class:`CheckpointUnhealableError` when nothing can repair the
    checkpoint.
    """
    directory = str(directory)
    ancestor_dirs = [str(d) for d in (ancestor_dirs or ())]
    before = verify_checkpoint(directory)
    original = _identity(directory)

    # Path 1: stock-healthy.
    if before["healthy"]:
        healed_copies: list[str] = []
        if repair and before["degraded"]:
            weights = _read_bytes(os.path.join(directory, _WEIGHTS))
            healed_copies = _repair_shard_copies(directory, weights)
            if healed_copies:
                _ledger(ledger, "CHECKPOINT_HEALED", {
                    "directory": directory,
                    "generation": before["generation"],
                    "weights_sha256": original["weights_sha256_expected"],
                    "source": "weights_file",
                    "repaired_shards": [],
                    "repaired_copies": healed_copies,
                    "detail": "weights file healthy; degraded shard "
                              "copies rewritten from verified weights bytes",
                })
        model = SteelboundModel.load_checkpoint(directory)
        model.healing_report = {
            "status": "healthy" if not healed_copies else "healed_shard_copies",
            "healed": bool(healed_copies),
            "directory": directory,
            "generation": before["generation"],
            "weights_sha256": original["weights_sha256_expected"],
            "repaired_copies": healed_copies,
            "verify_before": before,
        }
        model.healed_from_ancestor = False
        model.healing_source_dir = directory
        return model

    # Corrupt from here on.
    if not repair:
        payload = {"directory": directory, "generation": before["generation"],
                   "weights_sha256_expected": original["weights_sha256_expected"],
                   "weights_sha256_actual": original["weights_sha256_actual"],
                   "problems": before["problems"],
                   "reason": "repair disabled (repair=False)"}
        _ledger(ledger, "CHECKPOINT_HEAL_FAILED", payload)
        raise CheckpointUnhealableError(
            f"checkpoint at {directory!r} is corrupt {before['problems']} "
            "and repair=False, so no healing was attempted")

    # Path 2: local shard repair.
    local, reason = _try_local_repair(directory, before)
    local_reason = reason
    if local is not None:
        data, repaired_shards = local
        repaired_copies: list[str] = []
        disk_repaired = True
        try:
            repaired_copies = _commit_local_repair(directory, data)
        except OSError:
            disk_repaired = False  # still load the verified bytes below
        after = verify_checkpoint(directory) if disk_repaired else before
        if disk_repaired and after["healthy"]:
            model = SteelboundModel.load_checkpoint(directory)
        else:
            model = _load_model_from_weights_bytes(directory, data)
        payload = {
            "directory": directory,
            "generation": before["generation"],
            "weights_sha256": _sha256_bytes(data),
            "source": "local_redundancy",
            "repaired_shards": repaired_shards,
            "repaired_copies": repaired_copies,
            "written_to_disk": disk_repaired,
            "problems_before": before["problems"],
        }
        _ledger(ledger, "CHECKPOINT_HEALED", payload)
        model.healing_report = {
            "status": "healed_locally",
            "healed": True,
            "directory": directory,
            "generation": before["generation"],
            "weights_sha256": _sha256_bytes(data),
            "repaired_shards": repaired_shards,
            "repaired_copies": repaired_copies,
            "written_to_disk": disk_repaired,
            "verify_before": before,
            "verify_after": after,
        }
        model.healed_from_ancestor = False
        model.healing_source_dir = directory
        return model

    # Path 3: nearest healthy ancestor (in memory only — the
    # corrupted directory is never overwritten by a load).
    ancestor, tried = _first_healthy_ancestor(ancestor_dirs)
    if ancestor is not None:
        ancestor_identity = _identity(ancestor)
        model = SteelboundModel.load_checkpoint(ancestor)
        payload = {
            "directory": directory,
            "generation": original["generation"],
            "weights_sha256_expected": original["weights_sha256_expected"],
            "weights_sha256_actual": original["weights_sha256_actual"],
            "ancestor_directory": ancestor,
            "ancestor_generation": ancestor_identity["generation"],
            "ancestor_weights_sha256": ancestor_identity["weights_sha256_expected"],
            "local_repair_failure": local_reason,
            "problems_before": before["problems"],
            "note": "returned model IS the ancestor checkpoint, not the "
                    "corrupted generation",
        }
        _ledger(ledger, "CHECKPOINT_HEALED_FROM_ANCESTOR", payload)
        model.healing_report = {
            "status": "healed_from_ancestor",
            "healed": True,
            "is_ancestor": True,
            "directory": directory,
            "generation": original["generation"],
            "weights_sha256_expected": original["weights_sha256_expected"],
            "ancestor_directory": ancestor,
            "ancestor_generation": ancestor_identity["generation"],
            "ancestor_weights_sha256":
                ancestor_identity["weights_sha256_expected"],
            "local_repair_failure": local_reason,
            "verify_before": before,
            "ancestors_tried": [t["directory"] for t in tried],
        }
        model.healed_from_ancestor = True
        model.healing_source_dir = ancestor
        return model

    # Path 4: nothing worked — fail loud.
    payload = {
        "directory": directory,
        "generation": original["generation"],
        "weights_sha256_expected": original["weights_sha256_expected"],
        "weights_sha256_actual": original["weights_sha256_actual"],
        "problems": before["problems"],
        "local_repair_failure": local_reason,
        "ancestors_tried": [t["directory"] for t in tried],
    }
    _ledger(ledger, "CHECKPOINT_HEAL_FAILED", payload)
    raise CheckpointUnhealableError(
        f"checkpoint at {directory!r} is unhealable: problems="
        f"{before['problems']}; local repair failed: {local_reason}; "
        f"no healthy ancestor among {[t['directory'] for t in tried]}")


# -- in-place directory repair ------------------------------------------------------

def repair_checkpoint_dir(directory, ancestor_dirs=(), ledger=None,
                          acknowledge_ancestor: bool = False) -> dict:
    """Repair a checkpoint directory in place; returns a report dict.

    Local repair rewrites ``weights.npz`` only with a reconstruction
    whose hash matches the checkpoint manifest, after which the
    directory loads via the stock ``load_checkpoint``. Ancestor
    fallback would replace this directory's weights, vocab and
    manifest with the ancestor's and regenerate its healing store —
    because that changes the directory's identity, it is refused
    unless ``acknowledge_ancestor=True`` (report status
    ``ancestor_fallback_requires_acknowledgement``, nothing written).
    An unhealable directory is reported with status ``unhealable``
    (and a ``CHECKPOINT_HEAL_FAILED`` ledger entry), not raised.
    """
    directory = str(directory)
    ancestor_dirs = [str(d) for d in (ancestor_dirs or ())]
    before = verify_checkpoint(directory)
    original = _identity(directory)
    report: dict = {
        "directory": directory,
        "generation": before["generation"],
        "healed": False,
        "status": "healthy",
        "verify_before": before,
        "verify_after": None,
    }
    if before["healthy"]:
        if before["degraded"]:
            weights = _read_bytes(os.path.join(directory, _WEIGHTS))
            healed_copies = _repair_shard_copies(directory, weights)
            if healed_copies:
                _ledger(ledger, "CHECKPOINT_HEALED", {
                    "directory": directory,
                    "generation": before["generation"],
                    "weights_sha256": original["weights_sha256_expected"],
                    "source": "weights_file",
                    "repaired_shards": [],
                    "repaired_copies": healed_copies,
                    "detail": "weights file healthy; degraded shard "
                              "copies rewritten from verified weights bytes",
                })
                report.update({"status": "healed_shard_copies",
                               "healed": True,
                               "repaired_copies": healed_copies})
        report["verify_after"] = verify_checkpoint(directory)
        return report

    local, reason = _try_local_repair(directory, before)
    if local is not None:
        data, repaired_shards = local
        repaired_copies = _commit_local_repair(directory, data)
        after = verify_checkpoint(directory)
        if not after["healthy"]:
            # Should be impossible (hash was verified pre-write); be loud
            # in the report rather than claiming a heal that did not land.
            report.update({"status": "repair_write_failed",
                           "repaired_shards": repaired_shards,
                           "verify_after": after})
            _ledger(ledger, "CHECKPOINT_HEAL_FAILED", {
                "directory": directory,
                "generation": original["generation"],
                "problems": before["problems"],
                "reason": "reconstructed weights written but directory "
                          "still does not verify",
            })
            return report
        _ledger(ledger, "CHECKPOINT_HEALED", {
            "directory": directory,
            "generation": before["generation"],
            "weights_sha256": _sha256_bytes(data),
            "source": "local_redundancy",
            "repaired_shards": repaired_shards,
            "repaired_copies": repaired_copies,
            "problems_before": before["problems"],
        })
        report.update({"status": "healed_locally", "healed": True,
                       "repaired_shards": repaired_shards,
                       "repaired_copies": repaired_copies,
                       "weights_sha256": _sha256_bytes(data),
                       "verify_after": after})
        return report

    ancestor, tried = _first_healthy_ancestor(ancestor_dirs)
    if ancestor is None:
        _ledger(ledger, "CHECKPOINT_HEAL_FAILED", {
            "directory": directory,
            "generation": original["generation"],
            "weights_sha256_expected": original["weights_sha256_expected"],
            "weights_sha256_actual": original["weights_sha256_actual"],
            "problems": before["problems"],
            "local_repair_failure": reason,
            "ancestors_tried": [t["directory"] for t in tried],
        })
        report.update({"status": "unhealable",
                       "local_repair_failure": reason,
                       "ancestors_tried": [t["directory"] for t in tried],
                       "verify_after": before})
        return report

    ancestor_identity = _identity(ancestor)
    if not acknowledge_ancestor:
        report.update({
            "status": "ancestor_fallback_requires_acknowledgement",
            "local_repair_failure": reason,
            "ancestor_candidate": ancestor,
            "ancestor_generation": ancestor_identity["generation"],
            "ancestor_weights_sha256":
                ancestor_identity["weights_sha256_expected"],
            "note": "writing the ancestor's weights into this directory "
                    "would change its identity; re-run with "
                    "acknowledge_ancestor=True to accept that explicitly",
            "verify_after": before,
        })
        return report

    # Acknowledged: replace this directory's identity with the
    # ancestor's, completely and verifiably (standard files plus a
    # regenerated healing store), and ledger both identities.
    store_checkpoint_healed(ancestor, directory)
    after = verify_checkpoint(directory)
    _ledger(ledger, "CHECKPOINT_HEALED_FROM_ANCESTOR", {
        "directory": directory,
        "generation": original["generation"],
        "weights_sha256_expected": original["weights_sha256_expected"],
        "weights_sha256_actual": original["weights_sha256_actual"],
        "ancestor_directory": ancestor,
        "ancestor_generation": ancestor_identity["generation"],
        "ancestor_weights_sha256": ancestor_identity["weights_sha256_expected"],
        "local_repair_failure": reason,
        "written_to_disk": True,
        "acknowledged": True,
        "note": "directory now contains the ancestor checkpoint, by "
                "explicit acknowledgement",
    })
    report.update({"status": "healed_from_ancestor", "healed": True,
                   "is_ancestor": True,
                   "ancestor_directory": ancestor,
                   "ancestor_generation": ancestor_identity["generation"],
                   "ancestor_weights_sha256":
                       ancestor_identity["weights_sha256_expected"],
                   "local_repair_failure": reason,
                   "verify_after": after})
    return report
