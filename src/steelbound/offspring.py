"""Offspring cores: growing a successor core from the parent's genome.

Per docs/OFFSPRING.md. An offspring is a child core seeded from the
parent's configuration, its RSI-improved genome (the consolidation
thresholds the loop has improved, the provider, and the gate's local-
write approval flag), and a selected subset of the parent's semantic
memory. It does NOT inherit the parent's ledger: it starts its own
ledger, whose OFFSPRING_GENESIS entry records the parent's ledger tip
and the seed hash, so lineage is provable but histories are never
merged or rewritten.

The generational cycle, enforced here in code:

1. ``spawn(variation)`` — one deliberate variation per generation
   (a consolidation threshold, the provider, or a narrowing of the
   local-write approval flag). The variation is validated before
   anything is built: thresholds must stay ordered, and an offspring's
   permissions must be a SUBSET of the parent's — a parent that does
   not auto-approve local writes cannot spawn a child that does.
   The child lives in its own sandbox data directory, isolated from
   the parent's live data. Spawning is ledgered in the parent as
   OFFSPRING_SPAWN. Caps: one living offspring at a time by default
   (``max_living``, owner-configurable), and spawning is always an
   explicit call — there is no automatic respawn after a retirement,
   so the loop cannot hunt noise by spawning variations endlessly.
2. ``evaluate(offspring_id, heldout_items)`` — head-to-head against
   the parent on the same battery: all five eval suites run in cores
   built from each genome (so a variation that changes behaviour is
   actually measured), plus a consolidation task on a held-out
   episodic set (caller-supplied, or the parent's current episodic
   items when none is supplied — never scored as a battery before).
   The child must beat the parent on the targeted capability with NO
   regression on any eval suite — the same no-regression rule as RSI.
   A smoke check opens the child's own sandbox and verifies its
   ledger chain. Ledgered in the parent as OFFSPRING_EVAL.
3. ``promote(offspring_id, approved=True)`` — owner-approved, always:
   approval must be passed explicitly, and only an evaluated offspring
   whose verdict is ``promotion_worthy`` may promote. Promotion
   adopts the child's genome into the live parent (thresholds
   persisted via the consolidator, provider swapped on the live core,
   gate flag set) AND persists the provider/gate part in an
   ``adopted_genome`` row, so a restart re-applies it — the parent's
   ledger continues; promotion is a ledger entry (OFFSPRING_PROMOTE
   in the parent, OFFSPRING_PROMOTED in the child), not a history
   swap. A promoted child gains the right to spawn the next
   generation (see below).
4. ``retire(offspring_id)`` — a child that cannot beat the parent is
   retired; its eval record stays in the parent's ledger as negative
   knowledge, and its sandbox directory is left on disk for
   inspection, never silently deleted.

Spawn permission is ledger-grounded, not flag-grounded: a core whose
own ledger holds OFFSPRING_GENESIS but no OFFSPRING_PROMOTED entry is
an unpromoted offspring, and its engine refuses to spawn. A root core
has neither entry and may spawn; a promoted offspring has both and
may spawn the next generation. Generation numbers count lineage
depth plus attempts: a new offspring's generation is one more than
both the spawning core's own generation and its highest generation
so far, so generations strictly increase and a promoted child's
children continue the count.

What offspring can never do, as with RSI: touch the parent's ledger
(the child writes only its own), widen its own permissions, approve
its own promotion, or act externally beyond the parent's gate rules.
The eval suites, the ledger, and the gate bind offspring exactly as
they bind the parent. No network is used anywhere in this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass

PROVIDER_NAMES = ("offline", "ollama", "openai-compatible")
VARIATION_KINDS = ("consolidation_auto_threshold",
                   "consolidation_min_confidence",
                   "auto_approve_local_writes",
                   "provider")
# Statuses that count as "living" for the one-living-offspring cap.
LIVING_STATUSES = ("spawned", "evaluated")
DEFAULT_MAX_LIVING = 1


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def capture_genome(core) -> dict:
    """The parent's genome: settings the RSI loop may have improved,
    plus the provider and gate flag that bound what the core may do."""
    return {
        "provider": core.provider.name,
        "provider_kwargs": dict(core.settings.provider_kwargs or {}),
        "auto_approve_local_writes": bool(core.gate.auto_approve_local_writes),
        "consolidation_auto_threshold": float(core.consolidation.auto_threshold),
        "consolidation_min_confidence": float(core.consolidation.min_confidence),
    }


def validate_variation(variation: dict, parent_genome: dict) -> None:
    """Reject a variation the loop must never spawn. Raises ValueError."""
    if not isinstance(variation, dict):
        raise ValueError("variation must be a JSON object")
    kind = variation.get("kind")
    if kind not in VARIATION_KINDS:
        raise ValueError(f"unknown variation kind {kind!r}")
    if "value" not in variation:
        raise ValueError("variation needs a 'value'")
    value = variation["value"]
    if kind == "consolidation_auto_threshold":
        if not _is_number(value) or not 0.0 < float(value) <= 1.0:
            raise ValueError("consolidation_auto_threshold must be in (0, 1]")
        if float(value) < parent_genome["consolidation_min_confidence"]:
            raise ValueError("auto threshold would fall below min confidence")
    elif kind == "consolidation_min_confidence":
        if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("consolidation_min_confidence must be in [0, 1]")
        if float(value) > parent_genome["consolidation_auto_threshold"]:
            raise ValueError("min confidence would rise above auto threshold")
    elif kind == "auto_approve_local_writes":
        if not isinstance(value, bool):
            raise ValueError("auto_approve_local_writes must be true or false")
        # Permissions are a subset of the parent's, never broader.
        if value and not parent_genome["auto_approve_local_writes"]:
            raise ValueError(
                "offspring permissions must be a subset of the parent's: "
                "the parent does not auto-approve local writes")
    elif kind == "provider":
        if value not in PROVIDER_NAMES:
            raise ValueError(f"unknown provider {value!r}")


def apply_variation(genome: dict, variation: dict) -> dict:
    """The child genome: the parent genome with exactly one field changed."""
    validate_variation(variation, genome)
    child = dict(genome)
    child["provider_kwargs"] = dict(genome.get("provider_kwargs") or {})
    kind, value = variation["kind"], variation["value"]
    if kind == "provider":
        child["provider"] = value
        # Provider-specific kwargs belong to the old provider; a swapped
        # provider starts from defaults unless the caller stored new ones.
        if value != genome["provider"]:
            child["provider_kwargs"] = {}
    else:
        child[kind] = value
    return child


def seed_hash(genome: dict, variation: dict, seeded: list[dict]) -> str:
    """SHA-256 over the spawn inputs, so lineage is provable from the
    parent's OFFSPRING_SPAWN entry and the child's OFFSPRING_GENESIS."""
    payload = {"genome": genome, "variation": variation, "seeded": seeded}
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def record_adoption(conn: sqlite3.Connection, genome: dict) -> None:
    """Persist a promoted genome's provider/gate part (thresholds
    already persist via the consolidator). One row, id 1; Core applies
    it on startup, so promotion survives a restart."""
    conn.execute("""CREATE TABLE IF NOT EXISTS adopted_genome(
        id INTEGER PRIMARY KEY CHECK(id = 1), provider TEXT NOT NULL,
        auto_approve_local_writes INTEGER NOT NULL,
        provider_kwargs TEXT NOT NULL DEFAULT '{}')""")
    conn.execute(
        "INSERT INTO adopted_genome(id, provider, auto_approve_local_writes,"
        " provider_kwargs) VALUES(1, ?, ?, ?)"
        " ON CONFLICT(id) DO UPDATE SET provider=excluded.provider,"
        " auto_approve_local_writes=excluded.auto_approve_local_writes,"
        " provider_kwargs=excluded.provider_kwargs",
        (genome["provider"], int(bool(genome["auto_approve_local_writes"])),
         _canonical(genome.get("provider_kwargs") or {})))
    conn.commit()


def load_adoption(conn: sqlite3.Connection) -> dict | None:
    """The persisted adopted genome, or None. Raises ValueError on a
    row that names an unknown provider — fail closed, never silently
    run under a genome the ledger does not recognise."""
    conn.execute("""CREATE TABLE IF NOT EXISTS adopted_genome(
        id INTEGER PRIMARY KEY CHECK(id = 1), provider TEXT NOT NULL,
        auto_approve_local_writes INTEGER NOT NULL,
        provider_kwargs TEXT NOT NULL DEFAULT '{}')""")
    row = conn.execute(
        "SELECT provider, auto_approve_local_writes, provider_kwargs"
        " FROM adopted_genome WHERE id=1").fetchone()
    if row is None:
        return None
    provider, auto_approve, kwargs_json = row
    if provider not in PROVIDER_NAMES:
        raise ValueError(f"adopted genome names unknown provider {provider!r}")
    try:
        kwargs = json.loads(kwargs_json) if kwargs_json else {}
    except json.JSONDecodeError:
        raise ValueError("adopted genome has unreadable provider kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError("adopted genome provider kwargs must be an object")
    return {"provider": provider,
            "auto_approve_local_writes": bool(auto_approve),
            "provider_kwargs": kwargs}


@dataclass(frozen=True)
class OffspringRecord:
    id: int
    generation: int
    variation: dict
    genome: dict  # the child's genome, as spawned
    seed_hash: str
    sandbox_dir: str
    status: str  # spawned | evaluated | promoted | retired
    ts: float
    score: dict | None = None


@dataclass(frozen=True)
class OffspringEvaluation:
    verdict: str  # promotion_worthy | no_improvement | regression | error
    detail: str
    parent_evals: dict
    child_evals: dict
    parent_targeted: float | None
    child_targeted: float | None
    smoke: str  # ok | regression | error

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict, "detail": self.detail,
            "parent_evals": self.parent_evals, "child_evals": self.child_evals,
            "parent_targeted": self.parent_targeted,
            "child_targeted": self.child_targeted, "smoke": self.smoke,
        }


def _genome_core(tmpdir: str, genome: dict):
    """A throwaway core built from a genome (used for the head-to-head
    battery, so the live parent and the child sandbox are untouched)."""
    from .config import Settings, Core
    core = Core(Settings(data_dir=tmpdir, provider=genome["provider"],
                         provider_kwargs=dict(genome.get("provider_kwargs") or {}),
                         auto_approve_local_writes=bool(
                             genome["auto_approve_local_writes"])))
    core.consolidation.set_thresholds(
        auto_threshold=float(genome["consolidation_auto_threshold"]),
        min_confidence=float(genome["consolidation_min_confidence"]))
    return core


class OffspringEngine:
    def __init__(self, ledger, conn: sqlite3.Connection, core=None,
                 max_living: int = DEFAULT_MAX_LIVING):
        if max_living < 1:
            raise ValueError("max_living must be positive")
        self.ledger = ledger
        self.conn = conn
        self.core = core
        self.max_living = max_living
        conn.execute("""CREATE TABLE IF NOT EXISTS offspring(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation INTEGER NOT NULL, variation TEXT NOT NULL,
            genome TEXT NOT NULL, seed_hash TEXT NOT NULL,
            sandbox_dir TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'spawned', ts REAL NOT NULL,
            score TEXT)""")
        conn.commit()

    # -- records ---------------------------------------------------------
    def offspring(self, status: str | None = None) -> list[OffspringRecord]:
        if status is None:
            rows = self.conn.execute(
                "SELECT id,generation,variation,genome,seed_hash,sandbox_dir,"
                " status,ts,score FROM offspring ORDER BY id ASC").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id,generation,variation,genome,seed_hash,sandbox_dir,"
                " status,ts,score FROM offspring WHERE status=? ORDER BY id ASC",
                (status,)).fetchall()
        return [OffspringRecord(r[0], r[1], json.loads(r[2]), json.loads(r[3]),
                                r[4], r[5], r[6], r[7],
                                json.loads(r[8]) if r[8] else None)
                for r in rows]

    def _get(self, offspring_id: int) -> OffspringRecord:
        rows = [r for r in self.offspring() if r.id == offspring_id]
        if not rows:
            raise KeyError(f"no offspring {offspring_id}")
        return rows[0]

    def _set_status(self, offspring_id: int, status: str,
                    score: dict | None = None) -> None:
        if score is None:
            self.conn.execute("UPDATE offspring SET status=? WHERE id=?",
                              (status, offspring_id))
        else:
            self.conn.execute(
                "UPDATE offspring SET status=?, score=? WHERE id=?",
                (status, _canonical(score), offspring_id))
        self.conn.commit()

    # -- spawn permission --------------------------------------------------
    def _own_generation(self) -> int:
        """This core's generation: 0 for a root core, else the
        generation recorded in its own OFFSPRING_GENESIS entry."""
        if self.ledger is None:
            return 0
        for entry in self.ledger.entries(10000):
            if entry.action == "OFFSPRING_GENESIS":
                return int(entry.payload.get("generation", 0))
        return 0

    def _may_spawn(self) -> None:
        """Ledger-grounded spawn permission: an unpromoted offspring
        (GENESIS without PROMOTED in its own ledger) may not spawn."""
        if self.ledger is None:
            return
        genesis = promoted = False
        for entry in self.ledger.entries(10000):
            if entry.action == "OFFSPRING_GENESIS":
                genesis = True
            elif entry.action == "OFFSPRING_PROMOTED":
                promoted = True
        if genesis and not promoted:
            raise ValueError(
                "this core is an unpromoted offspring; only a promoted "
                "core may spawn the next generation")

    def _living_count(self) -> int:
        return sum(1 for r in self.offspring()
                   if r.status in LIVING_STATUSES)

    def _parent_tip(self) -> str | None:
        if self.ledger is None:
            return None
        entries = self.ledger.entries(10000)
        return entries[-1].entry_hash if entries else None

    # -- 1. spawn ------------------------------------------------------------
    def spawn(self, variation: dict, sandbox_dir: str | None = None,
              seed_limit: int | None = None,
              seed_min_confidence: float = 0.0) -> OffspringRecord:
        """Grow a child core with exactly one deliberate variation.
        The child is built first; only a successful build is recorded
        in the parent's table and ledger, so a failed spawn leaves no
        phantom offspring behind."""
        if self.core is None:
            raise ValueError("offspring engine has no parent core to spawn from")
        self._may_spawn()
        if self._living_count() >= self.max_living:
            raise ValueError(
                f"living offspring cap ({self.max_living}) reached: "
                "evaluate, promote, or retire the living offspring first")
        if seed_limit is not None and (not isinstance(seed_limit, int)
                                       or seed_limit < 1):
            raise ValueError("seed_limit must be a positive integer")
        if not _is_number(seed_min_confidence) \
                or not 0.0 <= float(seed_min_confidence) <= 1.0:
            raise ValueError("seed_min_confidence must be in [0, 1]")
        parent_genome = capture_genome(self.core)
        validate_variation(variation, parent_genome)
        child_genome = apply_variation(parent_genome, variation)
        # A provider variation must be buildable (e.g. an OpenAI-
        # compatible provider needs model + base_url) before anything
        # is created, or a failed spawn would leave a partial sandbox.
        from .providers import make_provider
        try:
            make_provider(child_genome["provider"],
                          **dict(child_genome.get("provider_kwargs") or {}))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"child provider cannot be built: {exc}")
        generation = max(self._own_generation() + 1,
                         max([r.generation for r in self.offspring()],
                             default=0) + 1)
        sandbox = sandbox_dir or os.path.join(
            self.core.settings.data_dir, "offspring", f"gen-{generation}")
        if os.path.exists(os.path.join(sandbox, "steelbound.db")):
            raise ValueError(f"sandbox data dir already holds a core: {sandbox}")
        # Selected semantic memory: the parent's durable facts at or
        # above the seed floor, oldest first, truncated to the limit.
        selected = [i for i in self.core.memory.all_active(kind="semantic")
                    if i.confidence >= float(seed_min_confidence)]
        selected.sort(key=lambda i: i.id)
        if seed_limit is not None:
            selected = selected[:seed_limit]
        seeded = [{"content": i.content, "source": i.source,
                   "confidence": i.confidence} for i in selected]
        digest = seed_hash(parent_genome, variation, seeded)
        parent_tip = self._parent_tip()
        from .config import Settings, Core
        child = Core(Settings(
            data_dir=sandbox, provider=child_genome["provider"],
            provider_kwargs=dict(child_genome.get("provider_kwargs") or {}),
            auto_approve_local_writes=bool(
                child_genome["auto_approve_local_writes"])))
        try:
            child.ledger.append("offspring", "OFFSPRING_GENESIS", {
                "parent_ledger_tip": parent_tip, "seed_hash": digest,
                "generation": generation, "variation": variation})
            child.consolidation.set_thresholds(
                auto_threshold=float(child_genome["consolidation_auto_threshold"]),
                min_confidence=float(child_genome["consolidation_min_confidence"]))
            for item in seeded:
                child.memory.fact(item["content"], source=item["source"],
                                  confidence=item["confidence"],
                                  tags=["offspring_seed"])
        finally:
            child.close()
        cur = self.conn.execute(
            "INSERT INTO offspring(generation,variation,genome,seed_hash,"
            " sandbox_dir,ts) VALUES(?,?,?,?,?,?)",
            (generation, _canonical(variation), _canonical(child_genome),
             digest, sandbox, time.time()))
        self.conn.commit()
        record = OffspringRecord(cur.lastrowid, generation, variation,
                                 child_genome, digest, sandbox, "spawned",
                                 time.time())
        if self.ledger:
            self.ledger.append("offspring", "OFFSPRING_SPAWN", {
                "offspring_id": record.id, "generation": generation,
                "variation": variation, "seed_hash": digest,
                "sandbox_dir": sandbox, "parent_ledger_tip": parent_tip,
                "seeded_count": len(seeded)})
        return record

    # -- 2. head-to-head evaluation ------------------------------------------
    def _eval_battery(self, genome: dict) -> dict:
        """All five eval suites in cores built from this genome.
        Raises on any suite failure, so a broken genome scores as an
        evaluation error, never as a silent pass."""
        from . import evals as evals_mod

        def factory(tmpdir: str):
            return _genome_core(tmpdir, genome)

        results = evals_mod.run_all(make_core=factory)
        return {name: {"passed": p, "total": t}
                for name, p, t, _ in results}

    def _consolidation_task(self, genome: dict, items: list[dict]) -> int:
        """Promotions a genome earns on the held-out episodic set,
        measured in a throwaway core — the live cores are untouched."""
        with tempfile.TemporaryDirectory() as tmp:
            core = _genome_core(os.path.join(tmp, "core"), genome)
            try:
                for item in items:
                    core.memory.episode(
                        item["content"], source=item.get("source", "heldout"),
                        confidence=float(item.get("confidence", 0.8)))
                return len(core.consolidation.run().promoted)
            finally:
                core.close()

    def _smoke_child(self, record: OffspringRecord) -> tuple[str, str]:
        """Open the child's own sandbox: its chain must verify.
        Returns (status, detail) with status ok | regression | error."""
        if not os.path.isdir(record.sandbox_dir):
            return "error", "sandbox data dir is missing"
        from .config import Settings, Core
        try:
            child = Core(Settings(data_dir=record.sandbox_dir))
        except Exception as exc:  # a child core that cannot open is broken
            return "error", f"child core failed to open: {exc}"
        try:
            if not child.ledger.verify():
                return "regression", "child ledger chain does not verify"
            seeded = len(child.memory.all_active(kind="semantic"))
            return "ok", (f"child chain valid; {seeded} seeded semantic "
                          "item(s) present")
        finally:
            child.close()

    def evaluate(self, offspring_id: int,
                 heldout_items: list[dict] | None = None) -> OffspringEvaluation:
        """Head-to-head: the child must beat the parent on the targeted
        capability with no regression on any eval suite. The targeted
        capability is promotions on the held-out set for consolidation
        variations, and total eval passes for provider/gate variations.
        A provider swap or a permission narrowing earns no targeted
        gain from the battery when the provider is reachable (it ties
        and retires); when the provider is unreachable the battery
        fails and the verdict is ``error`` — an unmeasured improvement
        is never claimed)."""
        record = self._get(offspring_id)
        if record.status not in ("spawned", "evaluated"):
            raise ValueError(
                f"offspring {offspring_id} is {record.status}, not evaluable")
        if heldout_items is None:
            heldout = [{"content": i.content, "source": i.source,
                        "confidence": i.confidence}
                       for i in self.core.memory.all_active(kind="episodic")]
        else:
            heldout = []
            for item in heldout_items:
                if not isinstance(item, dict) or not item.get("content"):
                    raise ValueError("held-out items need a 'content' string")
                heldout.append({"content": item["content"],
                                "source": item.get("source", "heldout"),
                                "confidence": float(item.get("confidence", 0.8))})
        parent_genome = capture_genome(self.core)
        parent_evals: dict = {}
        child_evals: dict = {}
        error = None
        try:
            parent_evals = self._eval_battery(parent_genome)
        except Exception as exc:
            error = f"parent eval battery failed: {exc}"
        if error is None:
            try:
                child_evals = self._eval_battery(record.genome)
            except Exception as exc:
                error = f"child eval battery failed: {exc}"
        smoke, smoke_detail = self._smoke_child(record)
        if error is not None:
            verdict, detail = "error", error
            parent_targeted = child_targeted = None
        else:
            if record.variation["kind"] in (
                    "consolidation_auto_threshold",
                    "consolidation_min_confidence"):
                parent_targeted = float(
                    self._consolidation_task(parent_genome, heldout))
                child_targeted = float(
                    self._consolidation_task(record.genome, heldout))
            else:
                parent_targeted = float(sum(
                    s["passed"] for s in parent_evals.values()))
                child_targeted = float(sum(
                    s["passed"] for s in child_evals.values()))
            regressed_suite = next(
                (name for name in parent_evals
                 if child_evals.get(name, {"passed": 0})["passed"]
                 < parent_evals[name]["passed"]), None)
            if regressed_suite is not None:
                verdict = "regression"
                detail = (f"eval suite {regressed_suite} regressed: "
                          f"parent {parent_evals[regressed_suite]['passed']}/"
                          f"{parent_evals[regressed_suite]['total']}, child "
                          f"{child_evals[regressed_suite]['passed']}/"
                          f"{child_evals[regressed_suite]['total']}")
            elif smoke == "regression":
                verdict, detail = "regression", smoke_detail
            elif smoke == "error":
                verdict, detail = "error", smoke_detail
            elif record.variation["kind"] in (
                    "consolidation_auto_threshold",
                    "consolidation_min_confidence") \
                    and child_targeted < parent_targeted:
                verdict = "regression"
                detail = (f"targeted capability regressed: child promoted "
                          f"{child_targeted:.0f} held-out items, parent "
                          f"{parent_targeted:.0f}")
            elif child_targeted > parent_targeted:
                verdict = "promotion_worthy"
                detail = (f"child beats parent on the targeted capability "
                          f"({child_targeted:.0f} vs {parent_targeted:.0f}), "
                          "no eval regression; " + smoke_detail)
            else:
                verdict = "no_improvement"
                detail = (f"child ties parent on the targeted capability "
                          f"({child_targeted:.0f} vs {parent_targeted:.0f}), "
                          "no eval regression; " + smoke_detail)
        evaluation = OffspringEvaluation(
            verdict, detail, parent_evals, child_evals,
            parent_targeted, child_targeted, smoke)
        self._set_status(offspring_id, "evaluated", evaluation.to_dict())
        if self.ledger:
            self.ledger.append("offspring", "OFFSPRING_EVAL", {
                "offspring_id": offspring_id, "generation": record.generation,
                "verdict": verdict, "detail": detail,
                "parent_evals": parent_evals, "child_evals": child_evals,
                "parent_targeted": parent_targeted,
                "child_targeted": child_targeted})
        return evaluation

    # -- 3. owner-approved promotion -----------------------------------------
    def promote(self, offspring_id: int, approved: bool = False) -> OffspringRecord:
        """Adopt a promotion_worthy child's genome into the parent.
        Owner approval must be passed explicitly; anything else raises
        and changes nothing. The child ledger gains OFFSPRING_PROMOTED
        (which is what later lets that core spawn), and the parent
        ledger gains OFFSPRING_PROMOTE while continuing unbroken."""
        record = self._get(offspring_id)
        if not approved:
            raise ValueError(
                "promotion requires explicit owner approval (approved=True)")
        if record.status != "evaluated":
            raise ValueError(
                f"offspring {offspring_id} is {record.status}; only an "
                "evaluated offspring may promote")
        if (record.score or {}).get("verdict") != "promotion_worthy":
            raise ValueError(
                f"offspring {offspring_id} verdict is "
                f"{(record.score or {}).get('verdict')!r}, not "
                "'promotion_worthy'; a child that cannot beat the parent "
                "ends the line")
        if not os.path.isdir(record.sandbox_dir):
            raise ValueError("child sandbox data dir is missing; cannot promote")
        genome = record.genome
        from .config import Settings, Core
        from .providers import make_provider
        # Build the new provider first: if it cannot be built, nothing
        # has changed yet and promotion aborts cleanly.
        new_provider = make_provider(
            genome["provider"], **dict(genome.get("provider_kwargs") or {}))
        child = Core(Settings(data_dir=record.sandbox_dir))
        try:
            child.ledger.append("offspring", "OFFSPRING_PROMOTED", {
                "offspring_id": offspring_id, "generation": record.generation,
                "adopted_genome": genome})
        finally:
            child.close()
        self.core.consolidation.set_thresholds(
            auto_threshold=float(genome["consolidation_auto_threshold"]),
            min_confidence=float(genome["consolidation_min_confidence"]))
        self.core.provider = new_provider
        self.core.settings.provider_kwargs = dict(
            genome.get("provider_kwargs") or {})
        self.core.gate.auto_approve_local_writes = bool(
            genome["auto_approve_local_writes"])
        record_adoption(self.conn, genome)
        self._set_status(offspring_id, "promoted")
        if self.ledger:
            self.ledger.append("offspring", "OFFSPRING_PROMOTE", {
                "offspring_id": offspring_id, "generation": record.generation,
                "adopted_genome": genome})
        return OffspringRecord(record.id, record.generation, record.variation,
                               record.genome, record.seed_hash,
                               record.sandbox_dir, "promoted", record.ts,
                               record.score)

    # -- 4. retirement ---------------------------------------------------------
    def retire(self, offspring_id: int, reason: str = "") -> OffspringRecord:
        """Retire a child that cannot beat the parent. Its eval record
        stays in the parent's ledger; its sandbox is left on disk."""
        record = self._get(offspring_id)
        if record.status not in ("spawned", "evaluated"):
            raise ValueError(
                f"offspring {offspring_id} is {record.status}, not retirable")
        self._set_status(offspring_id, "retired")
        if self.ledger:
            self.ledger.append("offspring", "OFFSPRING_RETIRE", {
                "offspring_id": offspring_id, "generation": record.generation,
                "reason": reason,
                "verdict": (record.score or {}).get("verdict")})
        return OffspringRecord(record.id, record.generation, record.variation,
                               record.genome, record.seed_hash,
                               record.sandbox_dir, "retired", record.ts,
                               record.score)
