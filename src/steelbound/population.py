"""Quality-diversity population evolution — Synthesis S3
(docs/SYNTHESIS.md, decision D-016, Gen 3: "diversify").

Single-lineage evolution (offspring cores) keeps one winner: the child
that beats the parent replaces it, and every other line is retired.
That is a narrow search — a variation that is worse overall but best
at one thing is lost, and the next generation can only descend from
the single winner. Quality-diversity fixes that by keeping an
**archive**: offspring are filed by *behaviour* (what kind of core
they are), and each archive cell keeps only the best member ever
filed there. The population is then diverse by construction — many
cells, many strengths — instead of one line hunting one optimum.

Behaviour descriptor (the cell key), derived from the genome alone so
it is stable across runs:

* provider (offline / ollama / openai-compatible),
* consolidation auto-threshold rounded to the nearest 0.05,
* consolidation min-confidence rounded to the nearest 0.05,
* whether local writes are auto-approved.

Quality is the head-to-head score the offspring evaluation already
measures: the child's targeted capability (held-out promotions for
consolidation variations, total eval passes for provider/gate
variations), with total eval passes as the tie-break. A new member
replaces a cell's elite only when it is strictly better on quality,
or equal on quality and better on eval passes; a tie keeps the
incumbent, so the archive never churns on equal members.

Bounds, inherited from offspring cores and enforced here in code:

* Every member is spawned and evaluated through ``OffspringEngine``,
  so the same rules apply: one deliberate variation, permissions a
  subset of the parent's, head-to-head eval under the no-regression
  rule, owner-approved promotion only.
* A member whose verdict is ``regression`` or ``error`` is never
  archived: a cell must never claim a strength its member does not
  have. Such members are retired like any other losing child, and
  the run summary records them as rejected.
* Archiving is not promotion. An elite sits in the archive as a
  candidate; adopting one into the live parent is an explicit,
  owner-approved step (``adopt_elite``), mirroring offspring
  promotion, and it is ledgered as POPULATION_ADOPT.
* A run failure on one variation (bad variation, failed spawn,
  failed eval) is recorded in the summary and does not stop the
  remaining variations — a population sweep must not die on one
  bad member — but a failure is never silently filed as an elite.

Every archive write and every run is ledgered (POPULATION_ARCHIVED /
POPULATION_RUN / POPULATION_ADOPT) in the parent's ledger. No network
is used anywhere in this module.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass

from .offspring import record_adoption

# Verdicts that may occupy an archive cell: the member beat or tied
# the parent with no regression. Regressions and errors are rejected.
ARCHIVABLE_VERDICTS = ("promotion_worthy", "no_improvement")


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _bucket(value: float) -> float:
    """Round a threshold to the nearest 0.05, so small variations of
    the same behaviour share a cell instead of fragmenting it."""
    return round(round(float(value) / 0.05) * 0.05, 2)


def behaviour_descriptor(genome: dict) -> dict:
    """The archive cell of a genome: provider, threshold buckets, and
    the local-write approval flag. Thresholds are bucketed; the exact
    genome is stored with the elite, so bucketing loses nothing."""
    return {
        "provider": genome["provider"],
        "auto_threshold_bucket": _bucket(genome["consolidation_auto_threshold"]),
        "min_confidence_bucket": _bucket(genome["consolidation_min_confidence"]),
        "auto_approve_local_writes": bool(genome["auto_approve_local_writes"]),
    }


def cell_key(genome: dict) -> str:
    """The canonical cell key for a genome (a JSON string, so it is a
    stable primary key and prints readably in the CLI)."""
    return _canonical(behaviour_descriptor(genome))


@dataclass(frozen=True)
class Elite:
    id: int
    cell: str
    offspring_id: int
    generation: int
    variation: dict
    genome: dict
    quality: float
    eval_passes: int
    ts: float

    def descriptor(self) -> dict:
        return json.loads(self.cell)


@dataclass(frozen=True)
class PopulationMemberResult:
    variation: dict
    offspring_id: int | None
    verdict: str  # promotion_worthy | no_improvement | regression | error | failed
    quality: float | None
    archived: bool
    replaced_elite: bool
    detail: str


@dataclass(frozen=True)
class PopulationRun:
    results: tuple[PopulationMemberResult, ...]
    cells_occupied: int
    elites: int

    def archived_count(self) -> int:
        return sum(1 for r in self.results if r.archived)


class PopulationEngine:
    def __init__(self, ledger, conn: sqlite3.Connection, core=None):
        self.ledger = ledger
        self.conn = conn
        self.core = core
        conn.execute("""CREATE TABLE IF NOT EXISTS population_archive(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cell TEXT NOT NULL UNIQUE, offspring_id INTEGER NOT NULL,
            generation INTEGER NOT NULL, variation TEXT NOT NULL,
            genome TEXT NOT NULL, quality REAL NOT NULL,
            eval_passes INTEGER NOT NULL, ts REAL NOT NULL)""")
        conn.commit()

    # -- archive reads -----------------------------------------------------
    def elites(self) -> list[Elite]:
        rows = self.conn.execute(
            "SELECT id,cell,offspring_id,generation,variation,genome,"
            " quality,eval_passes,ts FROM population_archive"
            " ORDER BY cell ASC").fetchall()
        return [Elite(r[0], r[1], r[2], r[3], json.loads(r[4]),
                      json.loads(r[5]), r[6], r[7], r[8]) for r in rows]

    def diversity(self) -> int:
        """Occupied cells: the population's diversity score."""
        return len(self.elites())

    def _elite_for_cell(self, cell: str) -> Elite | None:
        rows = self.conn.execute(
            "SELECT id,cell,offspring_id,generation,variation,genome,"
            " quality,eval_passes,ts FROM population_archive WHERE cell=?",
            (cell,)).fetchall()
        if not rows:
            return None
        r = rows[0]
        return Elite(r[0], r[1], r[2], r[3], json.loads(r[4]),
                     json.loads(r[5]), r[6], r[7], r[8])

    # -- one member ----------------------------------------------------------
    def _archive(self, record, evaluation) -> tuple[bool, bool]:
        """File an evaluated member in its cell. Returns
        (archived, replaced_elite). A regression or error verdict is
        never archived, whatever its score."""
        if evaluation.verdict not in ARCHIVABLE_VERDICTS:
            return False, False
        quality = float(evaluation.child_targeted)
        eval_passes = int(sum(
            s["passed"] for s in evaluation.child_evals.values()))
        cell = cell_key(record.genome)
        incumbent = self._elite_for_cell(cell)
        if incumbent is not None:
            better = (quality > incumbent.quality
                      or (quality == incumbent.quality
                          and eval_passes > incumbent.eval_passes))
            if not better:
                return False, False
        self.conn.execute(
            "INSERT INTO population_archive(cell,offspring_id,generation,"
            " variation,genome,quality,eval_passes,ts) VALUES(?,?,?,?,?,?,?,?)"
            " ON CONFLICT(cell) DO UPDATE SET offspring_id=excluded.offspring_id,"
            " generation=excluded.generation, variation=excluded.variation,"
            " genome=excluded.genome, quality=excluded.quality,"
            " eval_passes=excluded.eval_passes, ts=excluded.ts",
            (cell, record.id, record.generation, _canonical(record.variation),
             _canonical(record.genome), quality, eval_passes, time.time()))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("population", "POPULATION_ARCHIVED", {
                "cell": cell, "offspring_id": record.id,
                "generation": record.generation,
                "variation": record.variation, "quality": quality,
                "eval_passes": eval_passes,
                "replaced_elite": incumbent is not None})
        return True, incumbent is not None

    # -- population sweep ------------------------------------------------------
    def run_population(self, variations: list[dict],
                       heldout_items: list[dict] | None = None) -> PopulationRun:
        """Spawn, evaluate, and archive one member per variation, then
        retire each member (retirement frees the living-offspring cap
        for the next member, and the archive — not the living list —
        is what keeps the population). Per-variation failures are
        recorded in the summary, never filed, and never stop the run."""
        if self.core is None or self.core.offspring is None:
            raise ValueError("population engine has no parent core to spawn from")
        if not isinstance(variations, list) or not variations:
            raise ValueError("run_population requires a non-empty variation list")
        results: list[PopulationMemberResult] = []
        for variation in variations:
            record = None
            try:
                record = self.core.offspring.spawn(variation)
                evaluation = self.core.offspring.evaluate(
                    record.id, heldout_items=heldout_items)
                archived, replaced = self._archive(record, evaluation)
                results.append(PopulationMemberResult(
                    variation, record.id, evaluation.verdict,
                    float(evaluation.child_targeted)
                    if evaluation.child_targeted is not None else None,
                    archived, replaced, evaluation.detail))
            except (ValueError, KeyError, OSError) as exc:
                results.append(PopulationMemberResult(
                    variation, record.id if record else None, "failed",
                    None, False, False, f"{type(exc).__name__}: {exc}"))
            finally:
                # Retire whatever is still living, so the next member
                # can spawn under the living-offspring cap. A member
                # that failed to spawn has nothing to retire.
                if record is not None:
                    try:
                        living = self.core.offspring._get(record.id)
                    except KeyError:
                        living = None
                    if living is not None and living.status in ("spawned", "evaluated"):
                        self.core.offspring.retire(
                            record.id, reason="population sweep: archived or rejected")
        run = PopulationRun(tuple(results), self.diversity(), len(self.elites()))
        if self.ledger:
            self.ledger.append("population", "POPULATION_RUN", {
                "members": [{
                    "variation": r.variation, "offspring_id": r.offspring_id,
                    "verdict": r.verdict, "quality": r.quality,
                    "archived": r.archived} for r in results],
                "cells_occupied": run.cells_occupied,
                "archived": run.archived_count()})
        return run

    # -- owner-approved adoption ----------------------------------------------
    def adopt_elite(self, elite_id: int, approved: bool = False) -> Elite:
        """Adopt an archived elite's genome into the live parent.
        Owner approval must be passed explicitly, mirroring offspring
        promotion; archiving alone never changes the live core."""
        if not approved:
            raise ValueError(
                "adopting an elite requires explicit owner approval (approved=True)")
        rows = self.conn.execute(
            "SELECT id,cell,offspring_id,generation,variation,genome,"
            " quality,eval_passes,ts FROM population_archive WHERE id=?",
            (elite_id,)).fetchall()
        if not rows:
            raise KeyError(f"no archived elite {elite_id}")
        r = rows[0]
        elite = Elite(r[0], r[1], r[2], r[3], json.loads(r[4]),
                      json.loads(r[5]), r[6], r[7], r[8])
        genome = elite.genome
        from .providers import make_provider
        # Build the new provider first: if it cannot be built, nothing
        # has changed yet and adoption aborts cleanly.
        new_provider = make_provider(
            genome["provider"], **dict(genome.get("provider_kwargs") or {}))
        self.core.consolidation.set_thresholds(
            auto_threshold=float(genome["consolidation_auto_threshold"]),
            min_confidence=float(genome["consolidation_min_confidence"]))
        self.core.provider = new_provider
        self.core.settings.provider_kwargs = dict(
            genome.get("provider_kwargs") or {})
        self.core.gate.auto_approve_local_writes = bool(
            genome["auto_approve_local_writes"])
        record_adoption(self.conn, genome)
        if self.ledger:
            self.ledger.append("population", "POPULATION_ADOPT", {
                "elite_id": elite.id, "cell": elite.cell,
                "offspring_id": elite.offspring_id,
                "generation": elite.generation, "genome": genome})
        return elite
