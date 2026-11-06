"""In-core scheduler: recurring jobs, driven by explicit run calls, all ledgered.

The scheduler owns three built-in jobs:

- ``consolidation`` — runs the episodic -> semantic consolidation job.
  Its own promotions, proposals, and decisions are already ledgered by
  the consolidator; the scheduler additionally ledgers one
  ``SCHEDULER_RUN`` entry per run with the outcome and counts.
- ``audit`` — the self-check: the ledger hash chain must verify, and
  every memory row must carry a non-empty provenance source. A failed
  audit is ledgered as a failed ``SCHEDULER_RUN``, never hidden, and
  appending that entry does not (and cannot) repair a broken chain —
  the next audit still fails until the underlying problem is fixed.
- ``rsi_cycle`` — RSI R3 scheduled recursion: one bounded RSI cycle
  (see ``rsi.RSIEngine.scheduled_cycle``) every 6 hours by default.
  The cycle's own proposals, scores, adoptions, and rollbacks are
  ledgered by the RSI engine, and the scheduler additionally ledgers
  one ``SCHEDULER_RUN`` entry carrying the cycle's result, so the
  full recursion history is readable in the ledger. The job body is
  bound by ``Core`` after the RSI engine exists (``bind_rsi_cycle``);
  a Scheduler built without that binding reports a failed run for
  ``rsi_cycle`` instead of crashing the sweep, like any other job
  failure. The binding is in-memory only — the job row itself
  persists, and every ``Core`` re-binds it on startup, so a restart
  never leaves a persisted job without its body (unlike
  ``register_job`` custom jobs, whose callables are caller-owned).

Design notes, kept deliberately boring:

- **Poll-driven, not threaded.** ``run_due(now)`` runs every enabled
  job whose interval has elapsed since its last run. The caller drives
  it — a CLI invocation, the local API loop, or a host cron — because
  a Core's SQLite connection is shared, not thread-safe, and a
  background thread would race the caller's own writes. A job that has
  never run is due immediately, so a fresh core's first ``run_due``
  establishes the baseline. ``run_job(name)`` forces one job now
  (due-check bypassed); a disabled job refuses a forced run too.
- **Persisted state.** Jobs live in ``scheduler_jobs`` (name, interval,
  enabled, last run, run count, last status/detail), so a restart
  neither re-runs a just-run job nor forgets that a job is disabled.
- **Failure isolation.** One job raising never stops the others:
  the failure is caught, ledgered as a failed run, and the remaining
  due jobs still run. ``run_due`` therefore never raises for a job
  failure (unknown-job and invalid-interval errors from configuration
  calls do raise, because those are caller mistakes, not job results).
- **Time is injectable.** Every run method takes ``now`` (seconds
  since the epoch, default ``time.time()``) so due logic is exactly
  testable without sleeping.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass

DEFAULT_CONSOLIDATION_INTERVAL = 6 * 3600   # every 6 hours
DEFAULT_AUDIT_INTERVAL = 3600               # every hour
# RSI R3: one scheduled cycle every 6 hours. Four cycles a day is
# exactly the RSI daily cycle cap, so the schedule can never outrun
# the cap — and run_cycle() enforces the cap anyway for forced runs.
DEFAULT_RSI_CYCLE_INTERVAL = 6 * 3600

BUILTIN_JOBS = ("consolidation", "audit", "rsi_cycle")


class JobFailure(Exception):
    """A job failure that still carries its diagnostic detail.

    The audit job raises this so a failed run's ledger entry keeps the
    full check result (chain validity, counts, missing provenance)
    instead of just an error string. Ordinary exceptions from custom
    jobs are still reported as ``{"error": ...}``.
    """

    def __init__(self, detail: dict):
        super().__init__(str(detail))
        self.detail = detail


@dataclass(frozen=True)
class Job:
    name: str
    interval_seconds: float
    enabled: bool
    last_run: float | None
    run_count: int
    last_status: str | None  # None (never run) | "ok" | "failed"
    last_detail: dict

    @property
    def next_run(self) -> float | None:
        """When this job next becomes due; None means due right now."""
        if self.last_run is None:
            return None
        return self.last_run + self.interval_seconds

    def is_due(self, now: float) -> bool:
        return self.enabled and (self.last_run is None
                                 or now >= self.last_run + self.interval_seconds)


@dataclass(frozen=True)
class JobRun:
    job: str
    status: str  # "ok" | "failed"
    detail: dict
    ts: float


class Scheduler:
    def __init__(self, ledger, conn: sqlite3.Connection,
                 consolidation=None):
        """``ledger`` is appended to for every run and config change.

        ``consolidation`` is the Consolidator for the built-in
        consolidation job; ``conn`` is used by the audit job for the
        provenance check over memory rows. Custom jobs registered with
        ``register_job`` need neither.
        """
        self.ledger = ledger
        self.conn = conn
        self.consolidation = consolidation
        self._callables: dict[str, object] = {}
        self._rsi_cycle_func = None  # bound by Core via bind_rsi_cycle
        conn.execute("""CREATE TABLE IF NOT EXISTS scheduler_jobs(
            name TEXT PRIMARY KEY, interval_seconds REAL NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1, last_run REAL,
            run_count INTEGER NOT NULL DEFAULT 0,
            last_status TEXT, last_detail TEXT NOT NULL DEFAULT '{}')""")
        conn.commit()
        for name, interval in (("consolidation", DEFAULT_CONSOLIDATION_INTERVAL),
                               ("audit", DEFAULT_AUDIT_INTERVAL),
                               ("rsi_cycle", DEFAULT_RSI_CYCLE_INTERVAL)):
            conn.execute(
                "INSERT OR IGNORE INTO scheduler_jobs(name, interval_seconds)"
                " VALUES(?,?)", (name, interval))
        conn.commit()

    def bind_rsi_cycle(self, func) -> None:
        """Bind the RSI R3 cycle body (``RSIEngine.scheduled_cycle``).

        Called by ``Core`` once the RSI engine exists; the scheduler
        is built before the engine, so the binding cannot be a
        constructor argument without a circular construction order.
        """
        if not callable(func):
            raise ValueError("rsi_cycle func must be callable")
        self._rsi_cycle_func = func

    # -- job registry ----------------------------------------------------
    def register_job(self, name: str, interval_seconds: float,
                     func, enabled: bool = True) -> None:
        """Register a custom recurring job.

        ``func`` is a zero-argument callable returning a JSON-able
        detail dict; raising means a failed run. Names must be new —
        re-registering a built-in or existing job is a caller error.
        """
        if not name or not isinstance(name, str):
            raise ValueError("job name must be a non-empty string")
        if not interval_seconds or interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if not callable(func):
            raise ValueError("job func must be callable")
        if self._get_row(name) is not None or name in self._callables:
            raise ValueError(f"job {name!r} already exists")
        self._callables[name] = func
        self.conn.execute(
            "INSERT INTO scheduler_jobs(name, interval_seconds, enabled)"
            " VALUES(?,?,?)", (name, float(interval_seconds), 1 if enabled else 0))
        self.conn.commit()

    def jobs(self) -> list[Job]:
        rows = self.conn.execute(
            "SELECT name,interval_seconds,enabled,last_run,run_count,"
            "last_status,last_detail FROM scheduler_jobs ORDER BY name ASC"
        ).fetchall()
        return [Job(r[0], r[1], bool(r[2]), r[3], r[4], r[5],
                    json.loads(r[6])) for r in rows]

    def get_job(self, name: str) -> Job:
        row = self._get_row(name)
        if row is None:
            raise KeyError(f"no scheduler job {name!r}")
        return Job(row[0], row[1], bool(row[2]), row[3], row[4], row[5],
                   json.loads(row[6]))

    def _get_row(self, name: str):
        return self.conn.execute(
            "SELECT name,interval_seconds,enabled,last_run,run_count,"
            "last_status,last_detail FROM scheduler_jobs WHERE name=?",
            (name,)).fetchone()

    def due_jobs(self, now: float | None = None) -> list[Job]:
        now = time.time() if now is None else now
        return [j for j in self.jobs() if j.is_due(now)]

    # -- configuration (ledgered, like everything significant) ------------
    def enable(self, name: str) -> Job:
        return self._set_enabled(name, True)

    def disable(self, name: str) -> Job:
        return self._set_enabled(name, False)

    def _set_enabled(self, name: str, enabled: bool) -> Job:
        job = self.get_job(name)  # KeyError for unknown jobs
        self.conn.execute("UPDATE scheduler_jobs SET enabled=? WHERE name=?",
                          (1 if enabled else 0, name))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("scheduler", "SCHEDULER_CONFIG", {
                "job": name, "change": "enabled" if enabled else "disabled"})
        return self.get_job(name)

    def set_interval(self, name: str, interval_seconds: float) -> Job:
        if not interval_seconds or interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.get_job(name)  # KeyError for unknown jobs
        self.conn.execute(
            "UPDATE scheduler_jobs SET interval_seconds=? WHERE name=?",
            (float(interval_seconds), name))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("scheduler", "SCHEDULER_CONFIG", {
                "job": name, "change": "interval_seconds",
                "interval_seconds": float(interval_seconds)})
        return self.get_job(name)

    # -- the built-in job bodies ------------------------------------------
    def _run_consolidation(self) -> dict:
        if self.consolidation is None:
            raise RuntimeError("consolidation job has no consolidator")
        result = self.consolidation.run()
        return {"promoted": list(result.promoted),
                "queued": list(result.queued),
                "skipped": list(result.skipped)}

    def _run_audit(self) -> dict:
        valid = self.ledger.verify() if self.ledger else False
        entries = self.ledger.count() if self.ledger else 0
        missing_provenance = 0
        memory_rows = 0
        try:
            rows = self.conn.execute(
                "SELECT source FROM memory_items").fetchall()
            memory_rows = len(rows)
            missing_provenance = sum(
                1 for (s,) in rows if s is None or not str(s).strip())
        except sqlite3.Error:
            # No memory table yet: nothing to check provenance on.
            memory_rows = 0
            missing_provenance = 0
        detail = {"valid": valid, "ledger_entries": entries,
                  "memory_items": memory_rows,
                  "missing_provenance": missing_provenance}
        if not valid or missing_provenance:
            detail["error"] = (
                "audit failed: " +
                ("ledger chain does not verify" if not valid else "") +
                ("; " if not valid and missing_provenance else "") +
                (f"{missing_provenance} memory row(s) lack provenance"
                 if missing_provenance else ""))
            raise JobFailure(detail)
        return detail

    def _run_rsi_cycle(self) -> dict:
        # RSI R3 scheduled recursion: rollback-first, then one bounded
        # cycle. The engine ledgers its own RSI_* entries; this detail
        # is what the SCHEDULER_RUN entry carries, so the scheduled
        # sweep itself is part of the readable recursion history.
        if self._rsi_cycle_func is None:
            raise RuntimeError(
                "rsi_cycle job has no RSI engine bound; Core binds it "
                "on startup via bind_rsi_cycle")
        return self._rsi_cycle_func()

    def _job_func(self, name: str):
        if name == "consolidation":
            return self._run_consolidation
        if name == "audit":
            return self._run_audit
        if name == "rsi_cycle":
            return self._run_rsi_cycle
        return self._callables[name]

    # -- running ------------------------------------------------------------
    def run_job(self, name: str, now: float | None = None) -> JobRun:
        """Force-run one job now, regardless of its due time.

        A disabled job refuses (fail closed: disabling must mean
        something); an unknown job raises KeyError.
        """
        now = time.time() if now is None else now
        job = self.get_job(name)
        if not job.enabled:
            raise ValueError(f"scheduler job {name!r} is disabled")
        return self._execute(job, now)

    def run_due(self, now: float | None = None) -> list[JobRun]:
        """Run every enabled job that is due at ``now`` (default: now).

        Never raises for a job failure: each failure is ledgered and
        the remaining due jobs still run.
        """
        now = time.time() if now is None else now
        runs: list[JobRun] = []
        for job in self.due_jobs(now):
            runs.append(self._execute(job, now))
        return runs

    def _execute(self, job: Job, now: float) -> JobRun:
        try:
            detail = self._job_func(job.name)()
            status = "ok"
        except JobFailure as exc:  # failed run, diagnostic detail kept
            detail = exc.detail
            status = "failed"
        except Exception as exc:  # job failure is a result, not a crash
            detail = {"error": f"{type(exc).__name__}: {exc}"}
            status = "failed"
        self.conn.execute(
            "UPDATE scheduler_jobs SET last_run=?, run_count=run_count+1,"
            " last_status=?, last_detail=? WHERE name=?",
            (now, status, json.dumps(detail), job.name))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("scheduler", "SCHEDULER_RUN", {
                "job": job.name, "status": status, "detail": detail})
        return JobRun(job.name, status, detail, now)
