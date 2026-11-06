"""Daily episode: the core's first-person narrative of its own day.

Implements docs/COGNITION.md section 4. Once a day, the core composes
an episode from its own records — what it did (ledger), what it
learned (memory changes), what it improved or proposed (RSI and
offspring entries), what it refused and why (gate denials), and what
it will do tomorrow (scheduler jobs plus pending work) — and stores
it three ways:

- a dated document, ``<data-dir>/workspace/episodes/YYYY-MM-DD.md``,
  which is what the Core App surfaces (its memory view shows the
  matching episodic item, and the document itself lives in the
  workspace, so backup covers it);
- an episodic memory item, source ``daily-episode``, tagged
  ``daily-episode`` so consolidation never promotes a narrative into
  a semantic fact (see ``consolidation.py``);
- one ``DAILY_EPISODE`` ledger entry carrying the counts and the
  per-section ledger entry ids, so every claim in the narrative is
  traceable to a ledger entry.

Nothing here is embellished: the narrative only states what the
records show, lists are capped (the full trace is in the ledger
payload and the document's Trace appendix), and a quiet day gets a
short episode — a day with no records at all gets one sentence.
One episode per day: composing twice for the same day returns the
existing episode with no new ledger entry, memory item, or document
write, so a daily scheduler job that fires twice cannot double the
record. The forward plan is included only for today's episode: it
describes the core's *current* standing work (scheduler jobs,
pending proposals, open unknowns), and attributing today's standing
work to a past day would be a fabrication, so past-day episodes end
without a plan. ``scheduled_episode()`` is a zero-argument body
suitable as a scheduler custom-job body, following the same pattern
as cognition's ``scheduled_self_query``.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

EPISODE_SOURCE = "daily-episode"
EPISODE_TAG = "daily-episode"

_MAX_LISTED = 5      # items named in a narrative paragraph
_MAX_CONTENT = 120   # characters of a memory content quoted in prose


def _local_day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def _short(content: str, limit: int = _MAX_CONTENT) -> str:
    text = " ".join(str(content).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def validate_day(day: str) -> str:
    """Normalize ``day`` to YYYY-MM-DD, rejecting bad dates and future
    days: an episode for a day that has not happened yet would be a
    narrative about nothing, so it is refused rather than composed."""
    if not day or not isinstance(day, str):
        raise ValueError("episode day must be a YYYY-MM-DD date string")
    try:
        parsed = datetime.strptime(day.strip(), "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"invalid episode day {day!r}: expected YYYY-MM-DD")
    normalized = parsed.strftime("%Y-%m-%d")
    if normalized > _today():
        raise ValueError(f"episode day {normalized} is in the future")
    return normalized


@dataclass(frozen=True)
class Episode:
    day: str
    text: str
    document_path: str
    memory_id: int | None
    ledger_entry_id: int | None
    counts: dict
    already_existed: bool = False


@dataclass(frozen=True)
class EpisodeRecord:
    day: str
    document_path: str
    memory_id: int | None
    ledger_entry_id: int | None


class EpisodeEngine:
    """Compose, store, and retrieve daily episodes for a Core."""

    def __init__(self, ledger, conn: sqlite3.Connection, data_dir: str,
                 memory=None, scheduler=None, consolidation=None,
                 rsi=None, cognition=None):
        self.ledger = ledger
        self.conn = conn
        self.data_dir = data_dir
        self.memory = memory
        self.scheduler = scheduler
        self.consolidation = consolidation
        self.rsi = rsi
        self.cognition = cognition
        conn.execute("""CREATE TABLE IF NOT EXISTS episodes(
            day TEXT PRIMARY KEY, text TEXT NOT NULL,
            document_path TEXT NOT NULL, memory_id INTEGER,
            ledger_entry_id INTEGER, ts REAL NOT NULL)""")
        conn.commit()

    # -- retrieval --------------------------------------------------------
    def episodes(self) -> list[EpisodeRecord]:
        rows = self.conn.execute(
            "SELECT day,document_path,memory_id,ledger_entry_id"
            " FROM episodes ORDER BY day ASC").fetchall()
        return [EpisodeRecord(r[0], r[1], r[2], r[3]) for r in rows]

    def get(self, day: str) -> Episode | None:
        day = validate_day(day)
        row = self.conn.execute(
            "SELECT text,document_path,memory_id,ledger_entry_id"
            " FROM episodes WHERE day=?", (day,)).fetchone()
        if row is None:
            return None
        return Episode(day, row[0], row[1], row[2], row[3], {},
                       already_existed=True)

    # -- gathering ----------------------------------------------------------
    def _day_ledger_entries(self, day: str) -> list:
        if self.ledger is None:
            return []
        return [e for e in self.ledger.entries(10000)
                if _local_day(e.ts) == day]

    def _day_memory_rows(self, day: str) -> list[dict]:
        try:
            rows = self.conn.execute(
                "SELECT id,kind,content,source,confidence,ts,tags"
                " FROM memory_items").fetchall()
        except sqlite3.Error:
            return []
        out = []
        for r in rows:
            if _local_day(float(r[5])) != day:
                continue
            try:
                tags = json.loads(r[6] or "[]")
            except Exception:
                tags = []
            out.append({"id": r[0], "kind": r[1], "content": r[2],
                        "source": r[3], "confidence": r[4], "ts": r[5],
                        "tags": tags})
        return out

    def _plan(self) -> tuple[str, dict]:
        """Tomorrow's standing work, from live state only. Returns
        (paragraph, counts); an empty paragraph means no plan source
        was available, and the section is then omitted."""
        parts = []
        counts: dict = {}
        if self.scheduler is not None:
            try:
                jobs = [j for j in self.scheduler.jobs() if j.enabled]
                if jobs:
                    descs = []
                    for j in jobs:
                        secs = j.interval_seconds
                        if secs % 3600 == 0:
                            when = f"every {secs / 3600:g}h"
                        elif secs % 60 == 0:
                            when = f"every {secs / 60:g}m"
                        else:
                            when = f"every {secs:g}s"
                        descs.append(f"{j.name} ({when})")
                    parts.append("scheduled jobs: " + ", ".join(descs))
                    counts["scheduled_jobs"] = len(jobs)
            except Exception:
                pass
        if self.consolidation is not None:
            try:
                pending = len(self.consolidation.pending())
                counts["pending_consolidation"] = pending
                if pending:
                    parts.append(f"{pending} consolidation proposal(s) "
                                 "awaiting an owner decision")
            except Exception:
                pass
        if self.rsi is not None:
            try:
                pending = len(self.rsi.pending())
                counts["pending_rsi"] = pending
                if pending:
                    parts.append(f"{pending} self-improvement proposal(s) "
                                 "awaiting an owner decision")
            except Exception:
                pass
        if self.cognition is not None:
            try:
                open_unknowns = len(self.cognition.unknowns(status="open"))
                counts["open_unknowns"] = open_unknowns
                if open_unknowns:
                    parts.append(f"{open_unknowns} known unknown(s) still open")
            except Exception:
                pass
        if not parts:
            return "", counts
        return ("Tomorrow, I will keep to my standing work: "
                + "; ".join(parts) + "."), counts

    # -- composition --------------------------------------------------------
    def compose(self, day: str | None = None) -> Episode:
        """Compose (or return the existing) episode for ``day``.

        Gathering happens before any write, so the episode's own
        memory item and ledger entry are never counted as the day's
        activity, and a second compose for the same day finds the
        stored row and returns it unchanged.
        """
        day = validate_day(day) if day is not None else _today()
        existing = self.get(day)
        if existing is not None:
            return existing

        entries = self._day_ledger_entries(day)
        mem_rows = self._day_memory_rows(day)
        by_action = Counter(e.action for e in entries)

        paragraphs: list[str] = []
        trace: dict[str, list] = {}
        counts: dict = {"ledger_entries": len(entries)}

        # 1. What I did.
        if not entries:
            paragraphs.append(f"I have no record of activity on {day}.")
            trace["did"] = []
        else:
            did = f"On {day}, I recorded {len(entries)} ledger " \
                  f"{'entry' if len(entries) == 1 else 'entries'}."
            top = by_action.most_common(_MAX_LISTED)
            did += " Most often: " + ", ".join(
                f"{action} ({n})" for action, n in top) + "."
            chat_turns = by_action.get("CHAT_TURN", 0)
            if chat_turns:
                did += f" I held {chat_turns} chat " \
                       f"{'turn' if chat_turns == 1 else 'turns'}."
            paragraphs.append(did)
            trace["did"] = [e.id for e in entries]
            counts["by_action"] = dict(by_action)

        # 2. What I learned.
        promoted_tags = {
            r["id"] for r in mem_rows
            if r["kind"] == "semantic"
            and any(isinstance(t, str)
                    and t.startswith("promoted_from_episodic:")
                    for t in r["tags"])}
        facts = [r for r in mem_rows
                 if r["kind"] == "semantic" and r["id"] not in promoted_tags]
        promotions = [e for e in entries if e.action == "MEMORY_PROMOTE"]
        corrections = [e for e in entries if e.action == "MEMORY_CORRECT"]
        web_results = [e for e in entries if e.action == "WEB_SEARCH_STORED"]
        learned_parts = []
        if facts:
            quoted = "; ".join(f"“{_short(r['content'])}”"
                               for r in facts[:_MAX_LISTED])
            learned_parts.append(
                f"I stored {len(facts)} new "
                f"{'fact' if len(facts) == 1 else 'facts'}: {quoted}")
        if promotions:
            learned_parts.append(
                f"consolidation promoted {len(promotions)} episodic "
                f"{'item' if len(promotions) == 1 else 'items'} "
                "to lasting facts")
        if corrections:
            learned_parts.append(
                f"{len(corrections)} "
                f"{'fact was' if len(corrections) == 1 else 'facts were'} "
                "corrected")
        if web_results:
            queries = sorted({str(e.payload.get("query", "?"))
                              for e in web_results})
            learned_parts.append(
                f"I stored {len(web_results)} web search "
                f"{'result' if len(web_results) == 1 else 'results'} "
                f"as episodic memory (queries: {', '.join(queries)})")
        if learned_parts:
            paragraphs.append("I learned: " + "; ".join(learned_parts) + ".")
            # Traceability must not conflate id spaces: the ledger
            # trace lists ledger entries only (a fact's storage is
            # ledgered as FACT_STORED, carrying the memory item id),
            # and the memory items themselves are listed separately.
            fact_ids = {r["id"] for r in facts}
            fact_entries = [e for e in entries
                            if e.action == "FACT_STORED"
                            and e.payload.get("item_id") in fact_ids]
            trace["learned"] = ([e.id for e in fact_entries]
                                + [e.id for e in promotions]
                                + [e.id for e in corrections]
                                + [e.id for e in web_results])
            if facts:
                trace["learned_memory"] = sorted(fact_ids)
            counts["facts_stored"] = len(facts)
            counts["promotions"] = len(promotions)
            counts["corrections"] = len(corrections)
            counts["web_results"] = len(web_results)

        # 3. What I improved or proposed.
        rsi_entries = [e for e in entries
                       if e.action.startswith("RSI_")]
        offspring_entries = [e for e in entries
                             if e.action.startswith("OFFSPRING_")]
        improved_parts = []
        if rsi_entries:
            rsi_counts = Counter(e.action for e in rsi_entries)
            bits = []
            if rsi_counts.get("RSI_PROPOSAL"):
                targets = sorted({str(e.payload.get("target", "?"))
                                  for e in rsi_entries
                                  if e.action == "RSI_PROPOSAL"})
                bits.append(f"proposed {rsi_counts['RSI_PROPOSAL']} "
                            f"self-improvement(s) ({', '.join(targets)})")
            if rsi_counts.get("RSI_EVALUATED"):
                verdicts = Counter(
                    str(e.payload.get("score", {}).get("verdict", "?"))
                    if isinstance(e.payload.get("score"), dict)
                    else str(e.payload.get("verdict", "?"))
                    for e in rsi_entries if e.action == "RSI_EVALUATED")
                bits.append("evaluated "
                            f"{rsi_counts['RSI_EVALUATED']} in the sandbox "
                            "(" + ", ".join(f"{v} {n}" for v, n in
                                            verdicts.most_common()) + ")")
            for action, verb in (("RSI_AUTO_ADOPTED", "auto-adopted"),
                                 ("RSI_ADOPTED", "adopted"),
                                 ("RSI_REJECTED", "rejected"),
                                 ("RSI_ROLLBACK", "rolled back")):
                if rsi_counts.get(action):
                    bits.append(f"{verb} {rsi_counts[action]}")
            if rsi_counts.get("RSI_CYCLE"):
                bits.append(f"ran {rsi_counts['RSI_CYCLE']} improvement "
                            f"{'cycle' if rsi_counts['RSI_CYCLE'] == 1 else 'cycles'}")
            improved_parts.append("self-improvement: " + "; ".join(bits))
        if offspring_entries:
            off_counts = Counter(e.action for e in offspring_entries)
            bits = []
            for action, verb in (("OFFSPRING_SPAWN", "spawned"),
                                 ("OFFSPRING_EVAL", "evaluated"),
                                 ("OFFSPRING_PROMOTE", "promoted"),
                                 ("OFFSPRING_RETIRE", "retired")):
                if off_counts.get(action):
                    bits.append(f"{verb} {off_counts[action]}")
            verdicts = [str(e.payload.get("verdict", "?"))
                        for e in offspring_entries
                        if e.action == "OFFSPRING_EVAL"]
            if verdicts:
                bits.append("evaluation verdicts: " + ", ".join(verdicts))
            improved_parts.append("offspring: " + "; ".join(bits))
        if improved_parts:
            paragraphs.append("I improved: " + "; ".join(improved_parts) + ".")
            trace["improved"] = [e.id for e in rsi_entries + offspring_entries]
            counts["rsi_entries"] = len(rsi_entries)
            counts["offspring_entries"] = len(offspring_entries)

        # 4. What I refused and why.
        refusals = []
        for e in entries:
            if e.action == "GATE_DECISION" and not e.payload.get("allowed", True):
                refusals.append((e.id, f"{e.payload.get('action', '?')} "
                                       f"(risk {e.payload.get('risk', '?')}, "
                                       "gate denied)"))
            elif e.action == "TOOL_DENIED":
                refusals.append((e.id, f"tool {e.payload.get('tool', '?')} "
                                       f"({e.payload.get('reason', 'denied')})"))
            elif e.action == "DELIBERATION_REQUIRED":
                refusals.append((e.id, f"{e.payload.get('action', '?')} "
                                       f"(risk {e.payload.get('risk', '?')}, "
                                       "no deliberation record)"))
        if refusals:
            listed = "; ".join(desc for _, desc in refusals[:_MAX_LISTED])
            extra = (f", and {len(refusals) - _MAX_LISTED} more"
                     if len(refusals) > _MAX_LISTED else "")
            paragraphs.append(f"I refused {len(refusals)} "
                              f"{'request' if len(refusals) == 1 else 'requests'}: "
                              f"{listed}{extra}.")
            trace["refused"] = [rid for rid, _ in refusals]
            counts["refusals"] = len(refusals)

        # 5. Tomorrow's plan — today only, from live standing work.
        if day == _today():
            plan, plan_counts = self._plan()
            if plan:
                paragraphs.append(plan)
                counts.update(plan_counts)

        text = "\n\n".join(paragraphs)

        # Store: dated document, episodic memory, ledger entry, row.
        episodes_dir = os.path.join(self.data_dir, "workspace", "episodes")
        os.makedirs(episodes_dir, exist_ok=True)
        document_path = os.path.join(episodes_dir, f"{day}.md")
        trace_lines = []
        for section, ids in trace.items():
            if section.endswith("_memory"):
                label = section[:-len("_memory")] + ": memory items"
            else:
                label = section + ": ledger entries"
            if ids:
                trace_lines.append(
                    f"- {label} " + ", ".join(str(i) for i in ids))
            else:
                trace_lines.append(f"- {label}: none")
        document = (f"# Daily episode — {day}\n\n{text}\n\n## Trace\n\n"
                    + "\n".join(trace_lines) + "\n")
        with open(document_path, "w", encoding="utf-8") as f:
            f.write(document)

        memory_id = None
        if self.memory is not None:
            memory_id = self.memory.episode(
                text, source=EPISODE_SOURCE, confidence=1.0,
                tags=[EPISODE_TAG, f"day:{day}"])
        ledger_entry_id = None
        if self.ledger is not None:
            entry = self.ledger.append("episode", "DAILY_EPISODE", {
                "day": day, "document": document_path,
                "memory_id": memory_id, "counts": counts, "trace": trace})
            ledger_entry_id = entry.id
        self.conn.execute(
            "INSERT INTO episodes(day,text,document_path,memory_id,"
            "ledger_entry_id,ts) VALUES(?,?,?,?,?,?)",
            (day, text, document_path, memory_id, ledger_entry_id,
             time.time()))
        self.conn.commit()
        return Episode(day, text, document_path, memory_id,
                       ledger_entry_id, counts, already_existed=False)

    def scheduled_episode(self) -> dict:
        """Zero-argument daily body for a scheduler custom job: compose
        today's episode and report what happened, in JSON-able form."""
        episode = self.compose()
        return {"day": episode.day, "memory_id": episode.memory_id,
                "document": episode.document_path,
                "ledger_entry_id": episode.ledger_entry_id,
                "already_existed": episode.already_existed}
