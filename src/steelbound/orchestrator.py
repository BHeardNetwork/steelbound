from __future__ import annotations
import hashlib
from .config import Core

SYSTEM = ("You are Steelbound, a local-first AI core. Be truthful over flattering. "
          "Say what you verified, what you remember (with provenance), and what you do not know. "
          "Never fabricate sources, facts, or results.")

def _digest(text: str) -> str: return hashlib.sha256(text.encode()).hexdigest()[:16]

class Orchestrator:
    def __init__(self, core: Core): self.core = core

    def chat(self, user_text: str) -> str:
        c = self.core
        c.memory.work_add(user_text)
        c.memory.episode(f"user: {user_text}", source="conversation")
        recalled = c.memory.recall(user_text, limit=5)
        mem_block = ""
        if recalled:
            lines = [f"- [{i.kind} via {i.source}] {i.content}" for i in recalled]
            mem_block = "MEMORY (with provenance):\n" + "\n".join(lines)
        messages = []
        if mem_block: messages.append({"role": "system", "content": mem_block})
        messages.append({"role": "user", "content": user_text})
        reply = c.provider.complete(SYSTEM, messages)
        c.memory.episode(f"assistant: {reply}", source="conversation")
        c.ledger.append("orchestrator", "CHAT_TURN", {"user_hash": _digest(user_text), "reply_hash": _digest(reply)})
        return reply

    def remember_fact(self, content: str, source: str = "owner") -> int:
        item_id = self.core.memory.fact(content, source=source)
        self.core.ledger.append("orchestrator", "FACT_STORED", {"item_id": item_id})
        return item_id

    def run_tool(self, name: str, args: dict | None = None, approved: bool = False):
        return self.core.tools.run(name, args or {}, approved=approved)

    def web_search(self, query: str, max_results: int = 5, approved: bool = False) -> dict:
        """Gated web search: approval is enforced by the tool registry,
        then each result is stored as episodic memory with its URL as
        provenance. Returns the results and the stored memory item ids."""
        dicts = self.run_tool("web_search", {"query": query, "max_results": max_results},
                              approved=approved)
        from .websearch import SearchResult
        results = [SearchResult(**d) for d in dicts]
        stored = self.core.websearch.store_results(
            self.core.memory, self.core.ledger, query, results)
        return {"results": dicts, "stored_item_ids": stored}

    def consolidate(self):
        """Run the episodic -> semantic consolidation job."""
        return self.core.consolidation.run()

    def schedule_run_due(self, now=None):
        """Run every scheduler job that is due; each run is ledgered."""
        return self.core.scheduler.run_due(now=now)

    def schedule_status(self):
        """Current state of every scheduler job."""
        return self.core.scheduler.jobs()

    def rsi_observe(self):
        """RSI R0: measure evals, ledger, memory, and scheduler state."""
        return self.core.rsi.observe()

    def rsi_propose(self, observation=None):
        """RSI R0: one ledgered proposal per measured weakness."""
        return self.core.rsi.propose(observation)

    def rsi_evaluate(self, proposal_id: int, repo_root=None):
        """RSI R1: sandbox-score a proposal; live repo/data untouched."""
        return self.core.rsi.evaluate(proposal_id, repo_root=repo_root)

    def rsi_decide(self, proposal_id: int, decision: str):
        """Owner gate for RSI proposals; approval never applies a patch."""
        return self.core.rsi.decide(proposal_id, decision)

    def rsi_auto_adopt(self, proposal_id: int):
        """RSI R2: bounded auto-adoption of an all-green config patch."""
        return self.core.rsi.auto_adopt(proposal_id)

    def rsi_rollback(self, proposal_id: int, reason: str = "manual"):
        """RSI R2: roll back one auto-adoption to its previous value."""
        return self.core.rsi.rollback(proposal_id, reason=reason)

    def rsi_rollback_if_regressed(self):
        """RSI R2: automatic rollback when the audit checks fail."""
        return self.core.rsi.rollback_if_regressed()

    def rsi_cycle(self, repo_root=None):
        """One bounded RSI cycle: observe, propose, sandbox-evaluate."""
        return self.core.rsi.run_cycle(repo_root=repo_root)

    def rsi_scheduled_cycle(self, repo_root=None):
        """RSI R3: one scheduled recursion sweep (rollback-first cycle)."""
        return self.core.rsi.scheduled_cycle(repo_root=repo_root)

    def rsi_history(self):
        """RSI R3: full proposal/score/adoption history from the ledger."""
        return self.core.rsi.history()

    def offspring_spawn(self, variation, sandbox_dir=None, seed_limit=None,
                        seed_min_confidence=0.0):
        """Offspring: grow a child core with one deliberate variation."""
        return self.core.offspring.spawn(variation, sandbox_dir=sandbox_dir,
                                         seed_limit=seed_limit,
                                         seed_min_confidence=seed_min_confidence)

    def offspring_evaluate(self, offspring_id: int, heldout_items=None):
        """Offspring: head-to-head eval vs the parent, no-regression rule."""
        return self.core.offspring.evaluate(offspring_id,
                                            heldout_items=heldout_items)

    def offspring_promote(self, offspring_id: int, approved: bool = False):
        """Offspring: owner-approved adoption of a promotion_worthy child."""
        return self.core.offspring.promote(offspring_id, approved=approved)

    def offspring_retire(self, offspring_id: int, reason: str = ""):
        """Offspring: retire a child that cannot beat the parent."""
        return self.core.offspring.retire(offspring_id, reason=reason)

    def offspring_list(self):
        """Offspring: every spawned child and its status."""
        return self.core.offspring.offspring()

    def cognition_self_query(self, action, goal=None, where=None, risk=None,
                             approved=False, authority="owner", schedule=None):
        """Cognition: the six-question self-query, ledgered as SELF_QUERY."""
        from .gate import Risk
        return self.core.cognition.self_query(
            action, goal=goal, where=where,
            risk=Risk.READ if risk is None else risk,
            approved=approved, authority=authority, schedule=schedule)

    def cognition_deliberate(self, action, options, evidence, chosen, reason,
                             risk=None):
        """Cognition: record a deliberation (options, evidence, choice, reason)."""
        from .gate import Risk
        return self.core.cognition.deliberate(
            action, options, evidence, chosen, reason,
            risk=Risk.EXTERNAL if risk is None else risk)

    def cognition_request_approval(self, action, risk, deliberation_id=None,
                                   approved=False):
        """Cognition: approval path for external/destructive actions —
        refused before the gate without a matching deliberation record."""
        return self.core.cognition.request_approval(
            action, risk, deliberation_id=deliberation_id, approved=approved)

    def cognition_record_unknown(self, question, context=""):
        """Cognition: record a question that could not be grounded."""
        return self.core.cognition.record_unknown(question, context=context)

    def cognition_unknowns(self, status=None):
        """Cognition: known unknowns, optionally filtered by status."""
        return self.core.cognition.unknowns(status=status)

    def cognition_resolve_unknown(self, unknown_id, resolution_memory_id):
        """Cognition: resolve an open unknown against a memory item."""
        return self.core.cognition.resolve_unknown(unknown_id, resolution_memory_id)

    def cognition_metacognition(self):
        """Cognition: eval scores, calibration, gate denials, open unknowns."""
        return self.core.cognition.metacognition()

    def cognition_comprehension_check(self):
        """Cognition: re-derive promoted facts from their sources."""
        return self.core.cognition.comprehension_check()

    def cognition_scheduled_self_query(self):
        """Cognition: the scheduled self-check (scheduler custom-job body)."""
        return self.core.cognition.scheduled_self_query()

    def prediction_predict(self, stream, context=""):
        """Predictive inference: predict the next outcome on a stream."""
        return self.core.prediction.predict(stream, context=context)

    def prediction_observe(self, stream, actual, prediction_id=None,
                           context=""):
        """Predictive inference: record an outcome and its surprise."""
        return self.core.prediction.observe(
            stream, actual, prediction_id=prediction_id, context=context)

    def prediction_surprise_for(self, stream, actual):
        """Predictive inference: surprise of an outcome, no recording."""
        return self.core.prediction.surprise_for(stream, actual)

    def prediction_summary(self, stream=None):
        """Predictive inference: surprise aggregates, per stream or all."""
        return self.core.prediction.summary(stream)

    def prediction_learning_queue(self, limit=10):
        """Predictive inference: unlearned outcomes, most surprising first."""
        return self.core.prediction.learning_queue(limit=limit)

    def prediction_mark_learned(self, outcome_id):
        """Predictive inference: mark one queued outcome learned."""
        return self.core.prediction.mark_learned(outcome_id)

    def prediction_promote_unknowns(self, limit=None, threshold_bits=None):
        """Predictive inference: surprising outcomes -> known unknowns."""
        return self.core.prediction.promote_to_unknowns(
            limit=limit, threshold_bits=threshold_bits)

    def prediction_scheduled_review(self):
        """Predictive inference: scheduled surprise review (custom-job body)."""
        return self.core.prediction.scheduled_surprise_review()

    def episode_compose(self, day=None):
        """Daily episode: compose (or return the existing) episode for a day."""
        return self.core.episode.compose(day)

    def episode_show(self, day=None):
        """Daily episode: the stored episode for a day, or None."""
        from .episode import _today, validate_day
        return self.core.episode.get(validate_day(day) if day else _today())

    def episode_list(self):
        """Daily episode: every stored episode, oldest first."""
        return self.core.episode.episodes()

    def episode_scheduled(self):
        """Daily episode: the scheduled daily body (scheduler custom-job body)."""
        return self.core.episode.scheduled_episode()

    def population_run(self, variations, heldout_items=None):
        """Synthesis S3: spawn, evaluate, and archive one population
        member per variation; regressions and errors are never filed."""
        return self.core.population.run_population(
            variations, heldout_items=heldout_items)

    def population_elites(self):
        """Synthesis S3: every archived elite, one per behaviour cell."""
        return self.core.population.elites()

    def population_diversity(self):
        """Synthesis S3: occupied archive cells — the diversity score."""
        return self.core.population.diversity()

    def population_adopt(self, elite_id: int, approved: bool = False):
        """Synthesis S3: owner-approved adoption of an archived elite."""
        return self.core.population.adopt_elite(elite_id, approved=approved)

    def synthesis_propose(self, name, description, code, cases,
                          timeout_seconds=10.0):
        """Synthesis S3: validate, sandbox-test, and keep-or-reject a
        candidate tool; kept only when every case passes."""
        return self.core.synthesis.propose(
            name, description, code, cases, timeout_seconds=timeout_seconds)

    def synthesis_list(self):
        """Synthesis S3: every kept synthesized tool."""
        return self.core.synthesis.synthesized()

    def synthesis_remove(self, name):
        """Synthesis S3: delete a kept tool, live and persisted."""
        return self.core.synthesis.remove(name)

    def ecosystem_register(self, manifest, approved: bool = False):
        """Ecosystem: register a member (app, offspring, toolpack, or
        service) from its manifest; permissions must be a subset of
        the core's ceiling, and registration is gate-approved."""
        return self.core.ecosystem.register(manifest, approved=approved)

    def ecosystem_list(self, status=None):
        """Ecosystem: every registered member, optionally by status."""
        return self.core.ecosystem.list_members(status=status)

    def ecosystem_inspect(self, name):
        """Ecosystem: one registered member by name."""
        return self.core.ecosystem.inspect(name)

    def ecosystem_retire(self, name, reason: str = "", approved: bool = False):
        """Ecosystem: retire a member; it can never act again.
        Destructive at the gate, so owner approval is required."""
        return self.core.ecosystem.retire(name, reason=reason, approved=approved)
