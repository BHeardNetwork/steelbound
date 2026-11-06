# Steelbound — Offspring Cores: growing a more improved intelligence

Owner direction (2026-11-01): the core "needs to be able to grow
another more improved intelligence from itself."

## What an offspring is

An offspring is a child core instance grown from the parent:

- **Seeded** from the parent's configuration, its improved genome
  (the settings and heuristics the RSI loop has improved), and a
  selected subset of the parent's semantic memory. It does NOT inherit
  the parent's ledger: it starts its own genesis entry that records its
  parent and seed hash, so lineage is provable but histories are never
  merged or rewritten.
- **Bounded harder than the parent**: an offspring's gate permissions
  are a subset of the parent's, never broader. It cannot spawn its own
  offspring until it has been promoted (below). It lives in a sandbox
  data directory, isolated from the parent's live data.
- **Ledgered, all of it**: `OFFSPRING_SPAWN` (with seed hash),
  `OFFSPRING_EVAL` (scores), `OFFSPRING_PROMOTE` / `OFFSPRING_RETIRE`
  are written to the PARENT's ledger, which the offspring cannot touch.

## The generational cycle

1. **Spawn** — the parent (via the RSI machinery, stages R2+) creates a
   child with one deliberate variation: a different memory policy,
   proposal heuristic, provider configuration, or consolidation
   strategy. One variation per generation — improvements must be
   attributable.
2. **Grow** — the child runs in its sandbox against the same task and
   eval battery as the parent, including held-out cases neither has
   seen scored before.
3. **Compare** — head-to-head: the child must beat the parent on the
   targeted capability with **no regression on any eval suite** — the
   same no-regression rule as RSI adoption.
4. **Promote or retire** — promotion is owner-approved, always. A
   promoted child becomes the new production core (its configuration
   and genome adopted; the parent's ledger continues — promotion is a
   ledger entry, not a history swap) and gains the right to spawn the
   next generation. A child that does not beat the parent is retired;
   its eval record stays in the parent's ledger as negative knowledge
   for the proposal engine.
5. **Repeat** — each generation starts from the best proven genome.
   Improvement compounds across generations while every step remains
   auditable to a ledger entry and a score sheet.

## Invariants (outside the loop's reach, as with RSI)

- The append-only ledger, the deliberation gate, the eval suites, and
  the clean-room policy bind offspring exactly as they bind the parent.
- No offspring can approve its own promotion, widen its own
  permissions, or act externally beyond the parent's gate rules.
- Caps: one living offspring at a time by default (configurable by the
  owner); a generation that cannot beat the parent ends the line —
  the loop may not spawn variations endlessly hunting for noise.

## Relationship to RSI stages

Offspring cores build on RSI R2 (bounded auto-adoption) and R3
(scheduled recursion): the RSI loop improves a core *in place*;
offspring grow *successor* cores. Queue placement: after RSI R3,
before the GUI shell.

## Implementation (2026-11-01, build-queue item completed)

`src/steelbound/offspring.py` (OffspringEngine, wired into Core as
`core.offspring`, plus Orchestrator and CLI `offspring`):

- Variation kinds: consolidation thresholds, provider, and
  auto_approve_local_writes (narrowing only — the subset rule is
  validated before anything is built, as is provider buildability,
  so a failed spawn leaves no phantom record or partial sandbox).
- Evaluation verdicts: `promotion_worthy` (targeted gain, no eval
  regression, child chain valid), `no_improvement` (tie, no
  regression), `regression` (any suite drop, a targeted drop, or an
  invalid child chain), `error` (battery or sandbox unusable — e.g.
  a provider variation naming a server that is not reachable).
  Provider/gate variations use total eval passes as the targeted
  metric, so an unmeasured provider gain is never claimed.
- Promotion adopts the stored child genome (not a recomputation
  against the possibly-changed parent): thresholds via the
  consolidator's persisted settings, provider/gate via a live swap
  plus an `adopted_genome` row that Core re-applies on startup.
- Spawn permission is read from the core's own ledger (GENESIS
  without PROMOTED blocks spawning), and generation numbers count
  lineage depth plus attempts, so a promoted child's children
  continue the count. The eval suites accept a core factory
  (`evals.run_all(make_core=...)`) so the head-to-head battery
  genuinely measures each genome; the default path is unchanged.
