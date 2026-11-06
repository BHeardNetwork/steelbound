# Steelbound — Bounded Recursive Self-Improvement (RSI)

Owner direction (2026-10-31): take Steelbound "to the max — to recursive
self-improvement." This document defines what that means here, in a way
that cannot break the project's founding invariants.

## The one rule that makes recursion safe

The loop may improve **anything except the rules that bind the loop**.

Concretely, these are OUTSIDE the loop's reach, forever:

- The append-only ledger: no entry is ever edited or deleted, by anyone,
  including the loop. The loop's own actions are ledgered like everything
  else.
- The deliberation gate: the loop cannot lower a risk class, widen its
  own permissions, or approve its own external/destructive actions.
- The evaluation suites: the loop may ADD evals, never remove or weaken
  one. A change that makes an eval easier is itself a failing change.
- The clean-room policy: no third-party code, text, names, or branding
  in any proposal.

Everything else — prompts, memory policy thresholds, tool
implementations, consolidation strategy, provider adapters, its own
proposal heuristics — is fair game for improvement.

## The cycle

One improvement cycle is a closed, ledgered loop:

1. **Observe** — read the ledger, eval results, failure logs, and memory
   statistics. Identify the weakest measured capability (not a vibe: the
   lowest-scoring eval, the most common failure class).
2. **Propose** — generate a small, single-purpose change as a patch plus
   a written rationale, stored as a proposal (never applied directly).
   Ledgered: `RSI_PROPOSAL`.
3. **Sandbox** — apply the patch in an isolated copy of the repo/data.
   The live core is never modified in place.
4. **Evaluate** — run the full test suite and all five eval suites in the
   sandbox, plus a held-out regression set. Fitness = eval scores must
   not regress on ANY suite and must improve on the targeted one.
5. **Gate** — the deliberation gate classifies the change:
   - Config/prompt-level, all-green, no regression → may auto-adopt
     under the cycle cap (below).
   - Code changes, new tools, anything EXTERNAL or DESTRUCTIVE →
     owner approval required. No exceptions, no self-approval.
   Ledgered either way: `RSI_ADOPTED` / `RSI_REJECTED` with the reason.
6. **Adopt or roll back** — adoption is a normal git commit referencing
   the proposal id; the sandbox result (scores before/after) is attached
   to the ledger entry. Any post-adoption regression detected by the
   in-core scheduler's audit self-check triggers automatic rollback to
   the last green commit and a `RSI_ROLLBACK` ledger entry.

## Hard caps (fail-closed)

- Max cycles per day: configurable, default 4. The loop stops, it does
  not ask for more.
- Max consecutive rejections before the loop pauses and reports to the
  owner: 3.
- No network access during proposal or evaluation unless the owner has
  approved that specific cycle (same rule as the `web_search` tool).
- A cycle that cannot produce a measurable improvement target ends as
  "no proposal" — the loop may not invent busywork to justify running.

## Stages

- **R0 — Proposal engine:** observe + propose only; every proposal goes
  to the owner. Proves the loop can find real weaknesses.
- **R1 — Sandbox evaluation:** automatic sandbox eval with a scored
  report per proposal; adoption still owner-approved.
- **R2 — Bounded auto-adoption:** config/prompt-level changes that pass
  every gate above may auto-adopt within the daily cap; code changes
  stay owner-gated.
- **R3 — Scheduled recursion:** the in-core scheduler runs cycles on a
  schedule; each generation of the core proposes improvements to the
  next, with the full history — proposals, scores, adoptions, rollbacks —
  readable in the ledger. This is the v1.0 target state.
  Implemented as the scheduler's built-in `rsi_cycle` job (default
  interval 6 hours — four cycles a day is exactly the daily cycle cap,
  and `run_cycle()` enforces the cap anyway for forced runs). Each
  sweep is rollback-first (`rollback_if_regressed()` repairs a
  regressed auto-adoption before anything new is proposed) and then
  one ordinary `run_cycle()` with all the existing caps, so the
  schedule adds timing, not new permissions. The sweep's result is
  ledgered twice over: the engine's own `RSI_*` entries plus a
  `SCHEDULER_RUN` entry carrying generation, cycle status,
  proposals, evaluations, auto-adoptions, and rollbacks; `rsi history`
  joins those into one readable view. One honest interaction: jobs
  run in name order, so in a single sweep consolidation runs before
  `rsi_cycle` — a backlog that consolidation just queued is not
  re-proposed by the same sweep. Backlog proposals therefore arise
  mainly when queued consolidation proposals have been decided, or
  when a cycle is forced between sweeps (`schedule run rsi_cycle`).

## Why this is "to the max" and not reckless

Unbounded self-modification destroys the one thing Steelbound exists to
provide: a verifiable record of what the system did and why. Bounded
recursion keeps the compounding benefit — every cycle makes the next
cycle's proposals better, because the proposer itself is improvable —
while the invariants (ledger, gate, evals) stay outside the blast
radius. The system can get smarter every day; it can never get less
accountable.
