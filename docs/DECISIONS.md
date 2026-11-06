# Decisions Log — Steelbound
Decisions are recorded here as they are made, with the work-ledger entry
that records them. The log is a human-readable index; the append-only
WORK_LEDGER.md remains the authoritative record. Decisions are never
rewritten — a superseded decision gets a new entry that names the one it
replaces.

## D-001 — License: our own Steelbound Source-Available License v1.0
- Date: 2026-10-30. Ledger: 0002. Owner-delegated ("you pick").
- The license is drafted in the repo LICENSE file. Counsel review is
  required before any public or commercial distribution.

## D-002 — Name: Steelbound (package `steelbound`)
- Date: 2026-10-30. Ledger: 0002. Owner-delegated ("you pick").
- Name check found no AI-software product under this name. "Verity Core"
  and "IronLedger" were rejected because both are already in use by
  AI/software products.

## D-003 — Provider: local-first adapter
- Date: 2026-10-30. Ledger: 0002. Owner-delegated ("you pick").
- Offline provider is deterministic and the default; Ollama covers local
  models; an OpenAI-compatible provider is optional, key via environment.

## D-004 — Clean-room policy
- Date: 2026-10-30. Ledger: 0001, 0003. Binding.
- Steelbound is built from scratch. No code, text, names, or branding
  from any third-party project is copied into it. Third-party material
  the owner shares is reference only and stays fenced outside the code.

## D-005 — Ledger first: everything significant, append-only
- Date: 2026-10-30. Ledger: 0003, 0005.
- The ledger is hash-chained and enforced by database triggers that abort
  UPDATE/DELETE on ledger entries. Corrections supersede, never erase —
  in memory and in this log. Around-the-clock build runs must end with
  tests + evals green and a new ledger entry appended.

## D-006 — Deliberation gate fails closed
- Date: 2026-10-30. Ledger: 0003.
- Risk classes: read, write_local, external, destructive, forbidden.
  Anything not explicitly approved is denied, and denials are ledgered.

## D-007 — Consolidation thresholds: auto-promote at >= 0.9, queue at 0.5–0.89
- Date: 2026-10-31. Ledger: 0006.
- Episodic items at confidence >= 0.9 auto-promote to semantic items,
  carrying original provenance; the episodic item is superseded, never
  erased. Candidates at 0.5–0.89 are queued for owner approval. Chat
  transcripts, duplicates of existing semantic items, and items below
  0.5 are skipped. Runs are idempotent.

## D-008 — Web search is EXTERNAL: gated before any network call
- Date: 2026-10-31. Ledger: 0007.
- The `web_search` tool requires explicit owner approval; without it the
  gate denies before any network call. Results are stored as episodic
  memory at confidence 0.8 — external claims, never semantic facts —
  with the result URL as provenance. The fetcher is pluggable; the
  default is a zero-dependency stdlib HTML endpoint fetcher.

## D-009 — Export/import: verify-then-write, re-chain onto the target
- Date: 2026-10-31. Ledger: 0008.
- A bundle is one JSON object (format "steelbound-bundle", version 1).
  Import verifies the ledger chain and the memory digest before writing
  anything, and re-chains imported entries onto the target tip, so a
  merge never breaks the target's chain. Memory ids are remapped so
  correction history and pending proposals survive a round trip.

## D-010 — Local HTTP API: loopback only, single-threaded, gate unchanged
- Date: 2026-10-31. Ledger: 0010.
- The stdlib server binds 127.0.0.1 only — a non-loopback host is refused
  before a socket opens — and handles requests sequentially. Tool runs
  go through the same gate as the CLI, so approval rules are identical.

## D-011 — Backup is an exact copy; restore verifies before writing
- Date: 2026-11-01. Ledger: 0011.
- A backup is a single zip (database via SQLite's online backup API, all
  workspace files, manifest with hashes and a memory digest). Restore
  verifies format, hashes, chain, and digest before writing anything,
  refuses a non-empty target unless forced, and appends no entry of its
  own — unlike import, which is a merge.

## D-012 — Scheduler is poll-driven, not threaded
- Date: 2026-11-01. Ledger: 0012.
- Built-in jobs: consolidation every 6 hours, audit self-check every
  hour. The caller (CLI, API loop, or host cron) drives `run_due()`,
  because the Core's SQLite connection is shared, not thread-safe, and
  a background thread would race the caller's writes. Job state
  persists across restarts; one job failing never stops the others.

## D-013 — Bounded recursive self-improvement (direction for v1.0)
- Date: 2026-10-31. Ledger: 0009. Owner direction: "to the max — to
  recursive self-improvement", adopted in bounded form.
- The improvement loop may improve anything EXCEPT the rules that bind
  it: the append-only ledger, the deliberation gate, the evaluation
  suites, and the clean-room policy are permanently outside its reach.
  Cycle: observe → propose (ledgered RSI_PROPOSAL) → sandbox →
  evaluate (no regression, improvement on the targeted suite) → gate
  (config/prompt-level all-green may auto-adopt within a daily cap;
  code changes and anything EXTERNAL or DESTRUCTIVE stay owner-gated)
  → adopt as a git commit or auto-rollback on regression (RSI_ROLLBACK).
  Scheduled recursion (RSI R3) is the v1.0 target state.

## D-014 — v0.2.0 release scope (this review)
- Date: 2026-11-01. Ledger: 0013.
- v0.2.0 ships everything built since v0.1.0: consolidation, web
  search, export/import, local HTTP API, backup & restore, and the
  in-core scheduler. The v0.2 review's demo is the recorded end-to-end
  run (CLI `demo` + full test/eval suites, both green) attached to
  ledger entry 0013; an interactive owner walkthrough remains available
  on request. The v0.2.0 tag is local-only: nothing is pushed to GitHub
  or shared externally.

## D-015 — Steelbound's own model supersedes the provider adapter
- Date: 2026-11-02. Ledger: 0024. Owner direction: "we supposed to
  built it", then "grow one on your local and then test it
  thoroughly", then "evolve it so we never have to use a third party
  model ever again". Supersedes the provider half of D-003.
- Steelbound's generation comes from its own trained model:
  Gen 0 (`model.py`) is a clean-room neural language model trained
  from a seeded random init, locally, on Steelbound's own corpus
  only, with a hashed, tamper-checked checkpoint. The `steelbound`
  provider serves the promoted generation. Ollama and
  OpenAI-compatible adapters remain only as transitional scaffolding
  so old configs do not break; nothing in the own-model path uses
  them, and no future generation may depend on them.
- Evolution (`evolution.py`, docs/MODEL.md): each generation starts
  from the previous generation's own weights, trains on Steelbound's
  own data with one recorded variation, and is judged head-to-head
  (held-out perplexity + generation probes) under the no-regression
  rule. Promotion is an owner-path action and is ledgered with
  parent/child weight hashes, back to Gen 0's random init.

## D-016 — Breakthrough synthesis adopted; compression is the scoreboard
- Date: 2026-11-02. Ledger: 0025. Owner direction: take all the major
  breakthroughs (computing/optimization, intelligence, recursive
  improvement, self-organization, DNA/genetics, self-introspection,
  compression/encoding/decoding, inference, self-healing, evolution,
  algorithmic methods, human knowledge) and combine them into
  Steelbound — "learned and studied and found the hidden insights
  and improved relentlessly", and "what are you waiting for".
- Design: docs/SYNTHESIS.md. Unifying principle: prediction =
  compression = intelligence. Each breakthrough maps to what
  Steelbound already holds at systems level (ledger = DNA/provenance
  lineage, genome + offspring = evolution, RSI = recursive
  improvement, cognition = self-introspection, backup/rollback =
  self-healing, memory consolidation = compression) and to the
  model-level gap it must close next.
- Conflicts resolved in the design: human knowledge enters only as
  provenance-tracked DATA (public-domain, authored, owner-approved,
  or teacher-authored and marked as such) — never third-party model
  weights or outputs; teacher-first curriculum (teacher-authored
  lessons, fading to a self-curriculum) is the Gen 1 knowledge
  source; the bloodline rule (the parent model produces each child,
  one recorded variation, head-to-head no-regression eval, promotion
  makes the child the new parent) governs every generation; the
  bounded invariants (ledger, gate, eval suites, clean-room policy)
  stay permanently outside the loop's reach.
- Roadmap: Gen 1 = attention + learned compression/encoding +
  knowledge curriculum; Gen 2 = predictive/active inference (S2);
  Gen 3 = quality-diversity population evolution + program synthesis
  + self-healing weights (S3). S1 (compression core: learned BPE
  encoding in compression.py, bits-per-token as a first-class
  evolution metric) is built with this decision; S2/S3 are queued.

## D-017 — Ecosystem foundation; version 1.0.0
- Date: 2026-11-05. Ledger: 0044–0049. Owner direction (voice,
  2026-11-05): further Steelbound's software, intelligence, and
  infrastructure and build the foundation of the ecosystem —
  developed as far forward as possible, now; publish to a new
  GitHub repo and save to Drive (the push itself awaits the owner's
  GitHub sign-in; the release is staged and Drive-saved with this
  decision).
- The ecosystem is a generalization of parts Steelbound already
  had, designed clean-room from our own docs: the offspring subset
  rule becomes the ecosystem-wide member rule (D-017a: a member's
  permissions are a subset of its host core's ceiling; `forbidden`
  is grantable to no one; over-ceiling manifests are refused, never
  silently clamped). Design: docs/ECOSYSTEM.md.
- Built under this decision: the ecosystem registry/runtime
  (`ecosystem.py`, ECOSYSTEM_REGISTER/ECOSYSTEM_RETIRE), the
  toolpack plugin SDK (`toolpacks.py`, docs/TOOLPACKS.md —
  statically validated packs, risks refused never downgraded,
  packs are ecosystem members), self-healing weights
  (`healing.py` — the Gen 3 item left unbuilt by S3, entry 0029:
  sharded redundant checkpoints, hash-verified repair, ancestor
  fallback always reported as the ancestor), deployment
  infrastructure (`deploy/`, `config/`), and the silent-growth
  Gen 1 experiment (entry 0047).
- Versioning: this release is **1.0.0** — the production v1.0
  foundation the goal names. The core was already at its v1.0
  target state for the bounded-RSI line (R3, entry 0013 line);
  1.0.0 marks the point where the core plus its ecosystem
  foundation (members, toolpacks, offspring fleet, self-healing,
  deployment) ship as one verified whole: full test suite and all
  five eval suites green, tagged locally `v1.0.0`.
- Unchanged by this decision: Gen 0 remains the live model. The
  silent-growth Gen 1 child passed the no-regression gate and is
  a promotion CANDIDATE (entry 0047); promotion remains an
  owner-path action under D-015/D-016 and does not happen by
  virtue of this release.
