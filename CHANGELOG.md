# Changelog
## [Unreleased]
- (none yet)

## [1.0.1] — 2026-11-07
- **Gen 1 is the live model.** The silent-growth Gen 1 child that
  1.0.0 recorded as a promotion candidate was promoted on
  2026-11-06, owner-approved after a six-point verification program
  passed in full. This release ships the promoted checkpoint in
  `src/steelbound/model_data/gen1_silent/` (380,870 params, weights
  SHA-256 20d038f52cc300559729cff021717884993392470fbb358b554478da5c759b3d)
  and the pointer that makes it live, `model_data/current.json`.
  Gen 0 remains in `model_data/gen0/` as the bloodline ancestor and
  rollback: deleting `current.json` restores it as the live model.
- Gen 2 experiment (`gen2.py`, `training/run_gen2_exposure.py`):
  the full 471-sequence teacher set at ≈25% share, testing the
  teacher-exposure diagnosis. The child improved on its parent
  (perplexity −8.09% in-run, core probes 10/10) but was REFUSED on
  the design's predefined criteria (teacher recall 22/78 vs the
  required 30; substance guard 9/46 vs the required 10). Gen 1
  remains live; the refused candidate is not shipped in this
  release. Logic (0/8) and reasoning (0/8) remain open problems,
  recorded honestly, and are the Gen 3 design questions.
- Live browser demo (`docs/`, served by GitHub Pages): the real
  Gen 1 checkpoint runs entirely in the visitor's browser via a
  JavaScript port of the model's own inference, parity-verified
  against the Python model (exact match on all 10 core probes).
- Deployment note, unchanged from 1.0.0: the Docker deployment
  files are verified statically only; a container run has not been
  performed and is not claimed.

## [1.0.0] — 2026-11-05
- Ecosystem foundation (decision D-017, design docs/ECOSYSTEM.md):
  Steelbound is now a core plus an ecosystem of members, all of it
  ledgered and gated the same way.
- Ecosystem core (`ecosystem.py`): a registry and runtime for
  members — apps, offspring, toolpacks, services — each with a
  manifest whose permissions are enforced as a subset of the core's
  ceiling (`forbidden` is grantable to no one; over-ceiling
  manifests are refused, never silently narrowed). Registration is
  a gated local write, retirement is destructive and
  owner-approved, retired members cannot act. CLI:
  `steelbound ecosystem {register|list|inspect|retire}`.
- Toolpack SDK (`toolpacks.py`, docs/TOOLPACKS.md): external tool
  packs (a directory with `pack.json` + Python modules) join
  through static validation in the synthesis discipline, declared
  risks refused rather than downgraded (default loader ceiling
  write_local; `forbidden` never loadable), gated load/unload,
  persistence across restarts, and registration as ecosystem
  members. Ships with an example pack (text statistics).
- Self-healing weights (`healing.py`, the Gen 3 item recorded
  unbuilt in entry 0029): sharded redundant checkpoint storage
  with per-shard hashes; corruption is detected, repaired from
  redundancy after hash verification, or healed from the nearest
  verified bloodline ancestor — always reported as the ancestor.
  Unhealable checkpoints are refused loudly. The stock loader
  still refuses corruption; healing is an explicit layer above it.
- Intelligence: the silent-growth Gen 1 experiment
  (`silentgrowth.py`, `training/run_silent_growth.py`) tested the
  multiverse diagnosis (growth shock) directly — zero-init growth,
  sub-epoch snapshot selection, ≈5% teacher share, A2 selection.
  The child passed the no-regression gate (perplexity 45.71 vs
  Gen 0's 57.71 on the same frozen ground, core probes 10/10) and
  is recorded as a promotion CANDIDATE only. Gen 0 remains the
  live model; promotion is the owner's decision.
- Deployment infrastructure (`deploy/`, `config/`): Dockerfile +
  docker-compose for the loopback API service, a systemd unit and
  install script, and a settings env template. Packaging verified
  end-to-end: `pip install .` in a clean venv, then evals + demo +
  serve all pass from the installed wheel.

## [0.3.0] — 2026-11-02
- Steelbound's own model, Gen 0 (`model.py`, decision D-015): a
  feed-forward neural language model written clean-room for this
  project — token embeddings over a 6-token context, one tanh hidden
  layer, softmax output; backpropagation and Adam implemented in the
  same file on numpy (numerical gradient check in the test suite).
  Trained from a seeded random init, locally, on Steelbound's own
  corpus only (authored curriculum + this repo's own prose; corpus
  hash in the checkpoint manifest). Checkpoint ships in
  `src/steelbound/model_data/gen0/` with SHA-256 hashes verified on
  load — a tampered checkpoint is refused.
- The `steelbound` provider (`providers.py`): generation with
  Steelbound's own trained weights — no network, no Ollama, no
  third-party model. Ollama/OpenAI-compatible adapters are now
  documented as transitional D-003 scaffolding only.
- Model evolution (`evolution.py`, `docs/MODEL.md`): generations grow
  from the parent generation's own weights on Steelbound's own data,
  one recorded variation each, judged head-to-head (held-out
  perplexity + generation probes) under the no-regression rule;
  promotion writes `model_data/current.json`, which the provider
  follows. This is the standing path by which the model improves
  without ever depending on a third-party model.

## [0.2.0] — 2026-11-01
- Scheduler inside the core (`scheduler.py`, CLI `schedule`): two
  built-in recurring jobs — memory consolidation every 6 hours and an
  audit self-check every hour — both ledgered as SCHEDULER_RUN, plus
  `register_job()` for custom recurring jobs. Deliberately poll-driven,
  not threaded (the Core's SQLite connection is shared, not
  thread-safe): the caller drives `run_due()`. Job state (interval,
  enabled, last run, run count, last status/detail) persists across
  restarts; one job failing never stops the others; a disabled job
  refuses even a forced run.
- Backup & restore (`backup.py`, CLI `backup` / `restore`): one-command
  zip snapshot of the data dir (consistent database copy via SQLite's
  online backup API, so a live core can be backed up; all workspace
  files; manifest with SHA-256 hashes, ledger count, and a memory
  digest). Restore verifies format, every hash, the ledger chain, and
  the memory digest before writing anything, refuses a non-empty
  target unless `--force`, and re-verifies the restored chain; a
  tampered ledger is rejected even if the manifest hashes were
  rewritten to match. Backup creation is ledgered as BACKUP_CREATED;
  restore itself is an exact copy (no entry appended by the restore).
- Local HTTP API (`server.py`, CLI `serve`): stdlib-only, bound to
  127.0.0.1 only (non-loopback hosts refused), single-threaded; POST
  /chat, GET /recall, GET /ledger/verify, GET /tools, POST /tools (alias
  /tools/run). Tool runs pass through the existing gate unchanged, so
  approvals work exactly as in the CLI. Core's SQLite connection is now
  opened with `check_same_thread=False` with a documented
  serialize-access contract, so the server thread can use it.
- Export/import (`exporter.py`, CLI `export` / `import`): memory +
  ledger export to a single JSON bundle (format "steelbound-bundle",
  version 1). Import verifies the ledger chain and the memory digest
  before writing anything, re-chains imported entries onto the target
  tip, and remaps memory ids so correction history and pending
  consolidation proposals survive a round trip. Tampered bundles are
  rejected with nothing written.
- Web search tool (`websearch.py`, CLI `search`): registered as
  EXTERNAL risk, so the gate denies it without explicit owner approval
  before any network call. Pluggable fetcher (stdlib HTML endpoint
  fetcher by default, endpoint configurable). Results are stored as
  episodic memory at confidence 0.8 with the result URL as provenance,
  never as semantic facts.
- Memory consolidation job (`consolidation.py`, CLI `consolidate`):
  episodic → semantic promotions; auto-promote at confidence ≥ 0.9,
  queue lower-confidence candidates for owner approval, ledger every
  promotion and decision; chat transcripts, duplicates of existing
  semantic items, and items below 0.5 confidence are skipped; runs are
  idempotent. CLI: `consolidate`, `consolidate pending`,
  `consolidate approve|reject <id>`.
- Decisions log (`docs/DECISIONS.md`): every decision since genesis,
  indexed to its work-ledger entry, including the bounded recursive
  self-improvement direction (docs/SELF_IMPROVEMENT.md) that sets the
  v1.0 target state.
- Release gate (`tests/test_release.py`): the version in
  `pyproject.toml` must match `steelbound.__version__`, and the
  changelog must carry a section for that version with nothing left
  pending under [Unreleased].

## [0.1.0] — 2026-10-30
- First production foundation: append-only hash-chained ledger enforced by
  database triggers; layered memory with provenance and supersede-corrections;
  deliberation gate (fail closed); sandboxed tool registry; model adapter with
  offline (default), Ollama, and OpenAI-compatible providers; CLI; five
  evaluation suites wired as the release gate; CI.
