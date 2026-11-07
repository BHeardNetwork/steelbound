# Steelbound — Ecosystem Architecture

Decision D-017 (2026-11-05, owner direction): Steelbound is no longer
a single core with features; it is the foundation of an ecosystem —
one sovereign core, and everything around it a *member* whose powers
are a subset of the core's, whose actions pass the same gate, and
whose existence is written into the same append-only ledger.

This document describes the ecosystem as built at v1.0.0. It is our
own design, developed from Steelbound's existing parts (the ledger,
the deliberation gate, offspring cores, program synthesis) — nothing
here is adapted from any third-party project.

## The shape

```
                        ┌─────────────────────────────┐
                        │        STEELBOUND CORE      │
                        │  ledger · gate · memory     │
                        │  cognition · prediction     │
                        │  RSI (bounded) · scheduler  │
                        │  own model (Gen 0, live)    │
                        └──────┬───────────────┬──────┘
              registers/gates  │               │  spawns/evaluates
        ┌──────────────────────┴──┐         ┌──┴───────────────────┐
        │   ECOSYSTEM MEMBERS     │         │   OFFSPRING FLEET    │
        │   apps · services       │         │   child cores, one   │
        │   toolpacks · offspring │         │   variation each,    │
        │   (subset permissions)  │         │   no-regression eval │
        └─────────────────────────┘         └──────────────────────┘
```

The core is the only sovereign. Members do not hold their own
ledgers, cannot widen their own permissions, and cannot act after
retirement. The ledger is the spine: every registration, load,
retirement, spawn, evaluation, and repair is an entry in it.

## The core (recap)

- **Ledger** — hash-chained, append-only, enforced by the database
  itself. Every significant action in the ecosystem lands here.
- **Gate** — risk classes read / write_local / external /
  destructive / forbidden. Fails closed; denials are ledgered.
- **Memory** — episodic → semantic consolidation with provenance
  and fixed promotion thresholds (D-007).
- **Own model** — a clean-room neural language model grown locally
  from random init (D-015). Gen 0 is live. The bloodline advances
  one owner-approved generation at a time (see "Intelligence").
- **Bounded RSI** — observe → propose → sandbox → evaluate → gate →
  adopt/rollback (D-013). The ledger, gate, eval suites, and
  clean-room policy are permanently outside its reach.

## Ecosystem members (`ecosystem.py`)

`EcosystemEngine` is the registry and runtime boundary for
everything that attaches to a core:

- **Manifest** — name (unique forever, even after retirement),
  kind (`app | offspring | toolpack | service`), capabilities,
  permissions, description, metadata.
- **Subset rule (hard invariant)** — a member's permissions must be
  a subset of the core's ceiling. The default ceiling is every risk
  class except `forbidden`, which no configuration can grant. A
  manifest that asks beyond the ceiling is *refused*, never
  silently clamped: what is stored and ledgered is always exactly
  what was asked for and approved. A bounded host (e.g. an
  offspring core) passes a narrower ceiling, and the same rule
  applies against it.
- **Gating** — registration is a gated local write
  (`ecosystem_register`); retirement is destructive
  (`ecosystem_retire`) and always needs explicit approval.
  Validation runs before the gate, so an invalid manifest consumes
  no gate decision and changes no state.
- **Ledger kinds** — `ECOSYSTEM_REGISTER`, `ECOSYSTEM_RETIRE`.
- **Act guard** — `can_act` / `require_active`: retired or unknown
  members fail closed. A retired member keeps its history and its
  name; it simply cannot act, ever again.
- **Surfaces** — `Core.ecosystem`, Orchestrator methods, and
  `steelbound ecosystem {register|list|inspect|retire}`.

## Toolpacks — the plugin SDK (`toolpacks.py`, docs/TOOLPACKS.md)

A toolpack is how third-party capability joins the ecosystem
without joining the codebase:

- **Format** — a directory with a `pack.json` manifest (name,
  version, tools with entrypoints and declared risks) plus Python
  modules.
- **Validation before import** — every module is statically
  validated (the same AST discipline as program synthesis:
  allowlisted imports, no `os`/`sys`/`subprocess`, no `open`,
  no dunder traversal) before anything is imported. One bad module
  refuses the whole pack; nothing registers partially.
- **Risk semantics** — declared risks are *refused, never
  downgraded*. The default loader ceiling is `write_local`; an
  above-ceiling tool loads only through an elevated loader on an
  approved load, at its true declared risk. `forbidden` is never
  loadable.
- **Lifecycle** — load/unload are gated local writes, ledgered
  (`TOOLPACK_LOADED` / `TOOLPACK_UNLOADED` / `TOOLPACK_REFUSED`).
  Load records persist in the database; packs reload at core
  startup (re-validated; a pack whose files changed on disk is
  refused) and register as ecosystem members of kind `toolpack`,
  retiring when unloaded.
- **Execution** — v1 runs validated pack code in-process behind a
  narrow wrapper (documented honestly in TOOLPACKS.md); every call
  still routes through the tool registry, hence through the gate
  and into the ledger.
- **Ships with** an example pack (`examples/toolpacks/text_stats`)
  providing read-only text statistics tools.

## The offspring fleet (`offspring.py`, `population.py`)

Offspring cores are the ecosystem's way of growing new intelligence
without risking the parent (docs/OFFSPRING.md):

- A child is seeded from the parent's genome + selected semantic
  memory, starts its own ledger with a genesis entry naming its
  parent and seed hash, and lives in a sandbox data directory.
- Its gate permissions are a subset of the parent's — the same
  subset rule the ecosystem registry now enforces for all members.
- One deliberate variation per generation; head-to-head evaluation
  against the parent under the no-regression rule; promotion is
  owner-approved only, retirement is ledgered.
- The quality-diversity population archive keeps one elite per
  behaviour cell; regressions are never filed.

## Self-healing weights (`healing.py`)

Checkpoints are the ecosystem's genetic material; v1.0.0 makes them
repairable instead of merely tamper-evident:

- **Redundant storage** — `store_checkpoint_healed` keeps the stock
  checkpoint files plus a sharded mirror of `weights.npz` (K shards
  × M copies, per-shard SHA-256). Concatenating one verified copy
  of each shard reproduces the original file byte-identically.
- **Repair ladder** — verify → local shard repair (rewritten only
  after the reconstructed file's hash matches the manifest) →
  nearest healthy bloodline ancestor (verified before use, and
  always *reported as the ancestor* — a healed-from-ancestor load
  never pretends to be the generation that was lost) → loud
  refusal (`CheckpointUnhealableError`).
- **Ledger kinds** — `CHECKPOINT_HEALED`,
  `CHECKPOINT_HEALED_FROM_ANCESTOR`, `CHECKPOINT_HEAL_FAILED`.
  Writing ancestor weights into a corrupted directory requires an
  explicit acknowledgement flag; load-time fallback never writes.
- The stock loader still refuses corruption. Healing is an
  explicit layer above it, not a weakening of it.

## Intelligence — the bloodline at v1.0.0

- **Gen 0 remains the live model.** No experiment in this release
  was promoted, and `model_data/current.json` does not exist.
- **Silent-growth Gen 1 (experiment, entry 0047)** — the multiverse
  diagnosis (growth shock: new capacity entering at random init
  injures the child) was tested directly: a Gen 1 child grown with
  zero-init ("silent") growth, sub-epoch snapshot selection, a
  ≈5% teacher share, and the A2 selection rule **passed the
  no-regression gate — the first Gen 1 child to do so**: perplexity
  45.71 vs Gen 0's 57.71 re-measured on the same frozen ground
  (−20.8%), core probes 10/10, teacher score improved. It is
  recorded in `model_data/silent_growth/manifest.json` as a
  **promotion candidate only**. Promotion remains the owner's
  decision, exactly as the bloodline rule requires.
- **Post-1.0.0 update (release 1.0.1, 2026-11-07):** the owner
  promoted that candidate on 2026-11-06 after a six-point
  verification program passed in full. **Gen 1 is now the live
  model** and ships in this release at `model_data/gen1_silent/`
  with the `model_data/current.json` pointer; Gen 0 is preserved
  as ancestor and rollback. A Gen 2 exposure child was grown from
  Gen 1 and refused at its gate on predefined criteria. The
  v1.0.0 bullets above describe the state as released then.

## Deployment (`deploy/`, `config/`)

- **pip** — `pip install .` (verified from a clean venv: evals and
  demo pass from the installed wheel, Gen 0 checkpoint included).
- **Docker** — `deploy/Dockerfile` + `deploy/docker-compose.yml`
  run the loopback API service; the host port is bound to
  127.0.0.1 only, matching the server's own loopback refusal.
- **systemd** — `deploy/steelbound.service` + `deploy/install.sh`
  (PREFIX-overridable) for a dedicated-user install.
- **Config** — the core is configured by CLI flags / `Settings`;
  `config/settings.env.example` documents the surface that exists
  (no invented config formats).

## Invariants the ecosystem may never break

1. The ledger is append-only; corrections supersede, never erase.
2. The gate fails closed; `forbidden` is grantable to no one.
3. Member permissions are a subset of the host's ceiling — always.
4. No self-modification loop may touch the ledger, gate, eval
   suites, or clean-room policy (D-013).
5. Model promotion is owner-approved, one generation at a time,
   under the no-regression rule. Experiments produce candidates,
   never fait accompli.
6. Clean-room: the ecosystem is built from Steelbound's own
   designs only (D-004).
