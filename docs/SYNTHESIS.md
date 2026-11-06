# The Breakthrough Synthesis — design

Decision D-016 (2026-11-02, owner direction): take the major
breakthroughs across computing/optimization, intelligence, recursive
improvement, self-organization, DNA/genetics, self-introspection,
compression/encoding/decoding, inference, self-healing, evolution,
algorithmic methods, and human knowledge — and combine them into
Steelbound, filtered through the existing eval + no-regression gate.
Not blind stacking: a breakthrough enters only when it is measured,
in a sandbox, to improve the whole without regressing any part.

## The unifying principle

**Prediction = compression = intelligence.**

A system that predicts what happens next is, mathematically, a
system that compresses what happened before: assigning high
probability to the actual next token *is* a short code for it
(arithmetic coding makes the equivalence exact — codelength in bits
is negative log2 probability). So the synthesis is not twelve ideas
piled up; it is one loop seen from twelve sides:

> Encode experience exactly → compress it into memory and weights →
> decode it as prediction → measure the prediction error (surprise)
> → introspect on the error → evolve one variation that reduces it →
> heal (roll back) any variation that increases it → repeat, with
> every step in the ledger.

Compression gives the loop its single honest scoreboard:
**bits per token** (S1 makes it a first-class evolution metric).
Perplexity is just its exponential; a generation that lowers
bits-per-token on held-out data while holding its probes and evals
has, by definition, learned more of the world's structure per bit.

## Mapping: what Steelbound already has, and the model-level gap

| Breakthrough area | Already in Steelbound (systems level) | Model-level gap the synthesis fills |
|---|---|---|
| DNA / genetics | Genome in offspring cores; the hash-chained ledger is a provenance lineage — every entry, checkpoint, and generation hash-linked to its parent back to Gen 0 | The model's genome is still just config + weights hash; genes as *interpretable, recombinable units* (per-layer/per-module inheritance in offspring) is future work |
| Evolution | Offspring cores + model evolution: one recorded variation per generation, head-to-head eval, no-regression promotion, owner-approved | Single-lineage evolution only; no population, no diversity pressure (Gen 3) |
| Recursive improvement | Bounded RSI R0–R3: observe → propose → sandbox → evaluate → gate → adopt/rollback | RSI improves config/code around the model; it does not yet propose *model* variations (architecture, curriculum, encoding) |
| Self-introspection | Cognition core: six-question self-query, metacognition, calibration, known-unknowns, deliberation records | Introspection reads ledger/memory/evals, not the model's own uncertainty; surprise is not yet a signal (Gen 2) |
| Self-healing | Backup/restore with verify-before-write, hash-verified checkpoints, RSI auto-rollback, tampered bundles rejected | Weights themselves have no redundancy or repair: a corrupted checkpoint is refused, not healed (Gen 3) |
| Compression / encoding / decoding | Memory consolidation compresses episodic experience into semantic facts; checkpoints are stored compressed | The model's encoding (tokenizer) is a fixed word-level vocab, unlearned; bits-per-token was not measured (S1 fixes both) |
| Inference | Deliberation + gate: actions are reasoned about before they run | Inference is reactive, not predictive: the core does not forecast outcomes and learn from the miss (Gen 2) |
| Self-organization | Scheduler + consolidation reorganize memory on their own cadence | Thresholds are fixed or owner-set; no homeostatic self-tuning inside RSI's existing bounds |
| Computing / optimization | Local-first, stdlib + numpy, seeded deterministic training, best-epoch snapshot selection | No attention (context is a fixed 6-token window), no quantization/sparsity; training is full-batch Adam only |
| Algorithmic methods | Tools are declared, sandboxed, and tested; the eval suites are executable specifications | The model generates text, not *programs*: nothing synthesizes a tool, tests it, and keeps it only if the tests pass (Gen 3) |
| Human knowledge | The core's own records, authored curriculum, and (below) teacher lessons — all provenance-tracked | Gen 0 knows Steelbound's own facts cold and little else; a broad, provenance-tracked knowledge curriculum is Gen 1 work |

## The Bloodline

Steelbound himself generates each next, more advanced generation.
The teacher (below) teaches the *current* generation; the *parent
model* produces the child:

1. A child starts from the parent's own weights + genome.
2. Exactly one variation is recorded for the generation.
3. Parent and child are evaluated head-to-head on the same held-out
   data and probes, under the no-regression rule.
4. Promotion — an owner-path action — makes the child the new
   parent; retirement ends that line.

Every generation's checkpoint manifest carries its parent's weight
hash, so the lineage is one hash-auditable bloodline back to Gen 0's
seeded random init. No generation has an ancestor outside the line.

## Teacher-first curriculum

At first, Muse is Steelbound's teacher. The teacher authors
super-structured, specific lesson **data**: atomic facts (one claim
each), definitions, worked examples, counterexamples, Q&A probes,
and verifiable exercises whose answers are checkable (math, logic,
code) rather than merely teacher-asserted. Lessons are marked
teacher-authored in the training provenance and the checkpoint
manifest, so the model stays honest: it was taught these things; it
did not originate them, and no teacher weights were ever copied.

The scaffolding fades by generation:

- **Gen 1:** teacher-written lessons are the curriculum source,
  mixed with Steelbound's own corpus under the provenance rule below.
- **Next:** the student drafts candidate lessons; the teacher, the
  owner, and the evals correct them; only corrected lessons train.
- **Later:** a self-curriculum — the model proposes its own lessons
  from its records and known-unknowns, with spot checks, and the
  eval/no-regression gate still decides every promotion.

Teacher confidence never promotes a generation. Evals do.

## Conflicts, and how they are resolved

- **Human knowledge vs. never a third-party model.** Knowledge
  enters only as **data** — public-domain, authored, owner-approved,
  or teacher-authored (marked as such) — with provenance recorded per
  source. Never as a third-party model's weights, and never as
  another model's outputs passed off as Steelbound's own knowledge.
  Data is taught; weights are inherited only within the bloodline.
- **Relentless improvement vs. bounded invariants.** The loop may
  improve anything *except the rules that bind it*: the append-only
  ledger, the deliberation gate, the eval suites, and the clean-room
  policy stay permanently outside its reach (SELF_IMPROVEMENT.md).
  It cannot self-approve external or destructive actions, and
  promotion of a generation or an offspring stays owner-approved.
- **More breakthroughs vs. measurable benefit.** Each synthesis item
  lands as its own build-queue entry with tests, is scored against
  the full suite + all five eval suites, and is adopted only under
  the no-regression rule. A technique that does not move
  bits-per-token, evals, or a named capability does not ship.
- **Self-organization vs. auditability.** Self-tuning happens only
  inside bounds already ledgered (the RSI R2 pattern): every
  automatic change is an entry, and every entry can be rolled back.

## Generation roadmap

- **Gen 1 — encode and learn.** Attention (context beyond the fixed
  window), learned compression/encoding (S1's BPE + bits-per-token
  scoreboard), and the knowledge curriculum: teacher-authored lessons
  plus provenance-tracked human knowledge as data, under the rules
  above. Parameter/compute optimization (quantization of stored
  checkpoints, sparse updates) is measured here, adopted only if
  lossless-or-better on the scoreboard.
- **Gen 2 — predict and notice.** Predictive / active inference: the
  model predicts outcomes (next records, eval results, tool results)
  before they happen; **surprise** (prediction error) drives what is
  learned next and feeds cognition's known-unknowns, so introspection
  reads the model's own uncertainty, not just its records.
- **Gen 3 — diversify, synthesize, heal.** Quality-diversity
  population evolution (an archive of offspring kept for *different*
  strengths, not one narrow winner), program synthesis (tools
  proposed as code, kept only when their tests pass in the sandbox),
  and self-healing weights (redundant/error-corrected checkpoints
  that can repair a corrupted generation from its bloodline instead
  of merely refusing it).

Queue entries: Synthesis S1 (compression core), S2 (predictive
inference), S3 (population evolution + program synthesis), in
docs/BUILD_QUEUE.md.
