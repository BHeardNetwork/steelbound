# Steelbound's Own Model — design and evolution

Decision D-015 (2026-11-02, owner direction) supersedes the provider
half of D-003: Steelbound's intelligence is its **own trained model**,
grown locally — not an adapter to Ollama, an OpenAI-compatible
endpoint, or any other third-party model. Those adapters remain in
`providers.py` only as transitional scaffolding so old configs do not
break; nothing in the own-model path calls, downloads, or wraps them.

## Gen 0

- Architecture (`src/steelbound/model.py`, clean-room, written for
  this project): a feed-forward neural language model — token
  embeddings for a 6-token context window, one tanh hidden layer,
  softmax over the vocabulary. Backpropagation and Adam are
  implemented in the same file on numpy; a numerical gradient check
  in `tests/test_model.py` verifies the analytic gradients.
- Every weight began as a seeded random init (seed 7). There is no
  pre-trained ancestor.
- Corpus (hash recorded in the checkpoint manifest and in
  `training/gen0_report.json`): only this project's own text —
  `training/curriculum.txt` (authored for Steelbound) plus prose from
  the repo's README, CHANGELOG, WORK_LEDGER and docs, with code,
  URLs and hashes stripped (`training/corpus.py`).
- Checkpoint: `src/steelbound/model_data/gen0/` — `weights.npz`,
  `vocab.json`, and `manifest.json` carrying SHA-256 hashes of both.
  Loading verifies the hashes and refuses a tampered checkpoint.
- Honest scope: Gen 0 is small (hundreds of thousands of parameters).
  It knows Steelbound's own facts and phrasing cold, and it is weak
  outside that — it says what generation it is and does not pretend
  otherwise. Capability grows through evolution, below, not through
  secretly borrowing a bigger third-party brain.

## Evolution — never needing a third-party model, ever

`src/steelbound/evolution.py` grows the model the same way offspring
cores grow, applied to weights:

1. A child generation starts **from the parent's own weights**
   (continued training) on Steelbound's own corpus plus new own
   records (daily episodes, ledgered work, owner-approved additions),
   with exactly one recorded variation per generation.
2. Parent and child are evaluated head-to-head on the *same*
   held-out set (perplexity) and the *same* greedy generation probes.
3. Promotion follows the no-regression rule: the child must be no
   worse on either axis and better on at least one. Promotion is an
   owner-path action; it writes `model_data/current.json`, and the
   `steelbound` provider follows that pointer — the promoted child
   becomes the live model with no code change.
4. Every evolution and promotion can be ledgered
   (`MODEL_EVOLVED` / `MODEL_PROMOTED`) with parent and child weight
   hashes, so lineage is auditable back to Gen 0's random init.

Because each generation's only ancestors are earlier Steelbound
generations and Steelbound's own data, the lineage never depends on a
third-party model at any step — that is the point of growing it this
way, even though a borrowed model would be stronger on day one.

## Synthesis roadmap

The owner's synthesis direction (2026-11-02) — combine the major
breakthroughs rather than stack features blindly — is filtered
through the same no-regression gate as everything else. The unifying
principle is *prediction is compression*: a generation that predicts
its own records better has, measurably, compressed them better.

Already present at the systems level: ledger = provenance and
lineage (the DNA record), genome + offspring = evolution, RSI =
recursive improvement, cognition = self-introspection, backup /
rollback = self-healing, memory consolidation = compression of
episodes into facts.

Gaps this roadmap queues at the model level, each admitted only as a
generation variation that wins head-to-head: attention (context
beyond the fixed window), learned compression of the corpus,
predictive / active inference for planning, a quality-diversity
population of generations instead of a single line, program
synthesis for tool use, and a provenance-tracked human-knowledge
curriculum — knowledge as data with sources, never a third-party
model's weights.

## Promotion to default

The `offline` deterministic provider remains the test-harness default
because the five eval suites specify its exact behavior. A future
generation becomes the live default only when it passes the full
suite + evals as the chat provider *and* the owner promotes it; until
then `--provider steelbound` (or `STEELBOUND` via Settings) selects
the own model explicitly.
