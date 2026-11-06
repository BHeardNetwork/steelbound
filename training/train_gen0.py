"""Train Steelbound Gen 0 from random weights, locally, on our own
corpus — then test it and write a full report.

Usage: PYTHONPATH=src:training python3 training/train_gen0.py [--epochs N]
Outputs: src/steelbound/model_data/gen0/ (checkpoint) and
training/gen0_report.json (initial/final losses, perplexities,
probe generations, corpus + weights hashes).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from corpus import curriculum_sequences, stable_split, unique_sequences  # noqa: E402
from steelbound.evolution import evaluate_model  # noqa: E402
from steelbound.model import SteelboundModel, Vocab, make_examples, sha256_text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--context", type=int, default=6)
    ap.add_argument("--embed-dim", type=int, default=48)
    ap.add_argument("--hidden-dim", type=int, default=160)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--curriculum-repeat", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "src" / "steelbound" / "model_data" / "gen0"))
    args = ap.parse_args()

    unique = unique_sequences()
    curriculum = curriculum_sequences()
    corpus_hash = sha256_text("\n".join(" ".join(s) for s in unique))

    train_unique, val_seqs = stable_split(unique)
    train_seqs = train_unique + curriculum * args.curriculum_repeat

    vocab = Vocab.build(unique, min_count=args.min_count)
    encode = lambda seqs: [vocab.encode(s) for s in seqs]  # noqa: E731
    X_train, y_train = make_examples(encode(train_seqs), args.context)
    X_val, y_val = make_examples(encode(val_seqs), args.context)

    model = SteelboundModel(vocab, context_len=args.context, embed_dim=args.embed_dim,
                            hidden_dim=args.hidden_dim, seed=args.seed)
    initial = evaluate_model(model, X_val, y_val)
    print(f"vocab={len(vocab)} params={model.param_count()} "
          f"train_examples={len(X_train)} val_examples={len(X_val)}")
    print(f"initial: val_loss={initial['val_loss']:.4f} "
          f"val_ppl={initial['val_perplexity']:.2f} probes={initial['probe_passed']}/{initial['probe_total']}")

    t0 = time.time()
    # Best-epoch snapshot: this corpus is small and the model overfits
    # fast (train loss keeps falling while val loss climbs), so the
    # shipped weights are a selected epoch's, not the last. Selection
    # score balances generalization (val loss) against actually
    # knowing the core facts (probe misses cost 0.3 each).
    from steelbound.evolution import DEFAULT_PROBES
    best = {"score": float("inf"), "epoch": None, "params": None}

    def snapshot(m, row):
        missed = sum(1 for prompt, expected in DEFAULT_PROBES
                     if expected.lower() not in m.generate(prompt, max_tokens=24).lower())
        score = row["val_loss"] + 0.3 * missed
        if score < best["score"]:
            best.update(score=score, epoch=row["epoch"],
                        params={k: p.copy() for k, p in m._params().items()})

    history = model.train(X_train, y_train, epochs=args.epochs, batch=args.batch,
                          lr=args.lr, seed=args.seed, X_val=X_val, y_val=y_val,
                          log=lambda r: print(f"epoch {r['epoch']}: train_loss={r['train_loss']:.4f} "
                                              f"val_loss={r['val_loss']:.4f} val_ppl={r['val_perplexity']:.2f}"),
                          epoch_hook=snapshot)
    seconds = time.time() - t0
    last_epoch = evaluate_model(model, X_val, y_val)
    if best["params"] is not None:
        for name, arr in best["params"].items():
            setattr(model, name, arr)
        model._adam = None
    final = evaluate_model(model, X_val, y_val)
    print(f"last epoch: val_ppl={last_epoch['val_perplexity']:.2f} probes={last_epoch['probe_passed']}/{last_epoch['probe_total']}")
    print(f"saved (best epoch {best['epoch']}): val_loss={final['val_loss']:.4f} val_ppl={final['val_perplexity']:.2f} "
          f"probes={final['probe_passed']}/{final['probe_total']} ({seconds:.1f}s)")

    manifest = model.save_checkpoint(args.out, generation=0, training={
        "method": "from_scratch_random_init",
        "corpus_sha256": corpus_hash,
        "unique_sequences": len(unique),
        "curriculum_sequences": len(curriculum),
        "curriculum_repeat": args.curriculum_repeat,
        "train_examples": int(len(X_train)),
        "val_examples": int(len(X_val)),
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
        "seed": args.seed, "seconds": round(seconds, 1),
        "initial_val_loss": initial["val_loss"],
        "final_val_loss": final["val_loss"],
        "selection": "best_epoch_by_val_loss_plus_probe_misses",
        "best_epoch": best["epoch"],
        "last_epoch_val_loss": last_epoch["val_loss"],
        "last_epoch_val_perplexity": last_epoch["val_perplexity"],
        "history": history,
    })
    report = {
        "generation": 0,
        "corpus_sha256": corpus_hash,
        "weights_sha256": manifest["weights_sha256"],
        "param_count": model.param_count(),
        "vocab_size": len(vocab),
        "initial": {k: initial[k] for k in ("val_loss", "val_perplexity", "probe_passed", "probe_total")},
        "final": {k: final[k] for k in ("val_loss", "val_perplexity", "probe_passed", "probe_total")},
        "best_epoch": best["epoch"],
        "selection": "best_epoch_by_val_loss_plus_probe_misses",
        "last_epoch": {k: last_epoch[k] for k in ("val_loss", "val_perplexity", "probe_passed", "probe_total")},
        "probes": final["probes"],
        "seconds": round(seconds, 1),
    }
    (ROOT / "training" / "gen0_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["final"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
