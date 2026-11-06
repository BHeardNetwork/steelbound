"""Steelbound's own language model — built from scratch, clean-room.

Gen 0 is a neural n-gram language model in the style of a feed-forward
neural probabilistic language model: token embeddings for a fixed
context window are concatenated, passed through one tanh hidden layer,
and projected to a softmax over the vocabulary. Every weight starts
from a seeded random init and is trained locally, by code in this
file, on Steelbound's own corpus (see training/curriculum.txt and the
training report). No external model, weights, or code are used.

Training and inference run on numpy. The rest of Steelbound stays
zero-dependency: this module is imported lazily, only when the
``steelbound`` provider or the training tools are used.

A checkpoint is a directory holding:
  weights.npz   — the trained arrays (float32 on disk)
  vocab.json    — the ordered vocabulary the weights were trained with
  manifest.json — architecture, training stats, and SHA-256 hashes of
                  both files above; loading verifies the hashes and
                  refuses a tampered or mismatched checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re

import numpy as np

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = 0, 1, 2, 3

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.?!,;:]")
_PUNCT = {".", ",", "?", "!", ";", ":"}


def tokenize(text: str) -> list[str]:
    """Lowercase word/punctuation tokens. Deterministic and reversible
    enough for generation: punctuation is its own token."""
    return _TOKEN_RE.findall(text.lower().replace("-", " "))


def detokenize(tokens: list[str]) -> str:
    out = ""
    for tok in tokens:
        if tok in _PUNCT:
            out += tok
        elif tok in SPECIALS:
            continue
        else:
            out += (" " if out else "") + tok
    out = out.strip()
    return out[:1].upper() + out[1:] if out else out


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, token_seqs: list[list[str]], min_count: int = 3,
              max_vocab: int = 4000) -> "Vocab":
        from collections import Counter
        counts = Counter(t for seq in token_seqs for t in seq)
        kept = [t for t, c in counts.items() if c >= min_count and t not in SPECIALS]
        kept.sort(key=lambda t: (-counts[t], t))
        return cls(SPECIALS + kept[: max_vocab - len(SPECIALS)])

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, UNK) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]


def make_examples(seqs: list[list[int]], context_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window (context -> next token) examples. Each sequence is
    wrapped in BOS/EOS and contexts are left-padded with PAD."""
    xs, ys = [], []
    for seq in seqs:
        wrapped = [BOS] + list(seq) + [EOS]
        for t in range(1, len(wrapped)):
            ctx = wrapped[max(0, t - context_len):t]
            xs.append([PAD] * (context_len - len(ctx)) + ctx)
            ys.append(wrapped[t])
    if not xs:
        return (np.zeros((0, context_len), dtype=np.int64),
                np.zeros((0,), dtype=np.int64))
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.int64)


class SteelboundModel:
    """Feed-forward neural language model, trained by backprop + Adam
    implemented in this file. Parameters: E, W1, b1, W2, b2."""

    def __init__(self, vocab: Vocab, context_len: int = 6, embed_dim: int = 48,
                 hidden_dim: int = 160, seed: int = 7):
        self.vocab = vocab
        self.context_len = int(context_len)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        v = len(vocab)
        d_in = self.context_len * self.embed_dim
        self.E = rng.normal(0, 0.5, (v, self.embed_dim))
        self.E[PAD] = 0.0
        self.W1 = rng.normal(0, math.sqrt(1.0 / d_in), (d_in, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.normal(0, math.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, v))
        self.b2 = np.zeros(v)
        self._adam = None

    # -- parameter plumbing -------------------------------------------------
    def _params(self) -> dict[str, np.ndarray]:
        return {"E": self.E, "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def param_count(self) -> int:
        return int(sum(p.size for p in self._params().values()))

    # -- forward / loss -------------------------------------------------------
    def _forward(self, X: np.ndarray):
        B = X.shape[0]
        emb = self.E[X].reshape(B, -1)
        h_pre = emb @ self.W1 + self.b1
        h = np.tanh(h_pre)
        logits = h @ self.W2 + self.b2
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return emb, h, probs

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        B = X.shape[0]
        emb, h, probs = self._forward(X)
        eps = 1e-12
        loss = float(-np.log(probs[np.arange(B), y] + eps).mean())
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T
        dh_pre = dh * (1.0 - h * h)
        dW1 = emb.T @ dh_pre
        db1 = dh_pre.sum(axis=0)
        demb = (dh_pre @ self.W1.T).reshape(B, self.context_len, self.embed_dim)
        dE = np.zeros_like(self.E)
        np.add.at(dE, X.reshape(-1), demb.reshape(-1, self.embed_dim))
        dE[PAD] = 0.0
        return loss, {"E": dE, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def loss(self, X: np.ndarray, y: np.ndarray, batch: int = 512) -> float:
        if len(X) == 0:
            raise ValueError("no examples to evaluate")
        total, n = 0.0, 0
        for i in range(0, len(X), batch):
            xb, yb = X[i:i + batch], y[i:i + batch]
            _, _, probs = self._forward(xb)
            total += float(-np.log(probs[np.arange(len(xb)), yb] + 1e-12).sum())
            n += len(xb)
        return total / n

    def perplexity(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(math.exp(min(self.loss(X, y), 50.0)))

    # -- training -------------------------------------------------------------
    def _adam_step(self, grads: dict[str, np.ndarray], lr: float, t: int,
                   beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                   lr_scales: dict[str, np.ndarray] | None = None):
        if self._adam is None:
            self._adam = {k: (np.zeros_like(p), np.zeros_like(p))
                          for k, p in self._params().items()}
        for name, param in self._params().items():
            m, v = self._adam[name]
            g = grads[name]
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            step = lr * m_hat / (np.sqrt(v_hat) + eps)
            if lr_scales is not None and name in lr_scales:
                # Per-element learning-rate scales (multiverse
                # differential rates: inherited weights move slowly,
                # newly grown rows/columns move at the full rate).
                # Scales of 1.0 reproduce the unscaled step exactly.
                step = step * lr_scales[name]
            param -= step
        self.E[PAD] = 0.0

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 20,
              batch: int = 128, lr: float = 0.005, seed: int = 7,
              X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
              log=None, epoch_hook=None,
              lr_scales: dict[str, np.ndarray] | None = None) -> list[dict]:
        rng = np.random.default_rng(seed)
        history = []
        step = 0
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(X))
            Xs, ys = X[order], y[order]
            for i in range(0, len(Xs), batch):
                step += 1
                _, grads = self.loss_and_grads(Xs[i:i + batch], ys[i:i + batch])
                self._adam_step(grads, lr, step, lr_scales=lr_scales)
            row = {"epoch": epoch, "train_loss": self.loss(X, y)}
            if X_val is not None:
                row["val_loss"] = self.loss(X_val, y_val)
                row["val_perplexity"] = float(math.exp(min(row["val_loss"], 50.0)))
            history.append(row)
            if log:
                log(row)
            if epoch_hook:
                epoch_hook(self, row)
        return history

    def expand_vocab(self, new_tokens: list[str], seed: int = 0) -> list[str]:
        """Grow the vocabulary in place — the bloodline's way of
        learning genuinely new words. Existing token ids never move
        (new tokens append), existing weights are kept exactly, and
        new embedding rows / output columns start from a seeded small
        random init that training then shapes. Returns the tokens
        that were actually added."""
        added = [t for t in new_tokens if t not in self.vocab.stoi]
        if not added:
            return []
        rng = np.random.default_rng(seed)
        n = len(added)
        self.E = np.vstack([self.E, rng.normal(0, 0.5, (n, self.embed_dim))])
        self.W2 = np.hstack([self.W2, rng.normal(
            0, math.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, n))])
        self.b2 = np.concatenate([self.b2, np.zeros(n)])
        self.vocab = Vocab(self.vocab.itos + added)
        self._adam = None
        return added

    def expand_hidden(self, extra_units: int, seed: int = 0) -> int:
        """Grow the hidden layer in place (capacity growth — the
        bloodline's answer to interference: new knowledge gets new
        units instead of evicting old knowledge). Existing weights
        are kept exactly; new W1 columns / W2 rows start from a seeded
        small random init. Returns the new hidden size."""
        if extra_units <= 0:
            return self.hidden_dim
        rng = np.random.default_rng(seed)
        d_in = self.W1.shape[0]
        self.W1 = np.hstack([self.W1, rng.normal(
            0, math.sqrt(1.0 / d_in), (d_in, extra_units))])
        self.b1 = np.concatenate([self.b1, np.zeros(extra_units)])
        self.W2 = np.vstack([self.W2, rng.normal(
            0, math.sqrt(1.0 / self.hidden_dim), (extra_units, self.W2.shape[1]))])
        self.hidden_dim += int(extra_units)
        self._adam = None
        return self.hidden_dim

    # -- generation -------------------------------------------------------------
    def _next_probs(self, context_ids: list[int]) -> np.ndarray:
        ctx = ([PAD] * self.context_len + list(context_ids))[-self.context_len:]
        _, _, probs = self._forward(np.asarray([ctx], dtype=np.int64))
        return probs[0]

    def generate(self, prompt: str, max_tokens: int = 40, temperature: float = 0.0,
                 seed: int = 0) -> str:
        """Continue the prompt with the model. Greedy when temperature
        is 0; otherwise samples from softmax(logits / temperature)."""
        rng = np.random.default_rng(seed)
        ids = self.vocab.encode(tokenize(prompt))
        context = [BOS] + ids
        produced: list[int] = []
        for _ in range(max(1, int(max_tokens))):
            probs = self._next_probs(context)
            probs = probs.copy()
            probs[PAD] = 0.0
            probs[BOS] = 0.0
            if temperature and temperature > 0:
                logits = np.log(probs + 1e-12) / float(temperature)
                logits -= logits.max()
                p = np.exp(logits)
                p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            else:
                nxt = int(np.argmax(probs))
            if nxt == EOS:
                break
            produced.append(nxt)
            context.append(nxt)
        return detokenize(self.vocab.decode(produced))

    # -- checkpoints -------------------------------------------------------------
    def save_checkpoint(self, directory: str, training: dict | None = None,
                        generation: int = 0) -> dict:
        os.makedirs(directory, exist_ok=True)
        weights_path = os.path.join(directory, "weights.npz")
        vocab_path = os.path.join(directory, "vocab.json")
        arrays = {k: p.astype(np.float32) for k, p in self._params().items()}
        np.savez_compressed(weights_path, **arrays)
        vocab_doc = {"itos": self.vocab.itos}
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_doc, f)
        manifest = {
            "format": "steelbound-model",
            "version": 1,
            "generation": int(generation),
            "architecture": {
                "context_len": self.context_len,
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "vocab_size": len(self.vocab),
                "param_count": self.param_count(),
            },
            "seed": self.seed,
            "training": training or {},
            "weights_sha256": _sha256_file(weights_path),
            "vocab_sha256": _sha256_file(vocab_path),
        }
        with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return manifest

    @classmethod
    def load_checkpoint(cls, directory: str) -> "SteelboundModel":
        weights_path = os.path.join(directory, "weights.npz")
        vocab_path = os.path.join(directory, "vocab.json")
        manifest_path = os.path.join(directory, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("format") != "steelbound-model":
            raise ValueError("not a steelbound-model checkpoint")
        if _sha256_file(weights_path) != manifest["weights_sha256"]:
            raise ValueError("weights hash mismatch: checkpoint tampered or corrupt")
        if _sha256_file(vocab_path) != manifest["vocab_sha256"]:
            raise ValueError("vocab hash mismatch: checkpoint tampered or corrupt")
        with open(vocab_path, encoding="utf-8") as f:
            vocab = Vocab(json.load(f)["itos"])
        arch = manifest["architecture"]
        if len(vocab) != arch["vocab_size"]:
            raise ValueError("vocab size does not match manifest")
        model = cls(vocab, context_len=arch["context_len"],
                    embed_dim=arch["embed_dim"], hidden_dim=arch["hidden_dim"],
                    seed=manifest.get("seed", 0))
        with np.load(weights_path) as data:
            for name in ("E", "W1", "b1", "W2", "b2"):
                arr = data[name].astype(np.float64)
                target = getattr(model, name)
                if arr.shape != target.shape:
                    raise ValueError(f"weight shape mismatch for {name}")
                setattr(model, name, arr)
        model.manifest = manifest
        model.generation = manifest.get("generation", 0)
        return model


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_checkpoint_dir() -> str:
    """The promoted generation when one exists, else Gen 0. Evolution
    writes model_data/current.json on promotion; the provider follows
    it, so a promoted child becomes the live model with no code change
    and no third-party model involved."""
    base = os.path.join(os.path.dirname(__file__), "model_data")
    pointer = os.path.join(base, "current.json")
    if os.path.exists(pointer):
        try:
            with open(pointer, encoding="utf-8") as f:
                target = json.load(f)["checkpoint"]
            candidate = os.path.join(base, target)
            if os.path.isdir(candidate):
                return candidate
        except Exception:
            pass
    return os.path.join(base, "gen0")
