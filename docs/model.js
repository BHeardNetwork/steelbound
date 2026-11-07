"use strict";
/*
 * Steelbound Gen 1 — faithful JavaScript port of the model's inference.
 *
 * Mirrors src/steelbound/model.py (SteelboundModel._forward,
 * _next_probs, generate, tokenize, detokenize) statement for statement,
 * so the text produced here is the text the Python checkpoint produces.
 * Greedy (temperature 0) generation only, exactly as the eval probes run.
 *
 * Weights arrive as raw little-endian float32 (the checkpoint's exact
 * stored values) and are computed in float64, matching the Python
 * loader, which casts the float32 arrays to float64 before inference.
 */

const SB_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"];
const SB_PAD = 0, SB_BOS = 1, SB_EOS = 2, SB_UNK = 3;
const SB_PUNCT = new Set([".", ",", "?", "!", ";", ":"]);
const SB_TOKEN_RE = /[a-z0-9]+(?:'[a-z]+)?|[.?!,;:]/g;

function sbTokenize(text) {
  const cleaned = text.toLowerCase().replace(/-/g, " ");
  return cleaned.match(SB_TOKEN_RE) || [];
}

function sbDetokenize(tokens) {
  let out = "";
  for (const tok of tokens) {
    if (SB_PUNCT.has(tok)) {
      out += tok;
    } else if (SB_SPECIALS.includes(tok)) {
      continue;
    } else {
      out += (out ? " " : "") + tok;
    }
  }
  out = out.trim();
  return out ? out.charAt(0).toUpperCase() + out.slice(1) : out;
}

class SteelboundJS {
  constructor(meta, weightBuffer) {
    const arch = meta.architecture;
    this.contextLen = arch.context_len;
    this.embedDim = arch.embed_dim;
    this.hiddenDim = arch.hidden_dim;
    this.itos = meta.vocab;
    this.stoi = new Map(this.itos.map((t, i) => [t, i]));
    const all = new Float32Array(weightBuffer);
    const view = (name) => {
      const a = meta.arrays[name];
      return all.subarray(a.offset, a.offset + a.count);
    };
    this.E = view("E");
    this.W1 = view("W1");
    this.b1 = view("b1");
    this.W2 = view("W2");
    this.b2 = view("b2");
    this.vocabSize = this.itos.length;
  }

  encode(tokens) {
    return tokens.map((t) => (this.stoi.has(t) ? this.stoi.get(t) : SB_UNK));
  }

  // contextIds -> probability row, mirroring _next_probs + _forward.
  nextProbs(contextIds) {
    const CL = this.contextLen, ED = this.embedDim, HD = this.hiddenDim, V = this.vocabSize;
    let ctx = new Array(CL).fill(SB_PAD).concat(contextIds);
    ctx = ctx.slice(ctx.length - CL);
    // emb = E[ctx] flattened (context order), length CL*ED = 288
    const emb = new Float64Array(CL * ED);
    for (let i = 0; i < CL; i++) {
      const base = ctx[i] * ED;
      for (let d = 0; d < ED; d++) emb[i * ED + d] = this.E[base + d];
    }
    // h = tanh(emb @ W1 + b1); W1 is (288, 224) row-major
    const h = new Float64Array(HD);
    for (let j = 0; j < HD; j++) {
      let s = this.b1[j];
      for (let k = 0; k < CL * ED; k++) s += emb[k] * this.W1[k * HD + j];
      h[j] = Math.tanh(s);
    }
    // logits = h @ W2 + b2; W2 is (224, 1158) row-major
    const logits = new Float64Array(V);
    for (let v = 0; v < V; v++) {
      let s = this.b2[v];
      for (let j = 0; j < HD; j++) s += h[j] * this.W2[j * V + v];
      logits[v] = s;
    }
    // softmax with max subtraction, as in _forward
    let max = -Infinity;
    for (let v = 0; v < V; v++) if (logits[v] > max) max = logits[v];
    const probs = new Float64Array(V);
    let sum = 0;
    for (let v = 0; v < V; v++) { probs[v] = Math.exp(logits[v] - max); sum += probs[v]; }
    for (let v = 0; v < V; v++) probs[v] /= sum;
    return probs;
  }

  // Mirrors generate(prompt, maxTokens, temperature=0) with greedy argmax:
  // PAD and BOS are zeroed before the argmax; EOS stops; numpy argmax
  // tie-breaking (first maximal index) is preserved by the strict > scan.
  generate(prompt, maxTokens = 24) {
    const ids = this.encode(sbTokenize(prompt));
    const context = [SB_BOS].concat(ids);
    const produced = [];
    const limit = Math.max(1, Math.floor(maxTokens));
    for (let step = 0; step < limit; step++) {
      const probs = this.nextProbs(context);
      probs[SB_PAD] = 0.0;
      probs[SB_BOS] = 0.0;
      let nxt = 0;
      for (let v = 1; v < probs.length; v++) if (probs[v] > probs[nxt]) nxt = v;
      if (nxt === SB_EOS) break;
      produced.push(nxt);
      context.push(nxt);
    }
    return sbDetokenize(produced.map((i) => this.itos[i]));
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { SteelboundJS, sbTokenize, sbDetokenize };
}
