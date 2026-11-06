import json, math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from steelbound.compression import BPEEncoder, bits_per_token, compression_report
from steelbound.model import SteelboundModel, Vocab, make_examples, tokenize
from steelbound.evolution import evaluate_model

CORPUS = [
    tokenize("the ledger is append only and the ledger is hash chained"),
    tokenize("the gate fails closed and the gate never approves itself"),
    tokenize("memory is episodic and memory is semantic and memory has provenance"),
    tokenize("the ledger and the gate and the memory are the core"),
    tokenize("append only means the ledger never changes an old entry"),
] * 4


class TestBPE(unittest.TestCase):
    def test_roundtrip_lossless(self):
        enc = BPEEncoder.train(CORPUS, num_merges=30)
        for seq in CORPUS:
            self.assertEqual(enc.decode(enc.encode(seq)), seq)

    def test_merges_actually_compress_token_count(self):
        enc = BPEEncoder.train(CORPUS, num_merges=30)
        before = sum(len(s) for s in CORPUS)
        after = sum(len(enc.encode(s)) for s in CORPUS)
        self.assertLess(after, before)
        self.assertGreater(len(enc), 0)

    def test_deterministic_merges(self):
        a = BPEEncoder.train(CORPUS, num_merges=30)
        b = BPEEncoder.train(CORPUS, num_merges=30)
        self.assertEqual(a.merges, b.merges)
        self.assertEqual(a, b)

    def test_serialization_roundtrip(self):
        enc = BPEEncoder.train(CORPUS, num_merges=30)
        rebuilt = BPEEncoder.from_json(enc.to_json())
        self.assertEqual(rebuilt, enc)
        self.assertEqual(json.loads(enc.to_json())["format"], "steelbound-bpe")
        for seq in CORPUS:
            self.assertEqual(rebuilt.encode(seq), enc.encode(seq))
            self.assertEqual(rebuilt.decode(rebuilt.encode(seq)), seq)

    def test_unseen_tokens_pass_through(self):
        enc = BPEEncoder.train(CORPUS, num_merges=10)
        seq = ["zzq", "never", "seen"]
        self.assertEqual(enc.decode(enc.encode(seq)), seq)


class TestBitsPerToken(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocab.build(CORPUS, min_count=1)
        seqs = [self.vocab.encode(s) for s in CORPUS]
        self.X, self.y = make_examples(seqs, context_len=4)
        self.model = SteelboundModel(self.vocab, context_len=4,
                                     embed_dim=8, hidden_dim=16, seed=3)

    def test_bits_per_token_is_loss_over_ln2(self):
        expected = self.model.loss(self.X, self.y) / math.log(2.0)
        self.assertAlmostEqual(bits_per_token(self.model, self.X, self.y),
                               expected, places=9)

    def test_compression_report_fields_and_ratio(self):
        seqs = [self.vocab.encode(s) for s in CORPUS]
        report = compression_report(self.model, seqs)
        for field in ("bits_per_token", "total_codelength_bits", "tokens",
                      "vocab_size", "naive_bits_per_token",
                      "naive_codelength_bits", "ratio"):
            self.assertIn(field, report)
        self.assertEqual(report["tokens"], len(self.y))
        self.assertAlmostEqual(report["total_codelength_bits"],
                               report["bits_per_token"] * report["tokens"],
                               places=6)
        self.assertAlmostEqual(report["naive_codelength_bits"],
                               report["naive_bits_per_token"] * report["tokens"],
                               places=6)
        self.assertAlmostEqual(report["ratio"],
                               report["total_codelength_bits"]
                               / report["naive_codelength_bits"], places=9)
        naive_bpt = max(1, math.ceil(math.log2(len(self.vocab))))
        self.assertEqual(report["naive_bits_per_token"], naive_bpt)

    def test_evaluate_model_reports_bits_per_token(self):
        metrics = evaluate_model(self.model, self.X, self.y, probes=[])
        self.assertIn("bits_per_token", metrics)
        self.assertAlmostEqual(metrics["bits_per_token"],
                               metrics["val_loss"] / math.log(2.0), places=9)


if __name__ == "__main__":
    unittest.main()
