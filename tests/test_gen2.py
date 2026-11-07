"""Tests for Gen 2's teacher-exposure mix builder (src/steelbound/gen2.py).

Fast unit tests only: full coverage as a floor, achieved share
close to target, determinism, and the recorded info's honesty.
The full experiment is training/run_gen2_exposure.py, separate
from the unit suite.
"""
import unittest
from collections import Counter

from steelbound.gen2 import build_teacher_exposure, example_count

# Distinct sequences of varied lengths (example counts 4..9).
TEACHER = [[f"t{i}"] * (3 + (i % 6)) for i in range(40)]


class TestTeacherExposure(unittest.TestCase):
    def test_full_coverage_even_at_low_share(self):
        # The Gen 1 failure mode: at a 5% target the old subset
        # dropped 80% of the sequences. Coverage is now a floor.
        block, info = build_teacher_exposure(TEACHER, base_examples=100000,
                                             target_share=0.05)
        counts = Counter(map(tuple, block))
        for seq in TEACHER:
            self.assertGreaterEqual(counts[tuple(seq)], 1)
        for seq in TEACHER:
            self.assertIn(seq, block)
        self.assertEqual(info["coverage_fraction"], 1.0)
        self.assertGreaterEqual(info["full_repeats"], 1)

    def test_share_close_to_target(self):
        base = 20000
        block, info = build_teacher_exposure(TEACHER, base_examples=base,
                                             target_share=0.25)
        achieved = example_count(block) / (base + example_count(block))
        self.assertAlmostEqual(achieved, 0.25, delta=0.01)
        self.assertAlmostEqual(info["teacher_share_examples"], achieved)

    def test_block_structure_is_repeats_plus_topup(self):
        block, info = build_teacher_exposure(TEACHER, base_examples=5000,
                                             target_share=0.25)
        n = len(TEACHER)
        r, top = info["full_repeats"], info["topup_count"]
        self.assertEqual(len(block), r * n + top)
        for k in range(r):
            self.assertEqual(block[k * n:(k + 1) * n], TEACHER)
        self.assertEqual(info["teacher_sequences_with_repeats"], len(block))
        self.assertEqual(info["teacher_examples"], example_count(block))

    def test_deterministic(self):
        a, info_a = build_teacher_exposure(TEACHER, base_examples=5000)
        b, info_b = build_teacher_exposure(TEACHER, base_examples=5000)
        self.assertEqual(a, b)
        self.assertEqual(info_a, info_b)

    def test_empty_teacher_set(self):
        block, info = build_teacher_exposure([], base_examples=1000)
        self.assertEqual(block, [])
        self.assertEqual(info["teacher_examples"], 0)
        self.assertEqual(info["coverage_fraction"], 0.0)

    def test_bad_share_rejected(self):
        with self.assertRaises(ValueError):
            build_teacher_exposure(TEACHER, base_examples=1000, target_share=0.0)
        with self.assertRaises(ValueError):
            build_teacher_exposure(TEACHER, base_examples=1000, target_share=1.0)


if __name__ == "__main__":
    unittest.main()
