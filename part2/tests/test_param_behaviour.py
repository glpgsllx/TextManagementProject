"""Tests for the k1 / b behaviour analysis (closed-form curves and top-10 profiles)."""

import unittest
from types import SimpleNamespace

from part2.param_behaviour import B_VALUES, K1_VALUES, length_curve, saturation_curve, top_profile


class CurveTests(unittest.TestCase):
    def test_grid_values(self):
        self.assertEqual(K1_VALUES, (0.0, 1.2, 1.6, 2.0, 3.0))
        self.assertEqual(B_VALUES, (0.0, 0.5, 0.75, 0.8, 1.0))

    def test_k1_zero_ignores_tf(self):
        self.assertEqual(saturation_curve(0.0, [1, 5, 50]), [1.0, 1.0, 1.0])

    def test_saturation_rises_towards_one_and_slower_for_larger_k1(self):
        low, high = saturation_curve(1.2, [1, 10]), saturation_curve(3.0, [1, 10])
        self.assertLess(low[0], low[1])
        self.assertLess(low[1], 1.0)
        self.assertLess(high[0], low[0])

    def test_b_zero_ignores_length_and_b_one_penalises_long_documents(self):
        flat = length_curve(0.0, [0.5, 1.0, 2.0])
        self.assertAlmostEqual(flat[0], flat[2])
        full = length_curve(1.0, [0.5, 1.0, 2.0])
        self.assertGreater(full[0], full[1])
        self.assertGreater(full[1], full[2])
        self.assertAlmostEqual(full[1], 1.0)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.index = SimpleNamespace(
            postings={"a": {"d1": 3, "d2": 1}, "b": {"d2": 1}},
            doc_lengths={"d1": 10, "d2": 30}, avg_doc_length=20.0)

    def test_profile(self):
        prof = top_profile(self.index, ["a", "b", "zz"], ["d1", "d2"])
        self.assertTrue(prof["multi_term"])
        self.assertAlmostEqual(prof["mean_tf"], 2.0)
        self.assertAlmostEqual(prof["all_terms_share"], 0.5)
        self.assertEqual(prof["ratios"], [0.5, 1.5])

    def test_empty(self):
        self.assertIsNone(top_profile(self.index, ["zz"], ["d1"]))


if __name__ == "__main__":
    unittest.main()