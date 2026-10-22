import unittest

from part1.statistics import heaps_points, heaps_statistics, log_fit, review_order


class StatisticsTests(unittest.TestCase):
    def test_known_heaps_relation(self):
        fit = log_fit([100, 400, 900, 1600], [10, 20, 30, 40])
        self.assertAlmostEqual(fit["slope"], 0.5)
        self.assertAlmostEqual(fit["coefficient"], 1)
        self.assertAlmostEqual(fit["r2_log10"], 1)

    def test_known_zipf_relation(self):
        fit = log_fit([1, 2, 3, 4, 5], [120, 60, 40, 30, 24])
        self.assertAlmostEqual(fit["slope"], -1)

    def test_nested_samples_conserve_full_corpus(self):
        ids = [f"r{i}" for i in range(100)]
        points = heaps_points(ids, [2] * 100, [(i, i + 1) for i in range(100)])
        self.assertEqual([p["reviews"] for p in points if p["required_checkpoint"]], [10, 30, 50, 100])
        self.assertEqual(points[-1]["tokens"], 200)
        self.assertEqual(points[-1]["types"], 101)
        self.assertEqual([p["types"] for p in points], sorted(p["types"] for p in points))
        self.assertEqual(points, heaps_points(ids, [2] * 100, [(i, i + 1) for i in range(100)]))
        self.assertNotEqual(review_order(ids, 42), review_order(ids, 43))
        self.assertEqual(heaps_statistics(points)["four_checkpoint_fit"]["n_points"], 4)


if __name__ == "__main__":
    unittest.main()
