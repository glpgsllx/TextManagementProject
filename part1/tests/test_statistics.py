import unittest

from part1.statistics import heaps_points, heaps_statistics, heaps_token_points, log_fit, review_order


class StatisticsTests(unittest.TestCase):
    def test_known_heaps_relation(self):
        fit = log_fit([100, 400, 900, 1600], [10, 20, 30, 40])
        self.assertAlmostEqual(fit["slope"], 0.5)
        self.assertAlmostEqual(fit["coefficient"], 1)
        self.assertAlmostEqual(fit["r2_log10"], 1)

    def test_known_zipf_relation(self):
        fit = log_fit([1, 2, 3, 4, 5], [120, 60, 40, 30, 24])
        self.assertAlmostEqual(fit["slope"], -1)

    def test_exact_token_prefixes_stop_inside_reviews(self):
        reviews = [["a", "b", "a"], ["c", "a", "d", "e"], ["e", "f"]]
        points = heaps_token_points(iter(reviews), checkpoints=(2, 4, 6, 9))
        self.assertEqual(points, [
            {"tokens": 2, "types": 2, "reviews_started": 1},
            {"tokens": 4, "types": 3, "reviews_started": 2},
            {"tokens": 6, "types": 4, "reviews_started": 2},
            {"tokens": 9, "types": 6, "reviews_started": 3},
        ])

    def test_unreached_token_checkpoint_is_not_reported_as_complete(self):
        with self.assertRaisesRegex(ValueError, "Only 3 tokens available"):
            heaps_token_points([["a", "b", "a"]], checkpoints=(2, 4))

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
