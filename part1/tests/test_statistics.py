import unittest

from part1.statistics import heaps_sampling_plan, heaps_statistics, heaps_token_points, log_fit, review_order


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

    def test_sampling_plan_has_a_fixed_fit_grid_and_review_checkpoints(self):
        ids = [f"r{i}" for i in range(100)]
        plan = heaps_sampling_plan(ids, [40] * 100)
        self.assertEqual(len(plan["fit_tokens"]), 80)
        self.assertEqual(min(plan["fit_tokens"]), 1000)
        self.assertEqual(max(plan["fit_tokens"]), 4000)
        self.assertEqual(plan["review_fractions"], {400: .1, 1200: .3, 2000: .5, 4000: 1.0})
        self.assertEqual(plan["checkpoints"], sorted(set(plan["checkpoints"])))
        self.assertTrue(plan["fit_tokens"].issubset(plan["checkpoints"]))
        self.assertEqual(plan["order"], review_order(ids, 42))
        self.assertNotEqual(plan["order"], review_order(ids, 43))

    def test_unified_fit_uses_only_grid_points_and_prediction_withholds_endpoint(self):
        points = [{"tokens": n, "types": v, "fit_checkpoint": True}
                  for n, v in ((100, 10), (400, 20), (900, 30), (1600, 40), (2500, 50))]
        # An annotation-only point does not change the fitted parameters.
        points.insert(1, {"tokens": 200, "types": 30, "fit_checkpoint": False})
        stats = heaps_statistics(points)
        self.assertAlmostEqual(stats["fit"]["k"], 1)
        self.assertAlmostEqual(stats["fit"]["beta"], .5)
        self.assertEqual(stats["fit"]["n_points"], 5)
        self.assertEqual(stats["prediction"]["max_training_tokens"], 1600)
        self.assertEqual(stats["prediction"]["n_points"], 4)
        self.assertAlmostEqual(stats["prediction"]["predicted_types"], 50)
        self.assertAlmostEqual(stats["prediction"]["absolute_percentage_error"], 0)

    def test_token_prefix_full_endpoint_conserves_counts(self):
        reviews = [[str(i), str(i + 1)] for i in range(100)]
        points = heaps_token_points(reviews, checkpoints=(20, 60, 100, 200))
        self.assertEqual(points[-1]["tokens"], 200)
        self.assertEqual(points[-1]["types"], 101)
        self.assertEqual([p["types"] for p in points], sorted(p["types"] for p in points))


if __name__ == "__main__":
    unittest.main()
