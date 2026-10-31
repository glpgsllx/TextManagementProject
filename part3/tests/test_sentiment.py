import unittest

import numpy as np

from part3.data import business_sectors, season_of
from part3.sentiment import (
    bootstrap_ci, cohort_summaries, disagreements, excluded_businesses, group_summaries, polarity,
    same_business_change, score_reviews, select_sectors, star_validation, summarize,
)


def review(business_id="b1", sectors=("Italian",), year=2016, season="Spring", stars=5.0, compound=0.5, text=""):
    return {"review_id": f"{business_id}-{year}-{compound}", "business_id": business_id, "business": business_id,
            "sectors": frozenset(sectors), "year": year, "season": season, "stars": stars,
            "compound": compound, "text": text}


class DataTests(unittest.TestCase):
    def test_season_boundaries(self):
        months = {1: "Winter", 2: "Winter", 3: "Spring", 5: "Spring", 6: "Summer",
                  8: "Summer", 9: "Autumn", 11: "Autumn", 12: "Winter"}
        for month, season in months.items():
            self.assertEqual(season_of(month), season)

    def test_categories_match_exact_labels_only(self):
        sectors = business_sectors("Restaurants, Food Stands, Fast Food")
        self.assertIn("Fast Food", sectors)
        self.assertNotIn("Food", sectors)
        self.assertEqual(business_sectors(None), frozenset())


class SelectionTests(unittest.TestCase):
    def test_overlapping_business_and_out_of_range_years_are_excluded(self):
        reviews = [review("italian"), review("fast", ("Fast Food",)),
                   review("both", ("Italian", "Fast Food", "Pizza")), review("old", year=2013), review("other", ("Pizza",))]
        selected = select_sectors(reviews, ["Italian", "Fast Food"], 2014, 2021)
        self.assertEqual({r["business_id"]: r["sector"] for r in selected}, {"italian": "Italian", "fast": "Fast Food"})
        self.assertEqual(excluded_businesses(reviews, ["Italian", "Fast Food"]), ["both"])


class ScoringTests(unittest.TestCase):
    def test_vader_uses_raw_emphasis(self):
        reviews = [review(text="The food was good"), review(text="The food was GOOD!!!")]
        score_reviews(reviews)
        self.assertGreater(reviews[1]["compound"], reviews[0]["compound"])

    def test_polarity_thresholds_are_inclusive(self):
        self.assertEqual([polarity(x) for x in (0.05, 0.049, -0.049, -0.05)],
                         ["positive", "neutral", "neutral", "negative"])


class AggregateTests(unittest.TestCase):
    def test_business_equal_mean_differs_from_review_mean(self):
        reviews = [review("busy", compound=1.0, stars=5.0) for _ in range(3)] + [review("quiet", compound=0.0, stars=1.0)]
        row = summarize(reviews, np.random.default_rng(0), 50)
        self.assertEqual((row["reviews"], row["businesses"]), (4, 2))
        self.assertAlmostEqual(row["mean_compound"], 0.75)
        self.assertAlmostEqual(row["business_equal_mean"], 0.5)
        self.assertAlmostEqual(row["mean_stars"], 4.0)
        self.assertAlmostEqual(row["positive_share"] + row["neutral_share"] + row["negative_share"], 1.0)

    def test_bootstrap_is_reproducible_and_brackets_the_mean(self):
        values = [0.1, 0.4, 0.5, 0.9, -0.2, 0.3]
        first = bootstrap_ci(values, np.random.default_rng(42), 500)
        self.assertEqual(first, bootstrap_ci(values, np.random.default_rng(42), 500))
        self.assertLess(first[0], np.mean(values))
        self.assertGreater(first[1], np.mean(values))

    def test_constant_scores_have_zero_width_interval(self):
        self.assertEqual(bootstrap_ci([0.5] * 5, np.random.default_rng(0), 20), (0.5, 0.5))

    def test_group_summaries_conserve_reviews(self):
        reviews = [dict(review(year=y), sector=s) for y in (2015, 2016) for s in ("Italian", "Fast Food")]
        rows = group_summaries(reviews, "year", np.random.default_rng(0), 20)
        self.assertEqual(sum(r["reviews"] for r in rows), len(reviews))
        self.assertEqual([(r["sector"], r["year"]) for r in rows],
                         [("Fast Food", 2015), ("Fast Food", 2016), ("Italian", 2015), ("Italian", 2016)])

    def test_same_business_change_ignores_businesses_missing_a_period(self):
        reviews = [dict(review("kept", year=2015, compound=0.8), sector="Italian"),
                   dict(review("kept", year=2020, compound=0.2), sector="Italian"),
                   dict(review("opened_later", year=2020, compound=1.0), sector="Italian")]
        row = same_business_change(reviews, (2015, 2016), (2020, 2021), min_reviews=1)[0]
        self.assertEqual((row["businesses"], row["businesses_declined"]), (1, 1))
        self.assertEqual((row["early_mean"], row["late_mean"]), (0.8, 0.2))

    def test_cohorts_split_on_first_review_year(self):
        reviews = [dict(review("old", year=2015, compound=0.8), sector="Italian"),
                   dict(review("old", year=2019, compound=0.2), sector="Italian"),
                   dict(review("new", year=2019, compound=0.6), sector="Italian")]
        rows = cohort_summaries(reviews, 2015)
        self.assertEqual([(r["year"], r["cohort"], r["mean_compound"]) for r in rows],
                         [(2015, "incumbent", 0.8), (2019, "entrant", 0.6), (2019, "incumbent", 0.2)])

    def test_star_validation_and_disagreements(self):
        reviews = [dict(review(stars=1.0, compound=0.9, text="great  sarcasm"), sector="Italian"),
                   dict(review(stars=1.0, compound=-0.9), sector="Italian"),
                   dict(review(stars=5.0, compound=-0.2), sector="Italian")]
        rows = star_validation(reviews)
        self.assertEqual([(r["stars"], r["reviews"], r["positive_share"]) for r in rows], [(1.0, 2, 0.5), (5.0, 1, 0.0)])
        conflicts = disagreements(reviews)
        self.assertEqual([r["kind"] for r in conflicts], ["one_star_scored_positive", "five_star_scored_negative"])
        self.assertEqual(conflicts[0]["text"], "great sarcasm")


if __name__ == "__main__":
    unittest.main()
