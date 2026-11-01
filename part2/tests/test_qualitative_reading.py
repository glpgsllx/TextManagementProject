"""Tests for the W5 qualitative reading.

What is pinned down here is whether any query was missed and whether the agreement
counts are computed correctly. The readings themselves are written by hand — they
cannot be tested and should not be; what can be tested is that they are complete and
that the phenomena they cite really do appear in the results.
"""

import unittest

from part2.qualitative_reading import READINGS, agreement
from part2.queries import QUERIES


def _run(docs):
    return {"n_returned": len(docs),
            "top": [{"doc": d, "score": 1.0, "dl": 10, "snippet": ""} for d in docs]}


class ReadingsCoverageTest(unittest.TestCase):

    def test_every_query_has_a_reading(self):
        missing = [q["id"] for q in QUERIES if q["id"] not in READINGS]
        self.assertEqual(missing, [], f"no reading written yet for: {missing}")

    def test_no_reading_for_unknown_query(self):
        extra = set(READINGS) - {q["id"] for q in QUERIES}
        self.assertEqual(extra, set(), f"readings for unknown queries: {sorted(extra)}")

    def test_each_reading_is_one_sentence_not_a_dump(self):
        # The cap was 160 when the readings were written in Chinese (longest: 105
        # characters). English needs roughly twice the characters for the same
        # content, so it is rescaled to 260 -- observed longest English reading
        # is 223, giving about 15% headroom. Anything well past this is a dump,
        # not a one-line reading.
        for qid, text in READINGS.items():
            self.assertLess(len(text), 260, f"{qid} reading is too long to be 'one line'")
            self.assertTrue(text.strip(), f"{qid} reading is empty")


class AgreementTest(unittest.TestCase):

    def _runs(self, tf, bm):
        return {"tfidf": {k: _run(v) for k, v in tf.items()},
                "bm25": {k: _run(v) for k, v in bm.items()}}

    def test_same_top1_is_counted(self):
        runs = self._runs({"Q1": ["a", "b"]}, {"Q1": ["a", "c"]})
        self.assertEqual(agreement(runs)["same_top1"], 1)

    def test_different_top1_is_not_counted(self):
        runs = self._runs({"Q1": ["a", "b"]}, {"Q1": ["b", "a"]})
        self.assertEqual(agreement(runs)["same_top1"], 0)

    def test_top3_set_ignores_order(self):
        runs = self._runs({"Q1": ["a", "b", "c"]}, {"Q1": ["c", "a", "b"]})
        self.assertEqual(agreement(runs)["same_top3_set"], 1)

    def test_top3_set_requires_same_members(self):
        runs = self._runs({"Q1": ["a", "b", "c"]}, {"Q1": ["a", "b", "d"]})
        self.assertEqual(agreement(runs)["same_top3_set"], 0)

    def test_empty_results_do_not_crash(self):
        runs = self._runs({"Q1": []}, {"Q1": []})
        self.assertEqual(agreement(runs), {"n": 1, "same_top1": 0, "same_top3_set": 1})


if __name__ == "__main__":
    unittest.main()
