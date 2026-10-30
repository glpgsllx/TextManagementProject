"""W5 定性判读的测试。

这里钉住的是"判读有没有漏条目"和"一致性计数算得对不对"——判读内容本身是
人写的，测不了也不该测；能测的是它完整、且引用的现象确实在结果里。
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
        self.assertEqual(missing, [], f"这些查询还没写判读：{missing}")

    def test_no_reading_for_unknown_query(self):
        extra = set(READINGS) - {q["id"] for q in QUERIES}
        self.assertEqual(extra, set(), f"判读里有不存在的查询：{sorted(extra)}")

    def test_each_reading_is_one_sentence_not_a_dump(self):
        for qid, text in READINGS.items():
            self.assertLess(len(text), 160, f"{qid} 的判读过长，不是'一句'")
            self.assertTrue(text.strip(), f"{qid} 的判读是空的")


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
