"""Unit tests for lucene_experiment (pure logic; no JDK or Lucene required).

The parts that actually run Java are verified end to end via
`python -m part2.lucene_experiment`. Here we pin down only the two spots that tend to
break silently: aligning conventions after reading Lucene's output back, and the
definitions of the experiment groups.
"""

import unittest

from part2.lucene_experiment import (
    GROUP_PAIRS,
    LUCENE_TOPK,
    OUR_CONDITIONS,
    lucene_dl_from_explain,
    load_lucene,
)

EXPLAIN = """2.6385446 = weight(text:pizza in 14416) [BM25Similarity], result of:
  2.6385446 = score(freq=6.0), computed as boost * idf * tf from:
    2.9582634 = idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:
      1941 = n, number of documents containing term
      37401 = N, total number of documents with field
    0.8919235 = tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:
      6.0 = freq, occurrences of term within document
      1.2 = k1, term saturation parameter
      0.75 = b, length normalization parameter
      48.0 = dl, length of field (approximate)
      101.16277 = avgdl, average length of field
"""


class TestExplainParsing(unittest.TestCase):

    def test_reads_dl(self):
        self.assertEqual(lucene_dl_from_explain(EXPLAIN), 48.0)

    def test_returns_none_when_absent(self):
        self.assertIsNone(lucene_dl_from_explain(""))
        self.assertIsNone(lucene_dl_from_explain(None))


class TestLoadLucene(unittest.TestCase):

    def _write(self, tmp, payload):
        import json
        from pathlib import Path
        path = Path(tmp) / "lucene_results.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_tiebreak_matches_ours(self):
        """Lucene breaks score ties by internal doc id; we break them by ascending
        review_id.

        The results therefore have to be re-sorted by (-score, doc_id) when read back,
        otherwise the order inside a tied block is mistaken for a ranking difference.
        """
        import tempfile
        payload = {
            "lucene": "10.5.2", "java": "25", "topK": 10,
            "indexes": {},
            "queries": [{
                "id": "Q01", "text": "pizza",
                "results": {
                    "lucene_bm25_ours_tok": {"totalHits": 3, "top": [
                        {"rank": 1, "id": "zzz", "score": 2.0},
                        {"rank": 2, "id": "aaa", "score": 2.0},
                        {"rank": 3, "id": "mmm", "score": 3.0}]},
                    "lucene_classic_ours_tok": {"totalHits": 0, "top": []},
                    "lucene_bm25_std": {"totalHits": 0, "top": []},
                    "lucene_classic_std": {"totalHits": 0, "top": []},
                }}]}
        with tempfile.TemporaryDirectory() as tmp:
            runs, meta = load_lucene(self._write(tmp, payload))
        ranked = runs["lucene_bm25_ours_tok"]["Q01"]["ranked"]
        self.assertEqual([d for d, _ in ranked], ["mmm", "aaa", "zzz"])


class TestGroupDefinitions(unittest.TestCase):

    def test_every_pair_names_a_known_condition(self):
        known = set(OUR_CONDITIONS) | {
            "lucene_bm25_ours_tok", "lucene_classic_ours_tok",
            "lucene_bm25_std", "lucene_classic_std"}
        for group, name_a, name_b, note in GROUP_PAIRS:
            self.assertIn(name_a, known, f"{group} side A is unknown: {name_a}")
            self.assertIn(name_b, known, f"{group} side B is unknown: {name_b}")
            self.assertTrue(note.strip(), f"{group} has no note on the difference")

    def test_group_ids_are_unique(self):
        ids = [group for group, _a, _b, _note in GROUP_PAIRS]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
