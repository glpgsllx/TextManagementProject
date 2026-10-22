import unittest

from part1.tokenizer import scan, tokenize
from part1.boundary_cases import CASES


class TokenizerTests(unittest.TestCase):
    def test_boundary_cases(self):
        for name, text, expected in CASES:
            with self.subTest(case=name):
                self.assertEqual(tokenize(text), expected)

    def test_surface_stage_keeps_case_and_punctuation(self):
        self.assertEqual(scan("Don't miss the-best-food 😊!"),
                         ["Don't", "miss", "the", "best", "food", "😊", "!"])

    def test_normalization_includes_contractions(self):
        self.assertEqual(tokenize("DON’T go!"), ["do", "not", "go"])

    def test_type_error(self):
        with self.assertRaises(TypeError):
            tokenize(None)


if __name__ == "__main__":
    unittest.main()
