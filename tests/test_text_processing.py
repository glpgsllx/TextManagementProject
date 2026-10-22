import unittest

from text_processing import (
    available_tokenizers,
    get_tokenizer,
    surface_tokens,
    tokenize,
)


class SharedTokenizerTests(unittest.TestCase):
    def test_methods_expose_their_boundary_differences(self):
        examples = {
            "custom": ["great", "food", "4.89"],
            "whitespace": ["great-food", "4.89!"],
            "wordpunct": ["great", "food", "4", "89"],
        }
        self.assertEqual(set(available_tokenizers()), set(examples))
        for name, expected in examples.items():
            with self.subTest(method=name):
                self.assertEqual(get_tokenizer(name).tokenize("Great-food 4.89!"), expected)

    def test_query_and_document_use_the_same_terms(self):
        tokenizer = get_tokenizer()
        document = tokenizer.tokenize("I don't like the-best-food 😊!")
        query = tokenizer.tokenize("DON’T best-food 😊")
        self.assertEqual(query, ["do", "not", "best", "food", "😊"])
        self.assertTrue(set(query).issubset(document))
        self.assertEqual(tokenize("DON’T best-food 😊"), query)

    def test_surface_inspection_does_not_expand_or_lowercase(self):
        text = "DON’T the-best-food 😊!"
        expected = ["DON’T", "the", "best", "food", "😊", "!"]
        self.assertEqual(surface_tokens(text), expected)
        self.assertEqual(get_tokenizer().surface_tokens(text), expected)
        self.assertEqual(tokenize(text), ["do", "not", "the", "best", "food", "😊"])

    def test_normalization_combines_case_apostrophes_and_contractions(self):
        text = "DON’T eat CAFÉ food! 😊"
        expected = ["do", "not", "eat", "café", "food", "😊"]
        self.assertEqual(get_tokenizer().tokenize(text), expected)
        self.assertEqual(tokenize(text), expected)

    def test_methods_share_unicode_and_case_normalization(self):
        for name in available_tokenizers():
            with self.subTest(method=name):
                self.assertEqual(
                    get_tokenizer(name).tokenize("CAFÉ cafe\u0301 😊"),
                    ["café", "café", "😊"],
                )

    def test_custom_preserves_emoticon_case_and_complete_emoji(self):
        self.assertEqual(
            tokenize("NICE :D :P 👩🏽‍🍳 🇸🇬 1️⃣"),
            ["nice", ":D", ":P", "👩🏽‍🍳", "🇸🇬", "1️⃣"],
        )

    def test_unknown_method_names_have_a_useful_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown tokenizer.*choose from"):
            get_tokenizer("typo")


if __name__ == "__main__":
    unittest.main()
