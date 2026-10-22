"""Shared term normalization for custom and baseline tokenizers.

Keeping the postprocessing identical makes comparisons about token boundaries.
"""

from functools import lru_cache
import unicodedata

from .custom import HAS_LETTER_OR_NUMBER, WORD_PATTERN, is_emoji, is_emoticon


APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})
# These are lexical exceptions, not a learned vocabulary or an external model.
IRREGULAR = {
    "can't": ("can", "not"),
    "cannot": ("can", "not"),
    "won't": ("will", "not"),
    "shan't": ("shall", "not"),
    "let's": ("let", "us"),
    "y'all": ("you", "all"),
}
NEGATIVE_BASES = frozenset(
    "am is are was were have has had do does did can could should would must "
    "need dare might may ought".split()
)
SUFFIXES = (("'re", "are"), ("'ve", "have"), ("'ll", "will"))


def prepare_text(text: str) -> str:
    """NFC is shared by custom and baseline tokenizers; source files stay raw."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return unicodedata.normalize("NFC", text)


@lru_cache(maxsize=100_000)
def expand_word(word: str) -> tuple[str, ...]:
    """Expand clear auxiliaries; retain ambiguous possessive/'s/'d/ain't forms."""
    if word in IRREGULAR:
        return IRREGULAR[word]
    if word == "i'm":
        return ("i", "am")
    # Recursive suffix processing also covers wouldn't've -> would not have.
    for suffix, expansion in SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix):
            return expand_word(word[: -len(suffix)]) + (expansion,)
    if word.endswith("n't") and word[:-3] in NEGATIVE_BASES:
        return (word[:-3], "not")
    return (word,)


@lru_cache(maxsize=200_000)
def normalize_token(token: str, expand: bool = True) -> tuple[str, ...]:
    """Apply identical postprocessing to each tokenizer's complete tokens.

    Keep tokens containing letters/numbers and complete emoji/emoticons.
    Discard other punctuation-only tokens. A whitespace token such as food!
    remains food!, so boundary errors remain visible in baseline comparisons.
    """
    token = unicodedata.normalize("NFC", token).translate(APOSTROPHES)
    # Case can encode the mouth of an ASCII emoticon; preserve it verbatim.
    if is_emoji(token) or is_emoticon(token):
        return (token,)
    token = token.lower()
    if not HAS_LETTER_OR_NUMBER.search(token):
        return ()
    if expand and WORD_PATTERN.fullmatch(token):
        return expand_word(token)
    return (token,)


def normalize_tokens(tokens: list[str], *, expand: bool = True) -> list[str]:
    return [term for token in tokens for term in normalize_token(token, expand)]
