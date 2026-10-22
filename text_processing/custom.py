"""Ordered regex boundaries for noisy reviews, written for this assignment.

The regex library supplies Unicode properties; all token boundary rules live here.
Rule order keeps complete emoji and emoticons ahead of words and punctuation.
"""

from functools import lru_cache

import regex


WORD = r"(?=[\p{L}\p{M}\p{N}]*\p{L})[\p{L}\p{M}\p{N}]+(?:['’‘ʼ][\p{L}\p{M}\p{N}]+)*"
NUMBER = r"\p{N}+(?:[.,]\p{N}+)*"
# Complete sequences take priority over individual characters and punctuation.
PICTOGRAPH = r"\p{Extended_Pictographic}[\uFE0E\uFE0F]?\p{Emoji_Modifier}?"
EMOJI = (
    r"(?:"
    r"\U0001F3F4[\U000E0020-\U000E007E]+\U000E007F"
    r"|\p{Regional_Indicator}{2}"
    r"|[\#*0-9]\uFE0F?\u20E3"
    r"|" + PICTOGRAPH + r"(?:\u200D" + PICTOGRAPH + r")*"
    r")"
)
EMOTICON = r"(?:[:;=][\-^']?[)(DPp/\\]|(?<![\p{L}\p{N}])<3(?!\p{N}))"

RULES = [
    ("EMOJI", EMOJI),
    ("EMOTICON", EMOTICON),
    ("WORD", WORD),
    ("NUMBER", NUMBER),
    ("DASH", r"\p{Pd}+"),
    ("OTHER", r"\S"),
]
TOKEN_PATTERN = regex.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in RULES),
    regex.VERSION1,
)
WORD_PATTERN = regex.compile(r"(?:" + WORD + r")\Z", regex.VERSION1)
EMOJI_PATTERN = regex.compile(EMOJI + r"\Z", regex.VERSION1)
EMOTICON_PATTERN = regex.compile(EMOTICON + r"\Z", regex.VERSION1)
HAS_LETTER_OR_NUMBER = regex.compile(r"[\p{L}\p{N}]", regex.VERSION1)


def scan(text: str) -> list[str]:
    """Scan NFC text; keep case/punctuation and discard structural dashes.

    Public callers use Tokenizer.surface_tokens(), which prepares NFC text.
    """
    return [
        match.group()
        for match in TOKEN_PATTERN.finditer(text)
        if match.lastgroup != "DASH"
    ]


@lru_cache(maxsize=100_000)
def is_emoji(token: str) -> bool:
    return EMOJI_PATTERN.fullmatch(token) is not None


@lru_cache(maxsize=100_000)
def is_emoticon(token: str) -> bool:
    return EMOTICON_PATTERN.fullmatch(token) is not None
