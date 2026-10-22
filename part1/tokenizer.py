"""Compatibility imports for Part 1; teammates should use text_processing.

The original names stay available so existing scripts and notebooks keep working.
The implementation now lives in the shared package used by every project part.
"""

from text_processing.api import surface_tokens as scan, tokenize
from text_processing.custom import (
    EMOJI,
    EMOJI_PATTERN,
    EMOTICON,
    EMOTICON_PATTERN,
    HAS_LETTER_OR_NUMBER,
    NUMBER,
    PICTOGRAPH,
    RULES,
    TOKEN_PATTERN,
    WORD,
    WORD_PATTERN,
    is_emoji,
    is_emoticon,
)
from text_processing.normalization import (
    APOSTROPHES,
    IRREGULAR,
    NEGATIVE_BASES,
    SUFFIXES,
    expand_word,
    normalize_token,
    normalize_tokens,
    prepare_text,
)
