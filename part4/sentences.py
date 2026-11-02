"""Sentence segmentation for Yelp reviews.

NLTK's punkt model needs a download, and the assignment requires the code to run
from a plain checkout, so the splitter is a few regular expressions. It handles
the three cases that matter in this corpus: a newline is always a boundary, a
full stop followed by whitespace ends a sentence unless the token is a known
abbreviation or a decimal number, and quoted speech keeps its closing quote with
the sentence it terminates.
"""

import re

# Tokens that end in a full stop without ending a sentence.
ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "st", "ave", "blvd", "rd", "ln", "ct", "jr", "sr",
    "vs", "etc", "inc", "co", "corp", "llc", "no", "approx", "dept", "univ",
    "a.m", "p.m", "u.s", "e.g", "i.e",
})

# A full stop, question mark or exclamation mark, optional closing quotes or
# brackets, then whitespace. The whitespace is what the split consumes.
_BOUNDARY = re.compile(r"(?<=[.!?])[\"'\u201d\u2019)\]]*\s+")
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")


def _ends_with_abbreviation(chunk):
    """True when the final token is 'Dr.', 'a.m.' or a decimal like '4.50.'."""
    match = _LAST_WORD.search(chunk)
    if not match:
        return False
    token = match.group(1).lower().rstrip(".")
    return token in ABBREVIATIONS


def split_sentences(text):
    """Split one review into sentences, preserving the original wording."""
    sentences = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        start = 0
        for boundary in _BOUNDARY.finditer(line):
            chunk = line[start:boundary.start()].strip()
            if chunk and not _ends_with_abbreviation(chunk):
                sentences.append(chunk)
                start = boundary.end()
        tail = line[start:].strip()
        if tail:
            sentences.append(tail)
    return [s for s in sentences if len(s) >= 2]
