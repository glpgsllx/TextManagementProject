"""The comparison detector itself.

A sentence is a *business comparison* when it (1) contains a comparison cue,
(2) does not match one of the known non-comparison idioms, and (3) names another
business or leaves a target phrase that points at one ("the one downtown",
"other places"). Superlatives with a scope ("best pizza in town") are returned
as a separate class because they compare with every other business implicitly.
"""

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from .patterns import BUSINESS_NOUN, FAMILIES, NON_BUSINESS, PLACE_TARGET

# Chains that reviewers name but that are not in the 1,045-business corpus.
# Exact matches only: a partial name would turn "sub" into "Subway".
CHAIN_ALIASES = (
    "pizza hut", "domino's", "dominos", "papa john's", "papa johns", "little caesars",
    "mcdonald's", "mcdonalds", "mcd", "burger king", "wendy's", "wendys", "taco bell",
    "kfc", "kentucky fried chicken", "subway", "chick-fil-a", "chick fil a",
    "arby's", "arbys", "sonic drive-in", "sonic", "jack in the box", "carl's jr",
    "hardee's", "del taco", "in-n-out", "in n out", "five guys", "smashburger",
    "habit burger", "fatburger", "whataburger", "culver's", "culvers",
    "jimmy john's", "jimmy johns", "firehouse subs", "jersey mike's", "jersey mikes",
    "potbelly", "quiznos", "subway", "panda express", "chipotle", "qdoba",
    "costa vida", "cafe rio", "café rio", "baja fresh", "rubio's", "pei wei",
    "noodles & company", "noodles and company", "panera", "panera bread",
    "starbucks", "dutch bros", "big city coffee", "olive garden", "red lobster",
    "applebee's", "applebees", "chili's", "chilis", "tgi fridays", "fridays",
    "denny's", "dennys", "ihop", "village inn", "shari's", "black bear diner",
    "cracker barrel", "outback", "texas roadhouse", "red robin", "buffalo wild wings",
    "hooters", "cheesecake factory", "p.f. chang's", "pf changs", "buca di beppo",
    "carrabba's", "carrabbas", "maggiano's", "benihana", "golden corral", "sizzler",
    "chuck-a-rama", "chuck a rama", "sizzler", "round table pizza", "pizza pipeline",
)

_CUE_ORDER = (
    r"\bthan\b",
    r"\bcompared (?:to|with)\b",
    r"\bin comparison (?:to|with)\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bsame as\b",
    r"\bas (?:good|bad|nice) as\b",
    r"\bsimilar to\b",
    r"\bcomparable to\b",
    r"\bprefer\b.{0,40}\bover\b",
)


def normalize(text):
    """Lowercase, unify apostrophes and collapse whitespace."""
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u02bc", "'")
    text = text.replace("`", "'")
    return " ".join(text.lower().split())


def name_key(text):
    """Reduce a business name to the form it takes inside a normalized sentence."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = text.replace("&", " and ")
    text = text.replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class Comparison:
    sentence: str
    kind: str            # "business", "superlative" or "other"
    families: tuple      # which cue families fired
    comparator: str      # text after the comparison cue
    targets: tuple       # other businesses named in the sentence


class BusinessLexicon:
    """Exact, longest-first matching of the corpus business names in a sentence."""

    def __init__(self, names, aliases=CHAIN_ALIASES):
        forms = {name_key(n) for n in names}
        for alias in aliases:
            key = name_key(alias)
            forms.add(key)
            if key and not key.endswith("s"):
                forms.add(key + "s")      # "other Chipotles", "the Subways in town"
        forms = {f for f in forms if len(f) >= 4}
        ordered = sorted(forms, key=len, reverse=True)
        self._pattern = re.compile(
            r"(?<![a-z0-9])(" + "|".join(map(re.escape, ordered)) + r")(?![a-z0-9])"
        ) if ordered else None
        self._keys = forms

    def find(self, text):
        """Return the business names mentioned in a normalized sentence."""
        if self._pattern is None:
            return set()
        return {m.group(1) for m in self._pattern.finditer(name_key(text))}

    def has(self, name):
        return name_key(name) in self._keys


def comparator_span(text):
    """The phrase that follows the comparison cue: the thing compared against."""
    for pattern in _CUE_ORDER:
        match = re.search(pattern, text)
        if match:
            return text[match.end():].strip(" ,.;:!?\"'")[:90]
    return ""


class BusinessComparisonDetector:
    """Rule-based sentence classifier; see the module docstring for the rules."""

    def __init__(self, business_names, aliases=CHAIN_ALIASES):
        self.lexicon = BusinessLexicon(business_names, aliases)

    def analyze(self, sentence, own_business):
        """Return a Comparison for a cue-bearing sentence, else None."""
        text = normalize(sentence)
        if len(text) < 12 or len(text.split()) < 3:
            return None
        families = tuple(name for name, rx in FAMILIES.items() if rx.search(text))
        if not families:
            return None
        comparator = comparator_span(text)
        targets = tuple(sorted(self.lexicon.find(text) - {name_key(own_business)}))
        if NON_BUSINESS.search(text):
            kind = "other"
        elif targets:
            kind = "business"
        elif PLACE_TARGET.search(comparator) or BUSINESS_NOUN.search(comparator):
            kind = "business"
        elif "superlative" in families:
            kind = "superlative"
        else:
            kind = "other"
        return Comparison(sentence, kind, families, comparator, targets)
