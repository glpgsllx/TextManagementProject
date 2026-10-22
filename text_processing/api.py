"""One small interface for the three tokenization methods used by the team."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer

from .custom import scan
from .normalization import normalize_tokens, prepare_text


_SURFACE_TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "custom": scan,
    "whitespace": WhitespaceTokenizer().tokenize,
    "wordpunct": WordPunctTokenizer().tokenize,
}


@dataclass(frozen=True)
class Tokenizer:
    """A boundary method plus the same normalization policy for all methods.

    Use surface_tokens() to inspect case, punctuation and boundary decisions.
    Use tokenize() for normalized statistics or query/document index terms.
    """

    name: str
    _surface_tokenizer: Callable[[str], list[str]] = field(repr=False)
    expand_contractions: bool = True

    def surface_tokens(self, text: str) -> list[str]:
        return self._surface_tokenizer(prepare_text(text))

    def tokenize(self, text: str) -> list[str]:
        return normalize_tokens(
            self.surface_tokens(text), expand=self.expand_contractions
        )


def available_tokenizers() -> tuple[str, ...]:
    """Return method names accepted by get_tokenizer()."""
    return tuple(_SURFACE_TOKENIZERS)


@lru_cache(maxsize=6)
def get_tokenizer(
    name: str = "custom", *, expand_contractions: bool = True
) -> Tokenizer:
    """Select a boundary method; instances are immutable and safe to reuse."""
    if name not in _SURFACE_TOKENIZERS:
        raise ValueError(
            f"Unknown tokenizer {name!r}; choose from {available_tokenizers()}"
        )
    return Tokenizer(name, _SURFACE_TOKENIZERS[name], expand_contractions)


def tokenize(text: str, *, expand_contractions: bool = True) -> list[str]:
    """Default custom tokenizer, shared by statistics and search teammates."""
    return get_tokenizer(expand_contractions=expand_contractions).tokenize(text)


def surface_tokens(text: str) -> list[str]:
    """Inspect the default custom tokenizer before term normalization."""
    return get_tokenizer().surface_tokens(text)
