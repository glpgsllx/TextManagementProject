"""Part 4: a sentence-level detector for comparisons between businesses.

The assignment (Section 3.5) asks for a small NLP/IR tool. This one finds the
sentences in which a reviewer compares the business under review with another
business or place, and extracts the comparison target ("better than ___").
"""

__all__ = ["detect", "data", "sentences", "patterns", "evaluate"]

from .detect import BusinessComparisonDetector, Comparison

__version__ = "1.0.0"
