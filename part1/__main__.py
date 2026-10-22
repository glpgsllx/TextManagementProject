"""Interactive tokenizer entry point: python -m part1 --text 'your review'."""

import argparse
import json
import sys

from text_processing import available_tokenizers, get_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Review/query text; otherwise read standard input")
    parser.add_argument("--keep-contractions", action="store_true")
    parser.add_argument("--tokenizer", choices=available_tokenizers(), default="custom")
    args = parser.parse_args()
    text = args.text if args.text is not None else sys.stdin.read()
    tokenizer = get_tokenizer(args.tokenizer, expand_contractions=not args.keep_contractions)
    print(json.dumps({"surface_tokens": tokenizer.surface_tokens(text),
                      "normalized_terms": tokenizer.tokenize(text)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
