from __future__ import annotations

import argparse
from pathlib import Path

from structured_summaries.highlights import (
    convert_highlights_to_cards,
    load_highlights,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--backend", default="claude")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--card-format", choices=["qa", "cloze"], default="qa")
    parser.add_argument("--max-cards", type=int, default=20)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "anki_cards.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    highlights = load_highlights(args.input_file)
    output_csv, raw_response = convert_highlights_to_cards(
        title=args.title,
        highlights=highlights,
        backend=args.backend,
        model=args.model,
        output_csv=args.output_csv,
        card_format=args.card_format,
        max_cards=args.max_cards,
    )
    raw_path = output_csv.with_suffix(".raw.txt")
    raw_path.write_text(raw_response, encoding="utf-8")
    print(f"Saved cards to {output_csv}")
    print(f"Saved raw model response to {raw_path}")


if __name__ == "__main__":
    main()
