#!/usr/bin/env python3
"""
Organise a CSV by date, extract the top N newest articles, and write them to a new CSV.

Your columns:
URL, Article Category, Publication Date, Article Author, Article Title, Article Contents, Data Quality

Usage:
  python extract_newest_articles.py -i input.csv -o newest_602.csv
Optional:
  python extract_newest_articles.py -i input.csv -o newest_1000.csv --n 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


DEFAULT_DATE_COL = "webPublicationDate"
DEFAULT_N = 602


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="Path to input CSV")
    p.add_argument("-o", "--output", required=True, help="Path to output CSV")
    p.add_argument("--date-col", default=DEFAULT_DATE_COL, help="Date column name")
    p.add_argument("--n", type=int, default=DEFAULT_N, help="How many newest articles to keep")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(in_path)

    if args.date_col not in df.columns:
        print(
            f"ERROR: Date column '{args.date_col}' not found.\n"
            f"Available columns: {list(df.columns)}",
            file=sys.stderr,
        )
        return 1

    # Parse dates robustly (handles ISO strings, 'YYYY-MM-DD', etc.)
    df["_parsed_date"] = pd.to_datetime(df[args.date_col], errors="coerce", utc=True)

    # Drop rows where date couldn't be parsed
    before = len(df)
    df = df.dropna(subset=["_parsed_date"])
    dropped = before - len(df)

    if len(df) == 0:
        print("ERROR: No rows with parseable dates.", file=sys.stderr)
        return 1

    # Sort newest first and take top N
    newest = df.sort_values("_parsed_date", ascending=False).head(args.n)

    # Optional: sort the extracted set by date (newest -> oldest) for readability
    newest = newest.sort_values("_parsed_date", ascending=False)

    # Remove helper column and save
    newest = newest.drop(columns=["_parsed_date"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    newest.to_csv(out_path, index=False)

    print(f"Saved {len(newest)} newest articles to: {out_path}")
    if dropped:
        print(f"Note: Dropped {dropped} rows with unparseable '{args.date_col}' values.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())