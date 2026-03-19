"""
Sample n random sentences from sentences.csv.

Usage:
    python sample_sentences.py 100
    python sample_sentences.py 100 -i sentences.csv -o sample.csv
    python sample_sentences.py 100 --seed 42
"""

import argparse
import csv
import random


def sample_sentences(input_path: str, output_path: str, n: int, seed: int | None) -> None:
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if n > len(rows):
        print(f"Warning: requested {n} but only {len(rows)} sentences available. Returning all.")
        n = len(rows)

    random.seed(seed)
    sampled = random.sample(rows, n)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["index", "text", "label"])
        for row in sampled:
            writer.writerow([row["index"], row["text"], row["label"]])

    print(f"Done. {n} sentences written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample n random sentences from sentences.csv.")
    parser.add_argument("n", type=int, help="Number of sentences to sample")
    parser.add_argument("-i", "--input", default="sentences.csv", help="Input CSV (default: sentences.csv)")
    parser.add_argument("-o", "--output", default="sample.csv", help="Output CSV (default: sample.csv)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    sample_sentences(args.input, args.output, args.n, args.seed)
