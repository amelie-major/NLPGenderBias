"""
Split articles from a topic CSV into individual sentences.

Output format:
    index,text,label
    article_1,"Sentence from article.",0
    ...

Usage:
    python split_into_sentences.py articles_by_topic/Brexit_Referendum.csv
    python split_into_sentences.py articles_by_topic/Brexit_Referendum.csv -o output.csv
"""

import argparse
import csv
from pathlib import Path

import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize


def split_articles_to_sentences(input_path: str, output_path: str) -> None:
    sentences = []

    with open(input_path, newline="", encoding="utf-8") as f:
        for article in csv.DictReader(f):
            body = article.get("bodyContent", "").strip()
            if not body:
                continue
            for sentence in sent_tokenize(body):
                sentence = sentence.strip()
                if sentence:
                    sentences.append(sentence)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["index", "text", "label"])
        for i, sentence in enumerate(sentences, start=1):
            writer.writerow([f"article_{i}", sentence, 0])

    print(f"Done. {len(sentences)} sentences written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split articles into sentences.")
    parser.add_argument("input", help="Path to input CSV (e.g. articles_by_topic/Brexit_Referendum.csv)")
    parser.add_argument("-o", "--output", help="Path to output CSV (default: <input_stem>_sentences.csv)")
    args = parser.parse_args()

    p = Path(args.input)
    output_path = args.output or str(p.parent / (p.stem + "_sentences.csv"))

    split_articles_to_sentences(args.input, output_path)
