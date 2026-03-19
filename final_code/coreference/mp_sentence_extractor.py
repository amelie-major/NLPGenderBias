"""
MP Sentence Extractor
=====================
Takes tagged CSVs (output of coref_tagger.py) from a directory, splits each
article into sentences, and outputs a CSV of every sentence that mentions at
least one MP — one row per (sentence, MP) pair.

Output columns:
    sentence  — the sentence text (MP tags preserved, e.g. <boris_johnson>)
    mp        — canonical MP name (e.g. "Boris Johnson")
    gender    — MP gender from gazetteer (M/F)
    topic     — topic label derived from the source filename
    date      — article date (if present in source CSV)
    url       — article URL (if present in source CSV)

One CSV is written per topic file into --output-dir.

Usage:
    python mp_sentence_extractor.py
        --tagged-dir tagged_by_topic
        --mps ../mp_gazetteer/mp_gazetteer.csv
        --output-dir mp_sentences_by_topic
"""

import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Matches any <snake_case_tag> in tagged text
TAG_PATTERN = re.compile(r"<([a-z0-9_]+)>")

# Sentence boundary: punctuation followed by whitespace + capital/quote
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z\"\u2018\u2019\u201c\u201d])')


# ---------------------------------------------------------------------------
# MP tag lookup
# ---------------------------------------------------------------------------

def load_tag_to_name(mps_path: str) -> Dict[str, Dict[str, str]]:
    """Build a lookup from tag string (e.g. '<boris_johnson>') to {name, gender}."""
    df = pd.read_csv(mps_path)
    df.columns = [c.strip().lower() for c in df.columns]

    name_col = "name_final" if "name_final" in df.columns else "name"
    if name_col not in df.columns:
        raise ValueError(
            f"MP CSV must have a 'name' or 'name_final' column. Found: {list(df.columns)}"
        )

    has_gender = "gender" in df.columns

    lookup = {}
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        if not name or name == "nan":
            continue

        # Use explicit tag column if present, otherwise auto-generate
        explicit_tag = str(row.get("tag", "")).strip()
        if explicit_tag and explicit_tag != "nan":
            tag = explicit_tag
        else:
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            tag = f"<{slug}>"

        gender = str(row["gender"]).strip() if has_gender else ""
        if gender == "nan":
            gender = ""

        lookup[tag] = {"name": name, "gender": gender}

    logger.info(f"Loaded {len(lookup)} MP tags from {mps_path}")
    return lookup


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Split article text into sentences."""
    # Normalise newlines to spaces first
    text = re.sub(r"\s*\n+\s*", " ", text).strip()
    sentences = SENT_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_mp_sentences(
    tagged_dir: str,
    mps_path: str,
    output_dir: str,
    tagged_col: str = "tagged_text",
):
    tag_to_name = load_tag_to_name(mps_path)

    input_path = Path(tagged_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {tagged_dir}")

    logger.info(f"Found {len(csv_files)} topic CSV(s) in {tagged_dir}")

    for csv_file in csv_files:
        topic = csv_file.stem  # e.g. "0_covid_virus_vaccine_pandemic"
        df = pd.read_csv(csv_file)
        df.columns = [c.strip().lower() for c in df.columns]

        if tagged_col not in df.columns:
            logger.warning(f"Column '{tagged_col}' not found in {csv_file.name} — skipping")
            continue

        logger.info(f"Processing {csv_file.name} ({len(df)} articles)...")

        rows = []
        for _, article in df.iterrows():
            text = str(article.get(tagged_col, ""))
            if not text or text == "nan":
                continue

            date = article.get("date", "")
            url  = article.get("url", "")

            for sentence in split_sentences(text):
                tag_slugs = TAG_PATTERN.findall(sentence)
                if not tag_slugs:
                    continue

                for slug in tag_slugs:
                    full_tag = f"<{slug}>"
                    mp_info = tag_to_name.get(full_tag)
                    if mp_info is None:
                        continue

                    rows.append({
                        "sentence": sentence,
                        "mp":       mp_info["name"],
                        "gender":   mp_info["gender"],
                        "topic":    topic,
                        "date":     date,
                        "url":      url,
                    })

        out_df = pd.DataFrame(rows, columns=["sentence", "mp", "gender", "topic", "date", "url"])
        out_file = output_path / csv_file.name
        out_df.to_csv(out_file, index=False)
        logger.info(f"  Saved {len(out_df)} rows to {out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract sentences mentioning MPs from tagged article CSVs"
    )
    parser.add_argument(
        "--tagged-dir", required=True,
        help="Directory of tagged topic CSVs (output of coref_tagger.py)",
    )
    parser.add_argument("--mps", required=True, help="Path to MP CSV file")
    parser.add_argument("--output-dir", required=True, help="Directory to write per-topic sentence CSVs")
    parser.add_argument(
        "--tagged-col", default="tagged_text",
        help="Column name containing tagged text (default: tagged_text)",
    )

    args = parser.parse_args()

    extract_mp_sentences(
        tagged_dir=args.tagged_dir,
        mps_path=args.mps,
        output_dir=args.output_dir,
        tagged_col=args.tagged_col,
    )


if __name__ == "__main__":
    main()
