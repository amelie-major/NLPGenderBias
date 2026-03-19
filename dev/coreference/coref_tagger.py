"""
MP Coreference Tagger
=====================
Replaces all mentions of MPs in articles (including pronoun coreferences)
with normalized tags like <boris_johnson>.

Usage:
    python coref_tagger.py --mps mps.csv --articles articles.csv --output tagged_articles.csv
    python coref_tagger.py --mps mps.csv --articles articles.json --output tagged_articles.json

Requirements:
    pip install spacy fastcoref pandas rapidfuzz
    python -m spacy download en_core_web_trf   # best accuracy
    # OR: python -m spacy download en_core_web_sm  # faster, lighter

Modes:
    --mode coref    (default) Uses spaCy + fastcoref for full coreference resolution
    --mode ner      Falls back to NER-only matching (no pronoun resolution, but no torch needed)
"""

import argparse
import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
import spacy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MP:
    """Represents a normalized MP entry."""
    canonical_name: str          # e.g. "Boris Johnson"
    tag: str                     # e.g. "<boris_johnson>"
    aliases: list[str] = field(default_factory=list)  # extra name forms

    @staticmethod
    def name_to_tag(name: str) -> str:
        """Convert a name to a tag: 'Boris Johnson' -> '<boris_johnson>'"""
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return f"<{slug}>"


# ---------------------------------------------------------------------------
# MP Loading
# ---------------------------------------------------------------------------

def load_mps(path: str) -> list[MP]:
    """
    Load MPs from a CSV file.

    Expected CSV columns (flexible):
        - 'name' (required): the canonical/normalized name
        - 'aliases' (optional): semicolon-separated alternative names
        - 'tag' (optional): override the auto-generated tag

    Example CSV:
        name,aliases
        Boris Johnson,Boris;Johnson;Alexander Boris de Pfeffel Johnson
        Keir Starmer,Starmer;Sir Keir Starmer
    """
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    if "name" not in df.columns:
        raise ValueError(f"MP CSV must have a 'name' column. Found: {list(df.columns)}")

    mps = []
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        if not name or name == "nan":
            continue

        tag = str(row.get("tag", "")).strip()
        if not tag or tag == "nan":
            tag = MP.name_to_tag(name)

        aliases = []
        raw_aliases = str(row.get("aliases", "")).strip()
        if raw_aliases and raw_aliases != "nan":
            aliases = [a.strip() for a in raw_aliases.split(";") if a.strip()]

        # Auto-generate common aliases from the canonical name
        parts = name.split()
        if len(parts) >= 2:
            # Add surname alone as an alias if not already present
            surname = parts[-1]
            if surname not in aliases and surname != name:
                aliases.append(surname)
            # Add first + last if name has middle parts
            if len(parts) > 2:
                short = f"{parts[0]} {parts[-1]}"
                if short not in aliases:
                    aliases.append(short)

        mps.append(MP(canonical_name=name, tag=tag, aliases=aliases))

    logger.info(f"Loaded {len(mps)} MPs from {path}")
    return mps


# ---------------------------------------------------------------------------
# Article Loading
# ---------------------------------------------------------------------------

def load_articles(path: str, text_column: str = "text", id_column: str = None) -> pd.DataFrame:
    """
    Load articles from CSV or JSON.

    Args:
        path: Path to CSV or JSON file
        text_column: Name of the column containing article text
        id_column: Optional ID column name (auto-detected if not provided)
    """
    p = Path(path)
    if p.suffix.lower() == ".json":
        df = pd.read_json(path)
    elif p.suffix.lower() in (".csv", ".tsv"):
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    text_column = text_column.lower()

    if text_column not in df.columns:
        # Try common alternatives
        for alt in ["text", "body", "content", "article", "article_text", "bodyContent"]:
            if alt in df.columns:
                text_column = alt
                break
        else:
            raise ValueError(
                f"Column '{text_column}' not found. Available: {list(df.columns)}"
            )

    logger.info(f"Loaded {len(df)} articles from {path} (text column: '{text_column}')")
    return df, text_column


# ---------------------------------------------------------------------------
# Name Matching
# ---------------------------------------------------------------------------

def build_name_matcher(mps: list[MP]) -> dict[str, str]:
    """
    Build a lookup dict: lowercased name/alias -> tag.
    Longer names have priority (matched first).
    """
    lookup = {}
    for mp in mps:
        lookup[mp.canonical_name.lower()] = mp.tag
        for alias in mp.aliases:
            alias_lower = alias.lower()
            # Don't overwrite a longer/more-specific name with a short alias
            if alias_lower not in lookup:
                lookup[alias_lower] = mp.tag

    return lookup


def build_sorted_patterns(lookup: dict[str, str]) -> list[tuple[re.Pattern, str]]:
    """
    Build regex patterns sorted longest-first so 'Boris Johnson' matches
    before 'Johnson' alone.
    """
    patterns = []
    # Sort by length descending so longer matches take priority
    for name in sorted(lookup.keys(), key=len, reverse=True):
        tag = lookup[name]
        # Word-boundary match, case-insensitive
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        patterns.append((pattern, tag))
    return patterns


# ---------------------------------------------------------------------------
# Coreference Resolution (fastcoref mode)
# ---------------------------------------------------------------------------

def load_coref_pipeline(spacy_model: str = "en_core_web_trf"):
    """Load spaCy + fastcoref pipeline."""
    try:
        from fastcoref import spacy_component  # noqa: F401
    except ImportError:
        raise ImportError(
            "fastcoref not installed. Install with: pip install fastcoref\n"
            "Or use --mode ner for NER-only matching (no pronoun resolution)."
        )

    nlp = spacy.load(spacy_model)
    nlp.add_pipe("fastcoref")
    logger.info(f"Loaded spaCy model '{spacy_model}' + fastcoref")
    return nlp


def resolve_coref_text(doc) -> str:
    """
    Given a spaCy doc with fastcoref resolved, return text with pronouns
    replaced by their antecedent.
    """
    # fastcoref stores resolved output via doc._.resolved_text
    if hasattr(doc._, "resolved_text") and doc._.resolved_text:
        return doc._.resolved_text
    return doc.text


def tag_article_coref(
    text: str,
    nlp,
    patterns: list[tuple[re.Pattern, str]],
    lookup: dict[str, str],
) -> str:
    """
    Full pipeline: resolve coreferences first, then tag MP mentions.

    Steps:
        1. Run spaCy + fastcoref to resolve pronouns to their antecedents
        2. In the resolved text, replace all MP name mentions with tags
    """
    doc = nlp(text)
    resolved = resolve_coref_text(doc)

    # Now tag all MP mentions in the resolved text
    tagged = resolved
    for pattern, tag in patterns:
        tagged = pattern.sub(tag, tagged)

    return tagged


# ---------------------------------------------------------------------------
# NER-Only Matching (fallback mode)
# ---------------------------------------------------------------------------

def load_ner_pipeline(spacy_model: str = "en_core_web_sm"):
    """Load spaCy NER-only pipeline (no coref)."""
    nlp = spacy.load(spacy_model)
    logger.info(f"Loaded spaCy model '{spacy_model}' (NER-only, no coref)")
    return nlp


def tag_article_ner(
    text: str,
    nlp,
    patterns: list[tuple[re.Pattern, str]],
    lookup: dict[str, str],
) -> str:
    """
    NER-only fallback: just replace name mentions (no pronoun resolution).

    Uses spaCy NER to also catch names that might have slight variations,
    then fuzzy-matches them against the MP list.
    """
    tagged = text

    # Step 1: Use regex patterns for exact/alias matches (longest first)
    for pattern, tag in patterns:
        tagged = pattern.sub(tag, tagged)

    # Step 2 (optional): Use spaCy NER to catch PERSON entities not yet tagged
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            ent_lower = ent.text.lower().strip()
            # Check if already handled by patterns
            if ent_lower in lookup:
                tag = lookup[ent_lower]
                # Replace in output (using the original span position is tricky
                # after prior replacements, so we do a bounded text replace)
                tagged = re.sub(
                    r"\b" + re.escape(ent.text) + r"\b",
                    tag,
                    tagged,
                    flags=re.IGNORECASE,
                )
            else:
                # Try fuzzy matching
                tag = fuzzy_match_mp(ent.text, lookup)
                if tag:
                    tagged = re.sub(
                        r"\b" + re.escape(ent.text) + r"\b",
                        tag,
                        tagged,
                        flags=re.IGNORECASE,
                    )

    return tagged


def fuzzy_match_mp(name: str, lookup: dict[str, str], threshold: int = 85) -> str | None:
    """
    Fuzzy-match a detected name against the MP lookup.
    Returns the tag if a good match is found, else None.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None  # Skip fuzzy matching if rapidfuzz not installed

    name_lower = name.lower().strip()
    best_score = 0
    best_tag = None

    for mp_name, tag in lookup.items():
        score = fuzz.token_sort_ratio(name_lower, mp_name)
        if score > best_score:
            best_score = score
            best_tag = tag

    if best_score >= threshold:
        return best_tag
    return None


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def process_articles(
    mps_path: str,
    articles_path: str,
    output_path: str,
    text_column: str = "text",
    mode: str = "coref",
    spacy_model: str = None,
    batch_size: int = 50,
):
    """
    Main entry point: load data, process articles, save output.
    """
    # Load MPs
    mps = load_mps(mps_path)
    lookup = build_name_matcher(mps)
    patterns = build_sorted_patterns(lookup)

    logger.info(f"Built {len(patterns)} name patterns for matching")

    # Load articles
    df, text_col = load_articles(articles_path, text_column)

    # Load NLP pipeline
    if mode == "coref":
        model = spacy_model or "en_core_web_trf"
        nlp = load_coref_pipeline(model)
        tag_fn = lambda text: tag_article_coref(text, nlp, patterns, lookup)
    else:
        model = spacy_model or "en_core_web_sm"
        nlp = load_ner_pipeline(model)
        tag_fn = lambda text: tag_article_ner(text, nlp, patterns, lookup)

    # Process
    logger.info(f"Processing {len(df)} articles in '{mode}' mode...")
    tagged_texts = []
    for i, text in enumerate(df[text_col]):
        if pd.isna(text) or not str(text).strip():
            tagged_texts.append("")
            continue

        try:
            tagged = tag_fn(str(text))
            tagged_texts.append(tagged)
        except Exception as e:
            logger.warning(f"Error processing article {i}: {e}")
            tagged_texts.append(str(text))  # Keep original on error

        if (i + 1) % batch_size == 0:
            logger.info(f"  Processed {i + 1}/{len(df)} articles")

    df["tagged_text"] = tagged_texts

    # Save output
    out_path = Path(output_path)
    if out_path.suffix.lower() == ".json":
        df.to_json(output_path, orient="records", indent=2)
    else:
        df.to_csv(output_path, index=False)

    logger.info(f"Saved {len(df)} tagged articles to {output_path}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tag MP mentions in articles using coreference resolution"
    )
    parser.add_argument("--mps", required=True, help="Path to MP CSV file")
    parser.add_argument("--articles", required=True, help="Path to articles CSV/JSON")
    parser.add_argument("--output", required=True, help="Output file path (CSV or JSON)")
    parser.add_argument("--text-column", default="text", help="Column name with article text")
    parser.add_argument(
        "--mode",
        choices=["coref", "ner"],
        default="coref",
        help="'coref' = spaCy + fastcoref (needs torch), 'ner' = NER-only fallback",
    )
    parser.add_argument(
        "--spacy-model",
        default=None,
        help="spaCy model name (default: en_core_web_trf for coref, en_core_web_sm for ner)",
    )
    parser.add_argument("--batch-log", type=int, default=50, help="Log progress every N articles")

    args = parser.parse_args()

    process_articles(
        mps_path=args.mps,
        articles_path=args.articles,
        output_path=args.output,
        text_column=args.text_column,
        mode=args.mode,
        spacy_model=args.spacy_model,
        batch_size=args.batch_log,
    )


if __name__ == "__main__":
    main()
