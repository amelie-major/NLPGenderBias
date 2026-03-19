"""
MP Coreference Tagger
=====================
Replaces all mentions of MPs in articles (including pronoun coreferences)
with normalized tags like <boris_johnson>.

Usage:
    # Single file:
    python coref_tagger.py --mps mps.csv --articles articles.csv --output tagged_articles.csv

    # Per-topic (processes all CSVs in a directory):
    python coref_tagger.py --mps mps.csv --topic-dir articles_by_topic/ --output-dir tagged_by_topic/

Requirements:
    pip install spacy fastcoref pandas rapidfuzz
    python -m spacy download en_core_web_sm

Modes:
    --mode coref    (default) Uses spaCy + fastcoref for full coreference resolution
    --mode ner      Falls back to NER-only matching (no pronoun resolution, but no torch needed)
"""

import argparse
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
import spacy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ambiguous single words that happen to be surnames — never auto-generate
# these as standalone aliases. Users can still force them via the CSV aliases
# column if they know what they're doing.
# ---------------------------------------------------------------------------

AMBIGUOUS_SURNAMES = {
    # Months / time words
    "may", "march", "august", "january", "april", "june",
    # Very common English nouns/adjectives that are also surnames
    "price", "young", "king", "long", "short", "green", "brown", "black",
    "white", "grey", "gray", "fox", "hunt", "field", "ball", "bell",
    "bird", "bond", "bush", "camp", "church", "cook", "cross", "dale",
    "dean", "drew", "duke", "wood", "heath", "hill", "hope", "lamb",
    "law", "lee", "love", "page", "park", "pike", "potter", "power",
    "rice", "rich", "rose", "rule", "rush", "sharp", "smart", "smith",
    "snow", "stone", "strong", "swift", "ward", "waters", "winter",
    "wise", "noble", "cash", "best", "frost", "grant", "glass", "banks",
    "berry", "bolt", "brake", "burn", "close", "cole", "cotton", "crane",
    "day", "fair", "farmer", "fisher", "ford", "gold", "grove", "hand",
    "hardy", "hart", "horn", "house", "lake", "lane", "little", "mark",
    "marsh", "miller", "more", "new", "north", "odd", "old", "palm",
    "post", "prime", "ray", "read", "reed", "rest", "root", "sale",
    "shell", "shore", "silk", "spring", "steel", "still", "storm",
    "sweet", "vine", "wall", "way", "west", "wild", "will", "wolf",
    "worth", "bacon", "barton", "mason", "mills", "moon",
}


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
    df.columns = [c.strip().lower() for c in df.columns]

    if "name_final" not in df.columns:
        raise ValueError(f"MP CSV must have a 'name_final' column. Found: {list(df.columns)}")

    mps = []
    for _, row in df.iterrows():
        name = str(row["name_final"]).strip()
        if not name or name == "nan":
            continue

        tag = str(row.get("tag", "")).strip()
        if not tag or tag == "nan":
            tag = MP.name_to_tag(name)

        aliases = []
        raw_aliases = str(row.get("aliases", "")).strip()
        if raw_aliases and raw_aliases != "nan":
            aliases = [a.strip() for a in raw_aliases.split(";") if a.strip()]

        # Auto-generate surname alias ONLY if it's not ambiguous
        parts = name.split()
        if len(parts) >= 2:
            surname = parts[-1]
            if (
                surname not in aliases
                and surname != name
                and surname.lower() not in AMBIGUOUS_SURNAMES
            ):
                aliases.append(surname)
            # First + last if name has middle parts
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
    p = Path(path)
    if p.suffix.lower() == ".json":
        df = pd.read_json(path)
    elif p.suffix.lower() in (".csv", ".tsv"):
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    df.columns = [c.strip().lower() for c in df.columns]
    text_column = text_column.lower()

    if text_column not in df.columns:
        for alt in ["text", "body", "content", "article", "article_text"]:
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
# Name Matching — split into SAFE (multi-word) and NER-GATED (single-word)
# ---------------------------------------------------------------------------

def build_name_matcher(mps: list[MP]) -> dict[str, str]:
    """Build a full lookup dict: lowercased name/alias -> tag."""
    lookup = {}
    for mp in mps:
        lookup[mp.canonical_name.lower()] = mp.tag
        for alias in mp.aliases:
            alias_lower = alias.lower()
            if alias_lower not in lookup:
                lookup[alias_lower] = mp.tag
    return lookup


def build_split_patterns(lookup: dict[str, str], mps: list[MP]):
    """
    Split the lookup into two groups:

    1. safe_patterns: multi-word names — always applied via regex
       (e.g. "Boris Johnson", "Sir Keir Starmer")

    2. ner_gated_lookup: single-word names — ONLY applied when:
       (a) spaCy NER confirms the word is a PERSON entity, AND
       (b) the MP's full name has already appeared in the same article.

       Maps: lowercased single word -> set of ALL possible tags (multiple
       MPs can share a surname, e.g. "Harris" -> {<carolyn_harris>, ...}).
       Built directly from the MP list to avoid deduplication issues in
       the flat lookup dict.

    Returns:
        (safe_patterns, ner_gated_lookup)
    """
    safe_patterns = []
    ner_gated_lookup = {}  # str -> set[str]

    # Build safe patterns from the flat lookup (multi-word only)
    for name in sorted(lookup.keys(), key=len, reverse=True):
        tag = lookup[name]
        if len(name.split()) >= 2:
            pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
            safe_patterns.append((pattern, tag))

    # Build NER-gated lookup directly from MP list to capture ALL MPs per surname
    for mp in mps:
        # Check canonical name's individual words
        parts = mp.canonical_name.lower().split()
        for part in parts:
            if part not in ner_gated_lookup:
                ner_gated_lookup[part] = set()
            ner_gated_lookup[part].add(mp.tag)

        # Check single-word aliases
        for alias in mp.aliases:
            if len(alias.split()) == 1:
                key = alias.lower()
                if key not in ner_gated_lookup:
                    ner_gated_lookup[key] = set()
                ner_gated_lookup[key].add(mp.tag)

    # Remove any single words that are also covered by a multi-word safe pattern
    # (e.g. don't NER-gate "boris" if "boris johnson" is already a safe pattern)
    # Actually keep them — they're only used when the full name is already found,
    # so they handle subsequent surname-only references.

    logger.info(
        f"Built {len(safe_patterns)} safe (multi-word) patterns + "
        f"{len(ner_gated_lookup)} NER-gated (single-word) patterns"
    )
    return safe_patterns, ner_gated_lookup


def apply_safe_patterns(text: str, safe_patterns: list[tuple[re.Pattern, str]]) -> tuple[str, set[str]]:
    """
    Apply multi-word name patterns — always safe, no false positives.

    Returns:
        (tagged_text, found_tags): the tagged text and the set of MP tags
        whose full names were matched. found_tags is used downstream to
        gate single-word surname matching.
    """
    found_tags = set()
    for pattern, tag in safe_patterns:
        if pattern.search(text):
            found_tags.add(tag)
        text = pattern.sub(tag, text)
    return text, found_tags


def apply_ner_gated_replacements(
    original_text: str,
    tagged_text: str,
    nlp,
    ner_gated_lookup: dict[str, set[str]],
    found_tags: set[str],
) -> str:
    """
    For single-word names, only replace when BOTH conditions are met:
      (a) spaCy NER confirms the word is a PERSON entity
      (b) The MP's full name was already found in this article

    Condition (b) prevents "Harris" from matching <carolyn_harris> in an
    article about Kamala Harris where Carolyn Harris was never mentioned.
    If a surname maps to multiple MPs (e.g. both Carolyn Harris and
    another Harris), only the one(s) whose full name appeared are used.
    If a surname is ambiguous (maps to multiple found MPs), it is skipped
    to avoid guessing.

    Args:
        original_text: the text to run NER on
        tagged_text: text after safe patterns have already been applied
        nlp: spaCy pipeline with NER
        ner_gated_lookup: single-word lowercased name -> set of possible tags
        found_tags: tags whose full names were already matched in this article
    """
    if not ner_gated_lookup or not found_tags:
        return tagged_text

    doc = nlp(original_text)

    # Collect all words that NER thinks are part of PERSON entities
    confirmed_person_words = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            confirmed_person_words.add(ent.text)
            for word in ent.text.split():
                confirmed_person_words.add(word)

    # Replace single-word aliases ONLY when:
    #   1. NER confirmed it as a person word
    #   2. Exactly one candidate MP's full name was already found in this article
    for person_word in confirmed_person_words:
        key = person_word.lower().strip()
        if key not in ner_gated_lookup:
            continue

        candidate_tags = ner_gated_lookup[key]
        # Intersect with tags whose full name appeared in this article
        valid_tags = candidate_tags & found_tags

        if len(valid_tags) == 1:
            # Unambiguous: exactly one MP with this surname was mentioned by full name
            tag = next(iter(valid_tags))
            tagged_text = re.sub(
                r"\b" + re.escape(person_word) + r"\b",
                tag,
                tagged_text,
            )
        elif len(valid_tags) > 1:
            # Ambiguous: multiple MPs with this surname were mentioned — skip
            logger.debug(
                f"Skipping ambiguous surname '{person_word}' — "
                f"matches {len(valid_tags)} found MPs: {valid_tags}"
            )
        # len(valid_tags) == 0: surname's MP was never mentioned by full name — skip

    return tagged_text


# ---------------------------------------------------------------------------
# Coreference Resolution (fastcoref mode)
# ---------------------------------------------------------------------------

def load_coref_pipeline(spacy_model: str = "en_core_web_sm"):
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
    """Return text with pronouns replaced by their antecedent."""
    if hasattr(doc._, "resolved_text") and doc._.resolved_text:
        return doc._.resolved_text
    return doc.text


def tag_article_coref(
    text: str,
    nlp_coref,
    nlp_ner,
    safe_patterns: list[tuple[re.Pattern, str]],
    ner_gated_lookup: dict[str, set[str]],
) -> str:
    """
    Full coref pipeline:
        1. Run fastcoref to resolve pronouns -> antecedents
        2. Apply safe multi-word patterns (always)
        3. Apply single-word patterns only where NER confirms PERSON
           AND the MP's full name was already found in the article
    """
    # Step 1: Resolve coreferences
    doc = nlp_coref(text)
    resolved = resolve_coref_text(doc)

    # Step 2: Apply safe multi-word patterns — also collect which MPs were found
    tagged, found_tags = apply_safe_patterns(resolved, safe_patterns)

    # Step 3: NER-gated single-word replacements, scoped to found MPs
    tagged = apply_ner_gated_replacements(
        resolved, tagged, nlp_ner, ner_gated_lookup, found_tags
    )

    return tagged


# ---------------------------------------------------------------------------
# NER-Only Mode (fallback, no coref)
# ---------------------------------------------------------------------------

def load_ner_pipeline(spacy_model: str = "en_core_web_sm"):
    """Load spaCy pipeline for NER-only mode."""
    nlp = spacy.load(spacy_model)
    logger.info(f"Loaded spaCy model '{spacy_model}' (NER-only, no coref)")
    return nlp


def tag_article_ner(
    text: str,
    nlp,
    safe_patterns: list[tuple[re.Pattern, str]],
    ner_gated_lookup: dict[str, set[str]],
    full_lookup: dict[str, str],
) -> str:
    """
    NER-only fallback:
        1. Apply safe multi-word patterns
        2. Apply NER-gated single-word patterns (scoped to found MPs)
        3. Fuzzy-match remaining PERSON entities
    """
    # Step 1: Safe multi-word patterns
    tagged, found_tags = apply_safe_patterns(text, safe_patterns)

    # Step 2: NER-gated single-word patterns, scoped to found MPs
    tagged = apply_ner_gated_replacements(
        text, tagged, nlp, ner_gated_lookup, found_tags
    )

    # Step 3: Fuzzy-match remaining PERSON entities
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            ent_lower = ent.text.lower().strip()
            if ent_lower not in full_lookup:
                tag = fuzzy_match_mp(ent.text, full_lookup)
                if tag:
                    tagged = re.sub(
                        r"\b" + re.escape(ent.text) + r"\b",
                        tag,
                        tagged,
                        flags=re.IGNORECASE,
                    )

    return tagged


def fuzzy_match_mp(name: str, lookup: dict[str, str], threshold: int = 85) -> str | None:
    """Fuzzy-match a detected name against the MP lookup."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None

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

def _build_tag_fn(mps_path, mode, spacy_model):
    """Load MPs and NLP pipelines once, return (tag_fn, full_lookup)."""
    mps = load_mps(mps_path)
    full_lookup = build_name_matcher(mps)
    safe_patterns, ner_gated_lookup = build_split_patterns(full_lookup, mps)

    if mode == "coref":
        coref_model = spacy_model or "en_core_web_sm"
        nlp_coref = load_coref_pipeline(coref_model)
        nlp_ner = spacy.load(spacy_model or "en_core_web_sm")
        logger.info("Loaded separate NER pipeline for single-word validation")

        tag_fn = lambda text: tag_article_coref(
            text, nlp_coref, nlp_ner, safe_patterns, ner_gated_lookup
        )
    else:
        model = spacy_model or "en_core_web_sm"
        nlp = load_ner_pipeline(model)
        tag_fn = lambda text: tag_article_ner(
            text, nlp, safe_patterns, ner_gated_lookup, full_lookup
        )

    return tag_fn


def _tag_dataframe(df, text_col, tag_fn, batch_size=50):
    """Apply tag_fn to every article in a DataFrame, returning it with a 'tagged_text' column."""
    logger.info(f"Processing {len(df)} articles...")
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
            tagged_texts.append(str(text))

        if (i + 1) % batch_size == 0:
            logger.info(f"  Processed {i + 1}/{len(df)} articles")

    df["tagged_text"] = tagged_texts
    return df


def _save_df(df, output_path):
    """Save DataFrame to CSV or JSON based on file extension."""
    out_path = Path(output_path)
    if out_path.suffix.lower() == ".json":
        df.to_json(output_path, orient="records", indent=2)
    else:
        df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} tagged articles to {output_path}")


def process_articles(
    mps_path: str,
    articles_path: str,
    output_path: str,
    text_column: str = "text",
    mode: str = "coref",
    spacy_model: str = None,
    batch_size: int = 50,
):
    """Process a single articles file."""
    tag_fn = _build_tag_fn(mps_path, mode, spacy_model)
    df, text_col = load_articles(articles_path, text_column)
    df = _tag_dataframe(df, text_col, tag_fn, batch_size)
    _save_df(df, output_path)
    return df


def process_topic_dir(
    mps_path: str,
    topic_dir: str,
    output_dir: str,
    text_column: str = "text",
    mode: str = "coref",
    spacy_model: str = None,
    batch_size: int = 50,
):
    """Process all CSV/JSON files in topic_dir, saving tagged versions to output_dir."""
    topic_path = Path(topic_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load pipelines once, reuse across all topic files
    tag_fn = _build_tag_fn(mps_path, mode, spacy_model)

    topic_files = sorted(
        f for f in topic_path.iterdir()
        if f.suffix.lower() in (".csv", ".tsv", ".json")
    )

    if not topic_files:
        logger.warning(f"No CSV/TSV/JSON files found in {topic_dir}")
        return

    logger.info(f"Found {len(topic_files)} topic files in {topic_dir}")

    for topic_file in topic_files:
        logger.info(f"--- Processing topic: {topic_file.name} ---")
        df, text_col = load_articles(str(topic_file), text_column)
        df = _tag_dataframe(df, text_col, tag_fn, batch_size)

        output_file = out_path / topic_file.name
        _save_df(df, str(output_file))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tag MP mentions in articles using coreference resolution"
    )
    parser.add_argument("--mps", required=True, help="Path to MP CSV file")
    parser.add_argument("--text-column", default="content", help="Column name with article text")
    parser.add_argument(
        "--mode",
        choices=["coref", "ner"],
        default="coref",
        help="'coref' = spaCy + fastcoref (needs torch), 'ner' = NER-only fallback",
    )
    parser.add_argument(
        "--spacy-model",
        default=None,
        help="spaCy model name (default: en_core_web_sm)",
    )
    parser.add_argument("--batch-log", type=int, default=50, help="Log progress every N articles")

    # Single-file mode
    parser.add_argument("--articles", default=None, help="Path to a single articles CSV/JSON")
    parser.add_argument("--output", default=None, help="Output file path (CSV or JSON)")

    # Per-topic directory mode
    parser.add_argument("--topic-dir", default=None, help="Directory of per-topic article files")
    parser.add_argument("--output-dir", default=None, help="Directory for tagged output files")

    args = parser.parse_args()

    if args.topic_dir:
        if not args.output_dir:
            parser.error("--output-dir is required when using --topic-dir")
        process_topic_dir(
            mps_path=args.mps,
            topic_dir=args.topic_dir,
            output_dir=args.output_dir,
            text_column=args.text_column,
            mode=args.mode,
            spacy_model=args.spacy_model,
            batch_size=args.batch_log,
        )
    elif args.articles:
        if not args.output:
            parser.error("--output is required when using --articles")
        process_articles(
            mps_path=args.mps,
            articles_path=args.articles,
            output_path=args.output,
            text_column=args.text_column,
            mode=args.mode,
            spacy_model=args.spacy_model,
            batch_size=args.batch_log,
        )
    else:
        parser.error("Provide either --articles (single file) or --topic-dir (per-topic directory)")


if __name__ == "__main__":
    main()