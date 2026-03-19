# dedupe_coref_snippets.py
# Dedupe your *coreference output* CSVs in coreference_snippets_by_topic/
# - Removes exact duplicates (same person + same snippet)
# - Removes "contained" near-duplicates (short snippet fully contained in a longer one)
# - Works per-topic CSV, and (optionally) per person to avoid cross-person effects

import re
import pandas as pd
from pathlib import Path

IN_DIR = Path("coreference_snippets_by_topic")
OUT_DIR = Path("coreference_snippets_by_topic_deduped")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Configure these to match your schema ----
PERSON_COL = "person"   # change if your column is called "mp", "name", etc.
SNIPPET_COL = "snippet"

# If you want to dedupe within each person separately (recommended)
DEDUPE_WITHIN_PERSON = True

# Normalise snippet text before dedupe (recommended)
NORMALISE_WHITESPACE = True
LOWERCASE_FOR_MATCH = False   # keep False unless casing is noisy

def normalise_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00a0", " ")  # NBSP
    if NORMALISE_WHITESPACE:
        s = re.sub(r"\s+", " ", s).strip()
    if LOWERCASE_FOR_MATCH:
        s = s.lower()
    return s

def remove_contained_snippets(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """
    Remove rows where df[key_col] is fully contained within another longer value.
    Keeps the longest version(s).
    Complexity is O(n^2) per group; OK for moderate group sizes.
    """
    df = df.copy()

    # Work on unique strings to reduce comparisons
    texts = df[key_col].tolist()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)

    kept = []
    kept_texts = []

    for idx in order:
        t = texts[idx]
        # If t is contained in any already-kept longer text, drop it
        if any(t and (t in big) and (len(t) < len(big)) for big in kept_texts):
            continue
        kept.append(idx)
        kept_texts.append(t)

    # kept is in length-desc order; restore original row order
    kept_index = sorted(kept)
    return df.iloc[kept_index]

def dedupe_one_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if SNIPPET_COL not in df.columns:
        raise ValueError(f"{path.name}: missing required column '{SNIPPET_COL}'")
    if PERSON_COL not in df.columns and DEDUPE_WITHIN_PERSON:
        raise ValueError(f"{path.name}: missing required column '{PERSON_COL}' (needed for per-person dedupe)")

    # Create a normalised version for matching (don’t overwrite original snippet)
    df["_snippet_norm"] = df[SNIPPET_COL].apply(normalise_text)

    # 1) Drop exact duplicates on (person, snippet_norm) or just (snippet_norm)
    subset = [PERSON_COL, "_snippet_norm"] if (DEDUPE_WITHIN_PERSON and PERSON_COL in df.columns) else ["_snippet_norm"]
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    after_exact = len(df)

    # 2) Drop contained near-duplicates (within-person recommended)
    if DEDUPE_WITHIN_PERSON and PERSON_COL in df.columns:
        df = (
            df.groupby(PERSON_COL, group_keys=False)
              .apply(lambda g: remove_contained_snippets(g, "_snippet_norm"))
              .reset_index(drop=True)
        )
    else:
        df = remove_contained_snippets(df, "_snippet_norm").reset_index(drop=True)

    after_contained = len(df)

    # Clean up helper column
    df = df.drop(columns=["_snippet_norm"])

    print(
        f"{path.name}: {before} -> {after_exact} after exact dedupe -> {after_contained} after contained dedupe "
        f"(removed {before - after_contained})"
    )
    return df

def main():
    files = sorted(IN_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {IN_DIR.resolve()}")

    for f in files:
        out_path = OUT_DIR / f.name
        deduped = dedupe_one_file(f)
        deduped.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

if __name__ == "__main__":
    main()