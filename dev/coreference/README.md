# MP Coreference Tagger

Replaces all mentions of MPs in articles—including pronoun coreferences—with normalized tags.

**Example:**
```
Input:  "Boris Johnson went to parliament. He gave a speech."
Output: "<boris_johnson> went to parliament. <boris_johnson> gave a speech."
```

## Setup

```bash
pip install -r requirements.txt

# For coref mode (recommended - resolves pronouns):
python -m spacy download en_core_web_trf

# For NER-only mode (lighter, no pronoun resolution):
python -m spacy download en_core_web_sm
```

## Usage

### Full coreference mode (resolves "he", "she", "they" etc.)
```bash
python coref_tagger.py \
    --mps sample_mps.csv \
    --articles sample_articles.csv \
    --output tagged_articles.csv \
    --mode coref
```

### NER-only mode (just replaces name mentions, no pronoun resolution)
```bash
python coref_tagger.py \
    --mps sample_mps.csv \
    --articles sample_articles.csv \
    --output tagged_articles.csv \
    --mode ner
```

### With JSON input/output
```bash
python coref_tagger.py \
    --mps mps.csv \
    --articles articles.json \
    --output tagged_articles.json \
    --mode coref
```

## MP CSV Format

| Column  | Required | Description                              |
|---------|----------|------------------------------------------|
| name    | Yes      | Canonical name (e.g. "Boris Johnson")    |
| aliases | No       | Semicolon-separated alternatives         |
| tag     | No       | Override auto-generated tag              |

Aliases for surname and first+last are auto-generated from the canonical name.

**Example:**
```csv
name,aliases
Boris Johnson,Boris;BoJo;Alexander Boris de Pfeffel Johnson
Keir Starmer,Starmer;Sir Keir Starmer
```

## Articles CSV/JSON Format

Must have a text column (auto-detects `text`, `body`, `content`, `article`).
Override with `--text-column your_column_name`.

## How It Works

### Coref mode (`--mode coref`)
1. Runs spaCy + fastcoref to resolve pronouns to their antecedents
2. In the resolved text, replaces all MP name mentions with `<tag>` tokens
3. Longest-match-first prevents "Johnson" from matching inside "Boris Johnson"

### NER mode (`--mode ner`)
1. Regex matching against MP names/aliases (longest first)
2. spaCy NER catches PERSON entities and fuzzy-matches them to the MP list
3. No pronoun resolution—only explicit name mentions are tagged

## Python API

```python
from coref_tagger import process_articles

df = process_articles(
    mps_path="mps.csv",
    articles_path="articles.csv",
    output_path="tagged.csv",
    mode="coref",  # or "ner"
)
# df now has a 'tagged_text' column
```

## Notes

- **Coref mode** needs PyTorch + `en_core_web_trf` (~1.5 GB). Much better accuracy for pronoun resolution.
- **NER mode** needs only `en_core_web_sm` (~12 MB). Fast but won't resolve "He said..." to the right MP.
- `rapidfuzz` is optional—enables fuzzy matching for name typos/variations in NER mode.
- The script preserves all original columns and adds `tagged_text` alongside.
