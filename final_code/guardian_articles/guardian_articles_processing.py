import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_SECTIONS = {"opinion", "politics"}
YEAR_MIN, YEAR_MAX = 1990, 2024

# ---------------------------------------------------------------------------
# Load guardian_articles.csv
# schema: article_id, sectionName, webTitle, webUrl, bodyContent,
#         webPublicationDate, id
# ---------------------------------------------------------------------------
guardian_path = os.path.join(BASE_DIR, "guardian_articles.csv")
df_guardian = pd.read_csv(guardian_path, low_memory=False)

df_guardian = df_guardian.rename(columns={
    "webUrl":             "url",
    "sectionName":        "section",
    "webTitle":           "title",
    "bodyContent":        "content",
    "webPublicationDate": "date",
    "article_id":         "article_id",
})

df_guardian["source"] = "guardian_articles"

# ---------------------------------------------------------------------------
# Load articledb_part1.csv … articledb_part6.csv
# schema: URL, Article Category, Publication Date, Article Author,
#         Article Title, Article Contents, Data Quality
# ---------------------------------------------------------------------------
part_frames = []
for i in range(1, 7):
    path = os.path.join(BASE_DIR, f"articledb_part{i}.csv")
    if os.path.exists(path):
        part_frames.append(pd.read_csv(path, low_memory=False))
    else:
        print(f"Warning: {path} not found, skipping.")

df_parts = pd.concat(part_frames, ignore_index=True) if part_frames else pd.DataFrame()

df_parts = df_parts.rename(columns={
    "URL":               "url",
    "Article Category":  "section",
    "Publication Date":  "date",
    "Article Author":    "author",
    "Article Title":     "title",
    "Article Contents":  "content",
    "Data Quality":      "data_quality",
})

df_parts["source"] = "articledb"

# ---------------------------------------------------------------------------
# Align to a common schema and merge
# ---------------------------------------------------------------------------
KEEP_COLS = ["url", "section", "date", "title", "content", "author", "source"]

for col in KEEP_COLS:
    if col not in df_guardian.columns:
        df_guardian[col] = None
    if col not in df_parts.columns:
        df_parts[col] = None

df = pd.concat(
    [df_guardian[KEEP_COLS], df_parts[KEEP_COLS]],
    ignore_index=True,
)

# ---------------------------------------------------------------------------
# Filter: section must be Opinion or Politics (case-insensitive)
# ---------------------------------------------------------------------------
df["section_lower"] = df["section"].str.strip().str.lower()
df = df[df["section_lower"].isin(TARGET_SECTIONS)].copy()
df.drop(columns=["section_lower"], inplace=True)

# ---------------------------------------------------------------------------
# Filter: publication year 1990–2024
# ---------------------------------------------------------------------------
df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
df = df[df["date_parsed"].dt.year.between(YEAR_MIN, YEAR_MAX)].copy()

# ---------------------------------------------------------------------------
# Filter: drop articles with no content
# ---------------------------------------------------------------------------
df = df[df["content"].notna() & (df["content"].str.strip() != "")].copy()

# ---------------------------------------------------------------------------
# Deduplicate: prefer rows with a URL; fall back to title+content
# ---------------------------------------------------------------------------
before = len(df)

# Exact URL duplicates (keep first)
df = df[df["url"].isna() | ~df["url"].duplicated(keep="first")]

# Content-level duplicates where URL is missing
mask_no_url = df["url"].isna()
content_key = df.loc[mask_no_url, "content"].str.strip().str[:200]
df = df[~(mask_no_url & content_key.duplicated(keep="first"))]

after = len(df)
print(f"Removed {before - after} duplicates ({before} → {after} rows)")

# ---------------------------------------------------------------------------
# Sort and export
# ---------------------------------------------------------------------------
df = df.sort_values("date_parsed").drop(columns=["date_parsed"]).reset_index(drop=True)

out_path = os.path.join(BASE_DIR, "guardian_combined.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} articles to {out_path}")

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
# Re-parse date for stats (already dropped date_parsed above)
date_parsed = pd.to_datetime(df["date"], errors="coerce", utc=True)

print("\n" + "=" * 50)
print("DATASET STATS")
print("=" * 50)

print(f"\nColumns: {list(df.columns)}")

print(f"\nTotal articles : {len(df):,}")
print(f"Unique URLs    : {df['url'].nunique():,}")
print(f"Missing URLs   : {df['url'].isna().sum():,}")

print(f"\nDate range     : {date_parsed.min().date()} → {date_parsed.max().date()}")
print(f"Missing dates  : {date_parsed.isna().sum():,}")

print("\nArticles by section:")
for section, count in df["section"].value_counts().items():
    print(f"  {section:<20} {count:>6,}")

print("\nArticles by source:")
for source, count in df["source"].value_counts().items():
    print(f"  {source:<20} {count:>6,}")

print("\nArticles by year:")
year_counts = date_parsed.dt.year.value_counts().sort_index()
for year, count in year_counts.items():
    print(f"  {int(year)}  {count:>6,}")

print("\nMissing values per column:")
for col, n in df.isna().sum().items():
    pct = 100 * n / len(df)
    print(f"  {col:<20} {n:>6,}  ({pct:.1f}%)")

avg_len = df["content"].dropna().str.len().mean()
med_len = df["content"].dropna().str.len().median()
print(f"\nContent length (chars)  avg: {avg_len:,.0f}  median: {med_len:,.0f}")

print("\nTop 10 authors:")
for author, count in df["author"].value_counts().head(10).items():
    print(f"  {str(author):<35} {count:>5,}")

print("=" * 50)
