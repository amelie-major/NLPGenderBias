import pandas as pd
from pathlib import Path

# Inputs
orig_path = Path("data/guardian_articles.csv")
ext_path  = Path("data/guardian_articles_extended.csv")

# Output
out_path = Path("data/guardian_articles_combined_1990_2024_dedup.csv")

# Load
orig_df = pd.read_csv(orig_path)
ext_df  = pd.read_csv(ext_path)

# --- Map ORIGINAL schema -> EXTENDED schema ---
orig_mapped = orig_df.rename(columns={
    "webUrl": "URL",
    "sectionName": "Article Category",
    "webPublicationDate": "Publication Date",
    "webTitle": "Article Title",
    "bodyContent": "Article Contents",
})

# Add columns that exist in extended but not original
orig_mapped["Article Author"] = pd.NA
orig_mapped["Data Quality"]   = pd.NA

# Keep only the extended schema columns in a consistent order
target_cols = [
    "URL",
    "Article Category",
    "Publication Date",
    "Article Author",
    "Article Title",
    "Article Contents",
    "Data Quality",
]
orig_mapped = orig_mapped[target_cols].copy()

# --- Normalize EXTENDED df to match types ---
# (Make sure the extended CSV has the expected columns)
missing = [c for c in target_cols if c not in ext_df.columns]
if missing:
    raise ValueError(f"guardian_articles_extended.csv is missing columns: {missing}")

ext_df = ext_df[target_cols].copy()

# --- Parse dates (handles 2016-01-31T22:30:10Z etc.) ---
orig_mapped["Publication Date"] = pd.to_datetime(orig_mapped["Publication Date"], utc=True, errors="coerce")
ext_df["Publication Date"]      = pd.to_datetime(ext_df["Publication Date"], utc=True, errors="coerce")

# --- Clean URL strings for better dedupe ---
orig_mapped["URL"] = orig_mapped["URL"].astype(str).str.strip()
ext_df["URL"]      = ext_df["URL"].astype(str).str.strip()

# Optional: strip fragments (#...) which can create false "duplicates"
orig_mapped["URL"] = orig_mapped["URL"].str.replace(r"#.*$", "", regex=True)
ext_df["URL"]      = ext_df["URL"].str.replace(r"#.*$", "", regex=True)

# --- Combine ---
combined = pd.concat([ext_df, orig_mapped], ignore_index=True)

# --- Filter to 1990–2024 inclusive ---
start = pd.Timestamp("1990-01-01", tz="UTC")
end   = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
combined = combined.loc[combined["Publication Date"].between(start, end)].copy()

# --- Deduplicate ---
# Keep "best" row per URL:
# 1) higher Data Quality if numeric
# 2) longer Article Contents
# 3) newer Publication Date
combined["_content_len"] = combined["Article Contents"].astype(str).str.len()
combined["_dq_num"] = pd.to_numeric(combined["Data Quality"], errors="coerce")

combined = combined.sort_values(
    by=["URL", "_dq_num", "_content_len", "Publication Date"],
    ascending=[True, False, False, False],
    kind="mergesort"
)
combined = combined.drop_duplicates(subset=["URL"], keep="first")

combined = combined.drop(columns=["_content_len", "_dq_num"])

# --- Save ---
combined.to_csv(out_path, index=False)

print(f"Saved: {out_path} | rows={len(combined):,}")
print("Combined date range:",
      combined["Publication Date"].min(),
      "->",
      combined["Publication Date"].max())