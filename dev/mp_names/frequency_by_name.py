import pandas as pd
import re
import matplotlib.pyplot as plt
# pip install pyahocorasick
import ahocorasick

articles_df = pd.read_csv("data/political_guardian_articles.csv")
df = pd.read_csv("commons_mps_gender_1990_2024.csv")

TITLES_PATTERN = re.compile(r"^(mr|mrs|ms|dr|sir|dame|lord|baroness)\s+", re.I)
PEERAGE_PATTERN = re.compile(r"\s+of\s+.*$", re.I)

def normalize_name(name: str) -> str:
    name = name.strip()
    name = TITLES_PATTERN.sub("", name)
    name = PEERAGE_PATTERN.sub("", name)
    name = re.sub(r"\s+", " ", name)
    return name.title()

df["name_normalized"] = df["name"].astype(str).apply(normalize_name)

# Build automaton of names (lowercased)
A = ahocorasick.Automaton()
names = df["name_normalized"].dropna().unique().tolist()
names_lc = [n.lower() for n in names]

for n in names_lc:
    A.add_word(n, n)
A.make_automaton()

# Concatenate all text once
corpus = " ".join(articles_df["bodyContent"].dropna().astype(str)).lower()

# Count matches with word-boundary checks
counts = {n: 0 for n in names_lc}
for end_idx, matched in A.iter(corpus):
    start_idx = end_idx - len(matched) + 1

    # "word boundary" around the whole phrase:
    left_ok = (start_idx == 0) or (not corpus[start_idx - 1].isalnum())
    right_ok = (end_idx + 1 == len(corpus)) or (not corpus[end_idx + 1].isalnum())

    if left_ok and right_ok:
        counts[matched] += 1

df["frequency"] = df["name_normalized"].str.lower().map(counts).fillna(0).astype(int)

df_to_plot = df.sort_values(by="frequency", ascending=False).head(20)
plt.figure(figsize=(12, 6))
plt.barh(df_to_plot["name_normalized"], df_to_plot["frequency"], color='skyblue')
plt.xlabel("Frequency in Articles")
plt.title("Top 20 Most Mentioned MPs in Political Guardian Articles (1990-2024)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
df.to_csv("data/mps_frequency_1990_2024.csv", index=False)