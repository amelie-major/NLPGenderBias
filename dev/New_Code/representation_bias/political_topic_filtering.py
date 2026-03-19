import pandas as pd
import re
import os.path
import glob

"""
Filter Guardian articles to those on political topics.
1) Start with opinion articles only.
2) Score articles by presence of political keywords in body text.
3) Add all articles from the "Politics" section.
4) Deduplicate.
"""

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_file = glob.glob(os.path.join(base_dir, "data/original_guardian_articles.csv"))
df = pd.read_csv(csv_file[0])

# 1) Opinion-only
OPINION_SECTIONS = {"Opinion"}  # expand if needed
op = df[df["sectionName"].isin(OPINION_SECTIONS)].copy()

# 2) Political keywords
POLITICAL_TERMS = [
    r"\belection(s)?\b", r"\bvot(e|ing|ers)?\b", r"\bballot(s)?\b",
    r"\bcampaign\b", r"\breferendum\b", r"\bparliament\b", r"\bcongress\b",
    r"\bgovernment\b", r"\bminister(s)?\b", r"\bmp(s)?\b", r"\bsenator(s)?\b",
    r"\bpresident\b", r"\bprime minister\b", r"\bwhite house\b",
    r"\bsupreme court\b", r"\blegislat(ion|ive|ure)\b",
    r"\bpolicy\b", r"\bimmigration\b", r"\btax(es|ation)?\b",
    r"\bparty\b", r"\bcoalition\b", r"\bconservative(s)?\b", r"\blabou?r\b",
    r"\bdemocrat(s|ic)?\b", r"\brepublican(s)?\b",
]

NEGATIVE_PHRASES = [
    r"\boffice politics\b", r"\bworkplace politics\b", r"\bcampus politics\b",
]

politics_re = re.compile("|".join(POLITICAL_TERMS), re.IGNORECASE)
neg_re = re.compile("|".join(NEGATIVE_PHRASES), re.IGNORECASE)

def politics_score(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    if neg_re.search(text):
        return 0
    return len(politics_re.findall(text))

op["politics_score"] = op["bodyContent"].fillna("").map(politics_score)

POLITICS_THRESHOLD = 5
political_opinion_only = op[op["politics_score"] >= POLITICS_THRESHOLD].drop(columns=["politics_score"]).copy()

# 3) Add Politics section articles
politics_section = df[df["sectionName"] == "Politics"].copy()

combined = pd.concat([political_opinion_only, politics_section], ignore_index=True)

# 4) Deduplicate (prefer 'id' if it is unique/stable)
dedupe_key = "id" if "id" in combined.columns else "webUrl"
combined = combined.drop_duplicates(subset=[dedupe_key])

output_file = os.path.join(base_dir, "data", "political_guardian_articles.csv")
combined.to_csv(output_file, index=False)
print("Wrote:", output_file)

print("Opinion articles:", len(op))
print("Political opinion (scored) articles:", len(political_opinion_only))
print("Politics section articles:", len(politics_section))
print("Combined (deduped) articles:", len(combined))

print(op.sort_values("politics_score", ascending=False)[["webTitle","politics_score"]].head(20))
print(op.sort_values("politics_score", ascending=True)[["webTitle","politics_score"]].head(20))