import pandas as pd
import spacy
import re
from pathlib import Path

# ---- Load spaCy coref model ----
nlp = spacy.load("en_coreference_web_trf")

# ---- Helpers: build regex targets for many MPs ----
def _name_to_pattern(name: str) -> str:
    parts = [re.escape(p) for p in name.strip().split()]
    # return raw pattern string (not compiled) so we can OR-combine
    return r"\b" + r"\s+".join(parts) + r"\b"

def build_targets(
    mp_names: list[str],
    aliases: dict[str, list[str]] | None = None
) -> dict[str, re.Pattern]:
    aliases = aliases or {}
    targets: dict[str, re.Pattern] = {}

    for canon in mp_names:
        variants = [canon] + aliases.get(canon, [])
        pats = [_name_to_pattern(v) for v in variants]
        combo = r"(?:" + r"|".join(pats) + r")"
        targets[canon] = re.compile(combo, re.I)

    return targets


def entity_snippets(
    text: str,
    sent_window: int,
    targets: dict[str, re.Pattern],
) -> dict[str, list[str]]:
    doc = nlp(text)

    # Ensure sentence boundaries exist (the coref model usually has sentencizer, but be safe)
    sents = list(doc.sents)
    if not sents:
        return {canon: [] for canon in targets}

    # token index -> sentence index
    tok2sent: dict[int, int] = {}
    for si, s in enumerate(sents):
        for t in range(s.start, s.end):
            tok2sent[t] = si

    # Collect coref clusters (spaCy experimental stores them in doc.spans)
    clusters = [
        spangroup for key, spangroup in doc.spans.items()
        if key.startswith("coref_clusters_")
    ]

    out = {canon: [] for canon in targets}
    seen = {canon: set() for canon in targets}  # dedupe by (lo, hi)

    for spangroup in clusters:
        if not spangroup:
            continue

        # Which targets are explicitly mentioned in this cluster?
        hit_targets = [
            canon for canon, pat in targets.items()
            if any(pat.search(m.text) for m in spangroup)
        ]
        if not hit_targets:
            continue

        # Add snippets for each mention span in the cluster (incl pronouns)
        for mention in spangroup:
            si = tok2sent.get(mention.start)
            if si is None:
                continue

            lo = max(0, si - sent_window)
            hi = min(len(sents), si + sent_window + 1)
            snippet = " ".join(s.text.strip() for s in sents[lo:hi]).strip()

            span_key = (lo, hi)
            for canon in hit_targets:
                if span_key in seen[canon]:
                    continue
                seen[canon].add(span_key)
                out[canon].append(snippet)

    return out


def snippets_to_rows(
    articles: pd.Series,
    targets: dict[str, re.Pattern],
    gender_by_name: dict[str, str],
    sent_window: int = 1
):
    rows = []

    for article_idx, body_text in enumerate(articles):
        if not isinstance(body_text, str) or not body_text.strip():
            continue

        print(f"Processing article {article_idx}...")
        by_entity = entity_snippets(body_text, sent_window=sent_window, targets=targets)

        for person, snippets in by_entity.items():
            if not snippets:
                continue
            gender = gender_by_name.get(person, "unknown")
            for snip_idx, snippet in enumerate(snippets, 1):
                rows.append({
                    "person": person,
                    "article_index": article_idx,
                    "snippet_index": snip_idx,
                    "snippet": snippet,
                    "gender": gender,
                })

    return rows


def print_top_snippets(df: pd.DataFrame, max_per_person: int = 3):
    if df.empty:
        print("No snippets found.")
        return

    for person, grp in df.groupby("person", sort=True):
        grp = grp.sort_values(["article_index", "snippet_index"])
        top = grp.head(max_per_person)
        if top.empty:
            continue
        print(f"\nEntity: {person} (Gender: {top['gender'].iloc[0]})")
        for i, snip in enumerate(top["snippet"].tolist(), 1):
            print(f"  {i}. {snip}")


# ---- Load MP names + genders once ----
mps_df = pd.read_csv("data/mps_frequency_cropped_1990_2024.csv")
names = mps_df["names_normalized"].astype(str).tolist()
genders = mps_df["gender"].astype(str).tolist()

gender_by_name = dict(zip(names, genders))

ALIASES = {
    "Boris Johnson": ["BoJo", "Alexander Johnson"],
}

TARGETS = build_targets(names, aliases=ALIASES)
print(f"Compiled {len(TARGETS)} target patterns.")

# ---- Process per-topic CSVs ----
in_dir = Path("gb_news_data")
out_dir = Path("coreference_snippets_by_topic")
out_dir.mkdir(parents=True, exist_ok=True)

topic_files = sorted(in_dir.glob("*.csv"))
if not topic_files:
    raise FileNotFoundError(f"No topic CSVs found in: {in_dir.resolve()}")

for file in topic_files:
    topic = file.stem
    if topic == "Brexit_Referendum" or topic == "Covid-19" or topic == "Labour_Party":
        print(f"Skipping topic: {topic} already processed")
        continue  # skip these topics for now due to noisy data
    print(f"\nProcessing topic: {topic}")

    topic_df = pd.read_csv(file)

    if "bodyContent" not in topic_df.columns:
        raise KeyError(f"{file.name} missing required column: bodyContent")

    rows = snippets_to_rows(
        topic_df["bodyContent"],
        targets=TARGETS,
        gender_by_name=gender_by_name,
        sent_window=1
    )

    df_out = pd.DataFrame(rows)
    print_top_snippets(df_out, max_per_person=3)

    df_out.to_csv(out_dir / f"{topic}_snippets.csv", index=False)
    print(f"Saved: {out_dir / f'{topic}_snippets.csv'}")
