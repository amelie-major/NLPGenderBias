import json
from pathlib import Path
import pandas as pd

in_path = Path("gbnews_corpus/data/extracted_jsonl/gbnews_politics.jsonl")
out_path = Path("gbnews_corpus/data/extracted_jsonl/gbnews_politics_flat.csv")

records = []
with in_path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

df = pd.json_normalize(records, sep=".")
df.to_csv(out_path, index=False)
print(f"✅ Wrote {len(df)} rows to {out_path}")