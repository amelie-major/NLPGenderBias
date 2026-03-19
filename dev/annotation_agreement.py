import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report
)

# --- Load data ---

human = pd.read_csv("sample_labelled_human_annotation.csv")
llm = pd.read_csv("sample_labelled_llm.csv")
bert = pd.read_csv("sample_labelled_BERT.txt", sep="\t")

# Human and LLM are aligned by article index; BERT is aligned by row position.
# Verify the CSV files share the same article order.
assert list(human["index"]) == list(llm["index"]), "Human and LLM article indices don't match!"

y_human = human["label"].astype(int).values
y_llm   = llm["label"].astype(int).values
y_bert  = bert["prediction"].astype(int).values

assert len(y_human) == len(y_bert), (
    f"Row count mismatch: human={len(y_human)}, bert={len(y_bert)}"
)

# --- Helper ---

def compare(name_a, y_a, name_b, y_b):
    print(f"\n{'='*60}")
    print(f"  {name_a}  vs  {name_b}")
    print(f"{'='*60}")
    print(f"  Accuracy : {accuracy_score(y_a, y_b):.4f}")
    print(f"  Precision: {precision_score(y_a, y_b, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_a, y_b, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(y_a, y_b, zero_division=0):.4f}")
    print(f"  Cohen's κ: {cohen_kappa_score(y_a, y_b):.4f}")
    print()
    cm = confusion_matrix(y_a, y_b)
    print(f"  Confusion matrix (rows={name_a}, cols={name_b}):")
    print(f"           Pred 0  Pred 1")
    print(f"  True 0:  {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"  True 1:  {cm[1,0]:5d}   {cm[1,1]:5d}")
    print()
    print(classification_report(y_a, y_b, target_names=["No causality (0)", "Causality (1)"],
                                 zero_division=0))

# Human is treated as ground truth for LLM and BERT comparisons.
# For inter-annotator agreement between LLM and BERT, neither is ground truth.

compare("Human", y_human, "LLM",  y_llm)
compare("Human", y_human, "BERT", y_bert)
compare("LLM",   y_llm,   "BERT", y_bert)
