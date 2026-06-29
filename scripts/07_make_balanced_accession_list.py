import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

CLEAN_FILE = BASE_DIR / "data" / "training_accessions_clean.csv"
EXTRA_NEG_FILE = BASE_DIR / "data" / "extra_negative_accessions.csv"
OUTPUT_FILE = BASE_DIR / "data" / "training_accessions_balanced.csv"

clean_df = pd.read_csv(CLEAN_FILE)
extra_df = pd.read_csv(EXTRA_NEG_FILE)

clean_df["label"] = clean_df["label"].astype(int)
extra_df["label"] = extra_df["label"].astype(int)

combined = pd.concat([clean_df, extra_df], ignore_index=True, sort=False)

combined = combined.drop_duplicates(subset=["accession"], keep="first")

combined.to_csv(OUTPUT_FILE, index=False)

print("Done.")
print("Balanced accession list saved to:", OUTPUT_FILE)
print("\nLabel counts:")
print(combined["label"].value_counts())
print("\nTotal genomes:", len(combined))