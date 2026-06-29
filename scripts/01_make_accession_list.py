import pandas as pd
import re
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "Filtered_Probiotic_Dataset.xlsx"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Sheets and labels
sheets = {
    "Probiotic_Strains": 1,
    "Non_Probiotic_Strains": 0,
    "Unknown_Strains": "unknown"
}

records = []

def extract_accession(value):
    """
    Extract NCBI genome accession such as:
    GCF_000423445.1 or GCA_017504145.1
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    match = re.search(r"(GC[AF]_\d+\.\d+)", text)
    if match:
        return match.group(1)

    return None

for sheet_name, label in sheets.items():
    df = pd.read_excel(DATA_FILE, sheet_name=sheet_name)

    for _, row in df.iterrows():
        accession = extract_accession(row.get("Genome"))

        if accession:
            records.append({
                "accession": accession,
                "sheet": sheet_name,
                "label": label,
                "species": row.get("Species"),
                "strain": row.get("Strain"),
                "pmid": row.get("PMID"),
                "acid": row.get("Resistant to acid"),
                "bile": row.get("Resistant to bile"),
                "adhesion": row.get("Adhesion/Attachment"),
                "antimicrobial": row.get("Antimicrobial"),
                "immunomodulation": row.get("Immunomodulation"),
                "antiproliferative": row.get("Antiproliferative"),
                "antioxidant": row.get("Antioxidant"),
            })

result = pd.DataFrame(records)

# Save all extracted accessions
result.to_csv(OUTPUT_DIR / "all_genome_accessions.csv", index=False)

# Separate known training data from unknown data
known = result[result["label"].isin([0, 1])].copy()
unknown = result[result["label"] == "unknown"].copy()

# Find conflicting genome IDs that appear as both probiotic and non-probiotic
label_counts = known.groupby("accession")["label"].nunique()
conflicting_ids = label_counts[label_counts > 1].index.tolist()

conflicts = known[known["accession"].isin(conflicting_ids)]
clean_known = known[~known["accession"].isin(conflicting_ids)]

# Remove duplicate accessions within same label
clean_known = clean_known.drop_duplicates(subset=["accession", "label"])
unknown = unknown.drop_duplicates(subset=["accession"])

# Save outputs
clean_known.to_csv(OUTPUT_DIR / "training_accessions_clean.csv", index=False)
unknown.to_csv(OUTPUT_DIR / "unknown_accessions.csv", index=False)
conflicts.to_csv(OUTPUT_DIR / "conflicting_accessions_review.csv", index=False)

print("Done.")
print("All genome accession rows:", len(result))
print("Clean training genomes:", len(clean_known))
print("Unknown genomes:", len(unknown))
print("Conflicting genomes needing review:", len(conflicts))

print("\nTraining label counts:")
print(clean_known["label"].value_counts())