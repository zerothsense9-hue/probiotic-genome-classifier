import pandas as pd
from pathlib import Path
from itertools import product
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[1]

TRAINING_FILE = BASE_DIR / "data" / "training_accessions_balanced.csv"
FASTA_DIR = BASE_DIR / "genomes" / "fna_files"
OUTPUT_FILE = BASE_DIR / "features" / "kmer_features_k5_balanced.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

K = 5
NUCLEOTIDES = ["A", "C", "G", "T"]

def read_fasta_sequence(fasta_path):
    sequence_parts = []

    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            sequence_parts.append(line.upper())

    return "".join(sequence_parts)

def generate_all_kmers(k):
    return ["".join(p) for p in product(NUCLEOTIDES, repeat=k)]

def extract_kmer_frequencies(sequence, k, all_kmers):
    counts = Counter()
    total = 0

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]

        if set(kmer).issubset({"A", "C", "G", "T"}):
            counts[kmer] += 1
            total += 1

    if total == 0:
        return {kmer: 0 for kmer in all_kmers}

    return {kmer: counts[kmer] / total for kmer in all_kmers}

df = pd.read_csv(TRAINING_FILE)
df["label"] = df["label"].astype(int)

all_kmers = generate_all_kmers(K)
records = []

missing = []

for index, row in df.iterrows():
    accession = row["accession"]
    label = row["label"]

    fasta_path = FASTA_DIR / f"{accession}.fna"

    if not fasta_path.exists():
        print(f"Missing FASTA: {accession}")
        missing.append(accession)
        continue

    print(f"[{index + 1}/{len(df)}] Processing {accession}")

    sequence = read_fasta_sequence(fasta_path)
    features = extract_kmer_frequencies(sequence, K, all_kmers)

    record = {
        "accession": accession,
        "label": label,
        "sequence_length": len(sequence)
    }

    record.update(features)
    records.append(record)

features_df = pd.DataFrame(records)
features_df.to_csv(OUTPUT_FILE, index=False)

print("\nDone.")
print("Total accessions in balanced list:", len(df))
print("Genomes processed:", len(features_df))
print("Missing FASTA files:", len(missing))
print("Features created:", len(features_df.columns) - 3)
print("Saved to:", OUTPUT_FILE)

print("\nLabel counts in feature file:")
print(features_df["label"].value_counts())