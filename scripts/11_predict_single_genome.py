import sys
import joblib
import pandas as pd
from pathlib import Path
from itertools import product
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = BASE_DIR / "models" / "xgboost_k5_balanced_model.pkl"
FEATURE_COLUMNS_FILE = BASE_DIR / "models" / "k5_balanced_feature_columns.pkl"

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


def predict_genome(fasta_path):
    model = joblib.load(MODEL_FILE)
    feature_columns = joblib.load(FEATURE_COLUMNS_FILE)

    sequence = read_fasta_sequence(fasta_path)

    all_kmers = generate_all_kmers(K)
    features = extract_kmer_frequencies(sequence, K, all_kmers)

    X = pd.DataFrame([features])

    # Ensure same feature order as training
    X = X[feature_columns]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    non_probiotic_confidence = probability[0] * 100
    probiotic_confidence = probability[1] * 100

    print("\nPrediction Result")
    print("=================")
    print("Input file:", fasta_path)
    print("Sequence length:", len(sequence))

    if prediction == 1:
        print("Prediction: Likely Probiotic")
        print(f"Confidence: {probiotic_confidence:.2f}%")
    else:
        print("Prediction: Likely Non-probiotic / Pathogenic")
        print(f"Confidence: {non_probiotic_confidence:.2f}%")

    print("\nClass probabilities:")
    print(f"Non-probiotic: {non_probiotic_confidence:.2f}%")
    print(f"Probiotic:     {probiotic_confidence:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python scripts\\11_predict_single_genome.py path_to_genome.fna")
        sys.exit(1)

    fasta_file = Path(sys.argv[1])

    if not fasta_file.exists():
        print("Error: FASTA file not found:", fasta_file)
        sys.exit(1)

    predict_genome(fasta_file)