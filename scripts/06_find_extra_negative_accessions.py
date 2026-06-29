import subprocess
import re
import time
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATASETS_EXE = BASE_DIR / "datasets.exe"

CLEAN_TRAINING_FILE = BASE_DIR / "data" / "training_accessions_clean.csv"
UNKNOWN_FILE = BASE_DIR / "data" / "unknown_accessions.csv"

OUTPUT_FILE = BASE_DIR / "data" / "extra_negative_accessions.csv"

PER_TAXON = 5

pathogenic_taxa = [
    "Salmonella enterica",
    "Shigella flexneri",
    "Shigella dysenteriae",
    "Shigella sonnei",
    "Vibrio cholerae",
    "Vibrio parahaemolyticus",
    "Vibrio vulnificus",
    "Yersinia pestis",
    "Yersinia enterocolitica",
    "Listeria monocytogenes",
    "Staphylococcus aureus",
    "Streptococcus pyogenes",
    "Streptococcus pneumoniae",
    "Streptococcus agalactiae",
    "Clostridioides difficile",
    "Clostridium botulinum",
    "Clostridium perfringens",
    "Clostridium tetani",
    "Bacillus anthracis",
    "Bacillus cereus",
    "Pseudomonas aeruginosa",
    "Klebsiella pneumoniae",
    "Klebsiella oxytoca",
    "Acinetobacter baumannii",
    "Enterobacter cloacae",
    "Serratia marcescens",
    "Proteus mirabilis",
    "Campylobacter jejuni",
    "Campylobacter coli",
    "Helicobacter pylori",
    "Neisseria meningitidis",
    "Neisseria gonorrhoeae",
    "Haemophilus influenzae",
    "Bordetella pertussis",
    "Corynebacterium diphtheriae",
    "Mycobacterium tuberculosis",
    "Mycobacterium bovis",
    "Mycobacterium leprae",
    "Brucella abortus",
    "Brucella melitensis",
    "Francisella tularensis",
    "Legionella pneumophila",
    "Leptospira interrogans",
    "Treponema pallidum",
    "Borrelia burgdorferi",
    "Chlamydia trachomatis",
    "Chlamydia pneumoniae",
    "Rickettsia rickettsii",
    "Rickettsia prowazekii",
    "Bartonella henselae",
    "Pasteurella multocida",
    "Moraxella catarrhalis",
    "Bacteroides fragilis",
    "Fusobacterium nucleatum",
    "Gardnerella vaginalis",
    "Providencia stuartii",
    "Morganella morganii",
    "Aeromonas hydrophila",
    "Edwardsiella tarda",
    "Cronobacter sakazakii",
    "Citrobacter freundii",
    "Burkholderia pseudomallei",
    "Burkholderia cepacia"
]

if not DATASETS_EXE.exists():
    raise FileNotFoundError(f"datasets.exe not found at: {DATASETS_EXE}")

existing_accessions = set()

if CLEAN_TRAINING_FILE.exists():
    clean_df = pd.read_csv(CLEAN_TRAINING_FILE)
    existing_accessions.update(clean_df["accession"].dropna().astype(str).tolist())

if UNKNOWN_FILE.exists():
    unknown_df = pd.read_csv(UNKNOWN_FILE)
    existing_accessions.update(unknown_df["accession"].dropna().astype(str).tolist())

records = []

for i, taxon in enumerate(pathogenic_taxa, start=1):
    print(f"[{i}/{len(pathogenic_taxa)}] Searching: {taxon}")

    cmd = [
        str(DATASETS_EXE),
        "summary", "genome", "taxon", taxon,
        "--assembly-level", "complete,chromosome",
        "--assembly-source", "RefSeq",
        "--exclude-atypical",
        "--limit", str(PER_TAXON),
        "--report", "ids_only"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    text = result.stdout + "\n" + result.stderr

    accessions = re.findall(r"GC[AF]_\d+\.\d+", text)
    accessions = list(dict.fromkeys(accessions))

    if not accessions:
        print("  No accessions found.")
        continue

    added = 0

    for accession in accessions:
        if accession in existing_accessions:
            continue

        records.append({
            "accession": accession,
            "label": 0,
            "source_taxon": taxon,
            "source": "extra_pathogenic_ncbi"
        })

        existing_accessions.add(accession)
        added += 1

    print(f"  Found: {len(accessions)} | Added: {added}")

    time.sleep(0.3)

extra_df = pd.DataFrame(records)

if not extra_df.empty:
    extra_df = extra_df.drop_duplicates(subset=["accession"])

extra_df.to_csv(OUTPUT_FILE, index=False)

print("\nDone.")
print("Extra negative genomes collected:", len(extra_df))
print("Saved to:", OUTPUT_FILE)

if not extra_df.empty:
    print("\nFirst few rows:")
    print(extra_df.head())