import zipfile
import shutil
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

ZIP_DIR = BASE_DIR / "genomes" / "zip_files"
OUTPUT_DIR = BASE_DIR / "genomes" / "fna_files"
TEMP_DIR = BASE_DIR / "genomes" / "temp_extract"
INVENTORY_FILE = BASE_DIR / "data" / "fasta_inventory.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

records = []

zip_files = list(ZIP_DIR.glob("*.zip"))

print("ZIP files found:", len(zip_files))

for i, zip_path in enumerate(zip_files, start=1):
    accession = zip_path.stem
    print(f"[{i}/{len(zip_files)}] Processing {accession}")

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            fasta_members = [
                m for m in z.namelist()
                if m.endswith("_genomic.fna") or m.endswith(".fna") or m.endswith(".fasta") or m.endswith(".fa")
            ]

            if not fasta_members:
                records.append({
                    "accession": accession,
                    "status": "no_fasta_found",
                    "zip_file": str(zip_path),
                    "fasta_path": ""
                })
                continue

            # Prefer genomic.fna
            selected = None
            for m in fasta_members:
                if m.endswith("_genomic.fna"):
                    selected = m
                    break

            if selected is None:
                selected = fasta_members[0]

            output_file = OUTPUT_DIR / f"{accession}.fna"

            with z.open(selected) as source, open(output_file, "wb") as target:
                shutil.copyfileobj(source, target)

            records.append({
                "accession": accession,
                "status": "extracted",
                "zip_file": str(zip_path),
                "fasta_member": selected,
                "fasta_path": str(output_file),
                "file_size_mb": round(output_file.stat().st_size / (1024 * 1024), 2)
            })

    except Exception as e:
        records.append({
            "accession": accession,
            "status": "error",
            "zip_file": str(zip_path),
            "fasta_path": "",
            "error": str(e)
        })

inventory = pd.DataFrame(records)
inventory.to_csv(INVENTORY_FILE, index=False)

print("\nDone.")
print("Extracted FASTA files:", (inventory["status"] == "extracted").sum())
print("No FASTA found:", (inventory["status"] == "no_fasta_found").sum())
print("Errors:", (inventory["status"] == "error").sum())
print("Output folder:", OUTPUT_DIR)
print("Inventory saved:", INVENTORY_FILE)