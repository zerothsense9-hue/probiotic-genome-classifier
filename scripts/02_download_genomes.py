import pandas as pd
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

ACCESSION_FILE = BASE_DIR / "data" / "training_accessions_clean.csv"
DOWNLOAD_DIR = BASE_DIR / "genomes" / "zip_files"
LOG_FILE = BASE_DIR / "data" / "download_log.csv"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ACCESSION_FILE)
accessions = df["accession"].dropna().unique().tolist()

records = []

for i, accession in enumerate(accessions, start=1):
    output_zip = DOWNLOAD_DIR / f"{accession}.zip"

    if output_zip.exists():
        print(f"[{i}/{len(accessions)}] Already exists: {accession}")
        records.append({
            "accession": accession,
            "status": "already_downloaded",
            "zip_file": str(output_zip)
        })
        continue

    print(f"[{i}/{len(accessions)}] Downloading: {accession}")

    cmd = [
        str(BASE_DIR / "datasets.exe"),
        "download", "genome", "accession", accession,
        "--include", "genome",
        "--filename", str(output_zip)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and output_zip.exists():
            status = "downloaded"
        else:
            status = "failed"

        records.append({
            "accession": accession,
            "status": status,
            "zip_file": str(output_zip),
            "error": result.stderr[:500]
        })

    except Exception as e:
        records.append({
            "accession": accession,
            "status": "error",
            "zip_file": str(output_zip),
            "error": str(e)
        })

pd.DataFrame(records).to_csv(LOG_FILE, index=False)

print("\nDownload process completed.")
print("Log saved to:", LOG_FILE)