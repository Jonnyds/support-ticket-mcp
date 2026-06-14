"""Download the Tobi-Bueck customer-support-tickets CSV to data/tickets.csv.
Override the source with DATASET_URL. Usage: python download_data.py"""

import os
from pathlib import Path

import pandas as pd

DATASET_URL = os.getenv(
    "DATASET_URL",
    "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/"
    "resolve/main/aa_dataset-tickets-multi-lang-5-2-50-version.csv",
)

OUT = Path(__file__).resolve().parent / "data" / "tickets.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from:\n  {DATASET_URL}")
    try:
        df = pd.read_csv(DATASET_URL)
    except Exception as exc:
        raise SystemExit(
            f"Download failed: {exc}\n"
            f"Check your internet connection, or set DATASET_URL to a valid CSV."
        )
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows and {len(df.columns)} columns to {OUT}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
