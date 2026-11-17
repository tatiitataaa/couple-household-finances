"""
Split the labeled Excel workbook into separate train/test CSV files.

Usage:
    python source/split_data.py

Reads  data/raw/gabriela_transactions.xlsx  (sheets "training" and "test")
Writes data/processed/train.csv and data/processed/test.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "gabriela_transactions.xlsx"
OUT = ROOT / "data" / "processed"
SHEETS = {"training": "train.csv", "test": "test.csv"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sheet, fname in SHEETS.items():
        df = pd.read_excel(SRC, sheet_name=sheet)
        # Drop helper columns without a header (e.g. "Unnamed: 10").
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        # Dates come in mixed formats (e.g. "2024-11-11" and "10/1/26"), so parse each one on its own.
        df["Date"] = pd.to_datetime(df["Date"], format="mixed").dt.date
        df.to_csv(OUT / fname, index=False)
        counts = df["Solo / Group"].value_counts().to_dict()
        print(f"{sheet!r:16} -> {OUT.relative_to(ROOT) / fname}  ({len(df)} rows, {counts})")


if __name__ == "__main__":
    main()
