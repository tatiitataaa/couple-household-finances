# couple-household-finances

An edge AI project that predicts whether an expense is a **group (couple)** or **solo** expense.

The model is trained on transaction data (amount, merchant, category, date/time, etc.), then exported to a lightweight format (ONNX) so it can run on-device, keeping personal financial data local.

## Project structure

```
data/
  raw/          original expense data (gitignored, never committed)
  processed/    cleaned / feature-engineered data (gitignored)
  sample/       small synthetic or anonymized data that is safe to commit
notebooks/      exploration and experiments
src/
  features.py   feature engineering
  train.py      model training
  evaluate.py   metrics and evaluation
  export.py     export trained model for edge (ONNX, quantization)
models/         trained / exported models (gitignored)
edge/           on-device inference code
backend/        optional API
frontend/       optional UI
docs/           project documentation
tests/          unit tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

`data/raw/gabriela_transactions.xlsx` holds the labeled transactions, with two sheets:

| Sheet | Rows | Dates | Purpose |
|---|---|---|---|
| `Training Data` | 3,523 | 2024-11-11 to 2026-11-10 | model training |
| `Test` | 1,327 | 2025-04-12 to 2026-11-14 | held-out evaluation |

Columns: `Date`, `Merchant`, `Amount (USD)`, `Type`, `Category`, `Detailed category`, `Pending`, `Payment channel`, `Transaction ID`, and the label **`Solo / Group`**.

Load it with:

```python
import pandas as pd
train = pd.read_excel("data/raw/gabriela_transactions.xlsx", sheet_name="Training Data")
test  = pd.read_excel("data/raw/gabriela_transactions.xlsx", sheet_name="Test")
```

## Train the model

```bash
python src/split_data.py     # Excel -> data/processed/train.csv + test.csv
python src/train.py --data data/processed/train.csv --test data/processed/test.csv --out models/model.json
```

`model.json` is a logistic regression (merchant, category, detailed category, log amount) that the budgeting page loads in its Settings.

## Data privacy

This repository must stay **private**: it contains real personal financial data. Don't commit any other real data. Put synthetic or anonymized examples in `data/sample/`.
