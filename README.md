# couple-household-finances

An edge AI project that predicts whether an expense is a **group (couple)** or **solo** expense.

The model is trained on transaction data (amount, merchant, category, date/time, etc.), then exported to a lightweight format (ONNX) so it can run on-device, keeping personal financial data local.

## Project structure

```
data/
  raw/            labeled transactions (gabriela_transactions.xlsx)
  processed/      train/test CSVs + cached embeddings (gitignored)
source/
  split_data.py   Excel -> data/processed/train.csv + test.csv
  train.py        logistic regression -> models/model.json
  train_embed.py  same + text-embedding-3-small merchant embeddings -> models/model_embed.json
  features.py     feature engineering (placeholder)
  evaluate.py     metrics and evaluation (placeholder)
  export.py       export trained model for edge (placeholder)
models/           trained / exported models (gitignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

`data/raw/gabriela_transactions.xlsx` holds the labeled transactions. The two sheets used for modeling are:

| Sheet | Rows | Dates | Purpose |
|---|---|---|---|
| `training` | 3,554 | 2024-11-11 to 2026-11-15 | model training |
| `test` | 1,327 | 2025-04-12 to 2026-11-14 | held-out evaluation |

Columns: `Date`, `Merchant`, `Amount (USD)`, `Type`, `Category`, `Detailed category`, `Pending`, `Payment channel`, `Transaction ID`, and the label **`Solo / Group`**.

The workbook also has `data_pd` and `Upload Data` sheets, which the scripts don't use. Dates in `training` mix `YYYY-MM-DD` and `M/D/YY` formats.

Load it with:

```python
import pandas as pd
train = pd.read_excel("data/raw/gabriela_transactions.xlsx", sheet_name="training")
test  = pd.read_excel("data/raw/gabriela_transactions.xlsx", sheet_name="test")
train["Date"] = pd.to_datetime(train["Date"], format="mixed")
```

## Train the model

```bash
python source/split_data.py     # Excel -> data/processed/train.csv + test.csv
python source/train.py --data data/processed/train.csv --test data/processed/test.csv --out models/model.json
```

`model.json` is a logistic regression (merchant, category, detailed category, log amount) that the budgeting page loads in its Settings.

To compare against a version that also uses OpenAI `text-embedding-3-small` embeddings of the merchant name:

```bash
# put OPENAI_API_KEY=sk-... in .env (gitignored), then:
python source/train_embed.py --data data/processed/train.csv --test data/processed/test.csv --out models/model_embed.json
```

This prints baseline vs. embedding metrics on the same test set and writes `model_embed.json`, which has the same fields as `model.json` plus an `embedding` block (`weights` is dotted with the merchant's embedding and added to the score). Embeddings are cached in `data/processed/`, so each merchant name is sent to the API only once. Only merchant names are sent to OpenAI, not amounts or dates.

## Data privacy

This repository must stay **private**: it contains real personal financial data. Don't commit any other real data.
