# couple-household-finances

Predicts whether an expense is a **group (couple)** or **solo** expense.

`source/shared_model.py` trains a ridge (L2) logistic regression on each purchase's merchant, category, detailed category, payment channel and amount, and compares it with and without merchant-name embeddings (`all-MiniLM-L6-v2` from sentence-transformers, which runs locally).

## Project structure

```
data/raw/Transactions.xlsx   labeled transactions used for training
source/
  shared_model.py            load, split, train and compare the models
  export_model.py            export the trained model for the frontend
models/shared_model.json     exported model
notebooks/                   notebook version of the model
frontend/index.html          "Shared: budgeting for two" app (open in a browser)
requirements.txt
```

## Frontend

Open `frontend/index.html` in a browser. It starts with built-in sample data; upload transactions and a model JSON in **Settings → Data**. Everything runs in the browser, and files stay on the device.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Put the labeled spreadsheet at `data/Transactions.xlsx` (or `data/raw/Transactions.xlsx`), with a sheet named `data_pd` and these columns:
`Merchant`, `Amount (USD)`, `Detailed category`, `Payment channel`, `Transaction ID`, and the label **`Solo / Group`** (`Solo` or `Group`).

## Run

```bash
python source/shared_model.py
```

It evaluates two ways:

- **Split A (random):** a random 80/20 split, so most test stores were also seen in training.
- **Split B (new merchants, 5 folds):** every test store is absent from training, which is where embeddings should help most.

For each, it prints ROC-AUC and PR-AUC with and without embeddings. In our runs, embeddings made no meaningful difference (changes of about ±0.01, smaller than the spread between folds).

## Data privacy

Transaction data is personal. `.gitignore` excludes `data/` and all `.xlsx`, `.xls` and `.csv` files, so spreadsheets stay on your own computer. Never commit real transaction data.
