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
frontend/model.js            built-in model for the app (written by export_model.py)
requirements.txt
```

## Frontend

Open `frontend/index.html` in a browser (double-clicking works). It starts with built-in sample data; upload transactions in **Settings → Data**. Everything runs in the browser, and files stay on the device.

The model is built in: `frontend/model.js` holds the trained weights and the page loads it automatically. To update it, retrain and re-export, then refresh the page:

```bash
python source/export_model.py   # writes models/shared_model.json and frontend/model.js
```

`model.js` uses the same weights as `shared_model.json`, with one adjustment: the intercept is lowered by ln(#solo / #shared) to undo the `class_weight="balanced"` training, so the page's "chance it's shared" matches how often purchases really are shared (the page marks purchases shared automatically above 70%).

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
