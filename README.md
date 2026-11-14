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

## Data privacy

Real expense data must stay in `data/raw/` and is excluded from git. Only commit synthetic or fully anonymized examples to `data/sample/`.
