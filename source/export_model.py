"""Export the chosen shared-or-solo model to JSON for the frontend.

WHICH MODEL
    The baseline from shared_model.py: ridge logistic regression WITHOUT merchant
    embeddings. It scored the same as the embedding version on every test, and it is
    simpler, so the frontend needs no language model.

WHAT IT DOES
    1. Loads and cleans the labeled purchases exactly as shared_model.py does.
    2. Picks the penalty C by cross-validation grouped by merchant.
    3. Retrains on ALL labeled purchases with that C.
    4. Writes models/shared_model.json: the known values per field, the amount
       scaling, the weights, the intercept and the threshold.
    5. Checks that scoring from the JSON matches scikit-learn.

Run:  python source/export_model.py
Note: the JSON contains merchant names from your transactions, so treat it as
      personal data. models/ is gitignored.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [ROOT / "data" / "Gabriela Transactions.xlsx", ROOT / "data" / "raw" / "Gabriela Transactions.xlsx"]
PATH = next((p for p in CANDIDATES if p.exists()), None)
if PATH is None:
    raise FileNotFoundError(f"Spreadsheet not found. Looked in: {[str(p) for p in CANDIDATES]}")
OUT = ROOT / "models" / "shared_model.json"
SHEET = "data_pd"
C_VALUES = np.logspace(-3, 2, 10)

PRIMARIES = [
    "FOOD_AND_DRINK", "GENERAL_MERCHANDISE", "GENERAL_SERVICES", "HOME_IMPROVEMENT",
    "RENT_AND_UTILITIES", "TRANSPORTATION", "TRAVEL", "ENTERTAINMENT", "MEDICAL",
    "PERSONAL_CARE", "GOVERNMENT_AND_NON_PROFIT", "LOAN_PAYMENTS", "BANK_FEES",
    "TRANSFER_IN", "TRANSFER_OUT", "INCOME",
]
CAT_COLS = ["merchant", "cat", "det", "channel"]

# 1. Load and clean (same steps as shared_model.py)
df = pd.read_excel(PATH, sheet_name=SHEET).rename(columns={
    "Merchant": "merchant",
    "Amount (USD)": "amount",
    "Detailed category": "det",
    "Payment channel": "channel",
    "Solo / Group": "label",
})
df = df[df["label"].isin(["Solo", "Group"]) & (df["amount"] > 0)].copy()
df["merchant"] = df["merchant"].astype(str).str.strip()
df["det"] = df["det"].fillna("UNKNOWN")
df["channel"] = df["channel"].fillna("unknown")
df["cat"] = df["det"].map(lambda d: next((p for p in PRIMARIES if d.startswith(p)), "OTHER"))
df["y"] = (df["label"] == "Group").astype(int)
df["log_amount"] = np.log1p(df["amount"])

# 2. Pick C with merchant-grouped cross-validation, then 3. refit on all data
pipe = Pipeline([
    ("pre", ColumnTransformer([
        ("cats", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", StandardScaler(), ["log_amount"]),
    ])),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=5000)),
])
folds = list(GroupKFold(n_splits=5).split(df, df["y"], groups=df["merchant"]))
search = GridSearchCV(pipe, {"clf__C": C_VALUES}, scoring="average_precision", cv=folds, n_jobs=-1)
search.fit(df, df["y"])
model = search.best_estimator_  # GridSearchCV refits on all rows by default

# 4. Pull out everything the frontend needs
pre, clf = model.named_steps["pre"], model.named_steps["clf"]
enc, scaler = pre.named_transformers_["cats"], pre.named_transformers_["num"]
coefs = clf.coef_[0]

weights, start = {}, 0
for col, values in zip(CAT_COLS, enc.categories_):
    weights[col] = {str(v): round(float(w), 6) for v, w in zip(values, coefs[start:start + len(values)])}
    start += len(values)
amount_weight = float(coefs[start])

export = {
    "name": "shared-or-solo",
    "version": date.today().isoformat(),
    "model": "logistic regression (L2), no embeddings",
    "trained_on": {
        "purchases": int(len(df)),
        "merchants": int(df["merchant"].nunique()),
        "shared_share": round(float(df["y"].mean()), 4),
    },
    "C": float(search.best_params_["clf__C"]),
    "cv_average_precision": round(float(search.best_score_), 4),
    "how_to_score": [
        "Skip refunds: only score purchases with amount > 0.",
        "merchant: trim spaces; must match exactly (case-sensitive) to get a weight.",
        "det: the detailed category; use 'UNKNOWN' when missing.",
        "channel: the payment channel; use 'unknown' when missing.",
        "cat: the first entry of primary_categories that det starts with, else 'OTHER'.",
        "x = (ln(1 + amount) - amount_scaling.mean) / amount_scaling.std",
        "z = intercept + weights.merchant[merchant] + weights.cat[cat] + weights.det[det] "
        "+ weights.channel[channel] + amount_weight * x   (a value not in the list adds 0)",
        "p_shared = 1 / (1 + exp(-z)); label = 'Group' if p_shared >= threshold else 'Solo'",
    ],
    "threshold": 0.5,
    "primary_categories": PRIMARIES,
    "intercept": round(float(clf.intercept_[0]), 6),
    "amount_scaling": {"mean": float(scaler.mean_[0]), "std": float(scaler.scale_[0])},
    "amount_weight": round(amount_weight, 6),
    "weights": weights,
}

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")


# 5. Check: scoring from the JSON must match scikit-learn
def score(row, m):
    cat = next((p for p in m["primary_categories"] if row["det"].startswith(p)), "OTHER")
    x = (np.log1p(row["amount"]) - m["amount_scaling"]["mean"]) / m["amount_scaling"]["std"]
    z = (m["intercept"] + m["amount_weight"] * x
         + m["weights"]["merchant"].get(row["merchant"], 0) + m["weights"]["cat"].get(cat, 0)
         + m["weights"]["det"].get(row["det"], 0) + m["weights"]["channel"].get(row["channel"], 0))
    return 1 / (1 + np.exp(-z))


reloaded = json.loads(OUT.read_text(encoding="utf-8"))
from_json = df.apply(lambda r: score(r, reloaded), axis=1).to_numpy()
from_sklearn = model.predict_proba(df)[:, 1]
gap = np.abs(from_json - from_sklearn).max()
assert gap < 1e-4, f"JSON scoring differs from scikit-learn by {gap}"

print(f"Chosen C: {export['C']:.3f} (cross-validated PR-AUC {export['cv_average_precision']:.3f})")
print(f"Known values: " + ", ".join(f"{k} {len(v)}" for k, v in weights.items()))
print(f"JSON scoring matches scikit-learn (max difference {gap:.1e})")
print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")
