"""Shared-or-solo model

GOAL
    Predict whether a purchase is SHARED (a couple expense, "Group") or SOLO.

INPUT
    One row per labeled purchase: merchant, amount, category, detailed category,
    payment channel, and the label "Solo / Group".

HOW IT WORKS
    Part 1. Load and clean the purchases, then split them two ways:
            - Split A (random): most test stores also appear in training.
            - Split B (new merchants): no test store appears in training.
              This is the harder case: will it work for a store it has never seen?
    Part 2. Turn each merchant name into numbers ("embeddings") with a small
            pretrained language model, so that e.g. "Trader Joe's" ends up close
            to "grocery store" even if it never appeared in training.
    Part 3. Train two ridge logistic regressions and compare them:
            - without embeddings (merchant, categories, channel, amount)
            - with embeddings (the same features plus the merchant-name embeddings)

READING THE RESULTS
    ROC-AUC: 0.5 is random guessing, 1.0 is perfect.
    PR-AUC:  how well it finds shared purchases. Random guessing scores the
             share of shared purchases, so compare against that number.
    "change": embeddings minus no embeddings. Positive means embeddings helped.

Run:  python source/shared_model.py   (reads "data/Gabriela Transactions.xlsx" or
      "data/raw/Gabriela Transactions.xlsx", sheet "data_pd")
Needs: pip install -r requirements.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =====================================================================
# Part 1: load, clean, split
# =====================================================================
ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [ROOT / "data" / "Gabriela Transactions.xlsx", ROOT / "data" / "raw" / "Gabriela Transactions.xlsx"]
PATH = next((p for p in CANDIDATES if p.exists()), None)
if PATH is None:
    raise FileNotFoundError(f"Spreadsheet not found. Looked in: {[str(p) for p in CANDIDATES]}")
SHEET = "data_pd"
SEED = 0

# 1. Load and rename columns
df = pd.read_excel(PATH, sheet_name=SHEET).rename(columns={
    "Transaction ID": "tid",
    "Merchant": "merchant",
    "Amount (USD)": "amount",
    "Detailed category": "det",
    "Payment channel": "channel",
    "Solo / Group": "label",
})

# 2. Keep labeled purchases only; drop refunds (negative amounts)
df = df[df["label"].isin(["Solo", "Group"]) & (df["amount"] > 0)].copy()

# Repeated identical purchases (transit fares, subscriptions) are real, so no deduping.

# 3. Clean text fields and rebuild the broad category from the detailed one
PRIMARIES = [
    "FOOD_AND_DRINK", "GENERAL_MERCHANDISE", "GENERAL_SERVICES", "HOME_IMPROVEMENT",
    "RENT_AND_UTILITIES", "TRANSPORTATION", "TRAVEL", "ENTERTAINMENT", "MEDICAL",
    "PERSONAL_CARE", "GOVERNMENT_AND_NON_PROFIT", "LOAN_PAYMENTS", "BANK_FEES",
    "TRANSFER_IN", "TRANSFER_OUT", "INCOME",
]
df["merchant"] = df["merchant"].astype(str).str.strip()
df["det"] = df["det"].fillna("UNKNOWN")
df["channel"] = df["channel"].fillna("unknown")
df["cat"] = df["det"].map(lambda d: next((p for p in PRIMARIES if d.startswith(p)), "OTHER"))

# 4. Target and amount feature
df["y"] = (df["label"] == "Group").astype(int)
df["log_amount"] = np.log1p(df["amount"])

FEATURES = ["merchant", "cat", "det", "channel", "log_amount"]

# 5a. Split A: random 80/20, same share of shared purchases on both sides
trainA, testA = train_test_split(df, test_size=0.2, stratify=df["y"], random_state=SEED)

# 5b. Split B: merchant-held-out 80/20, also balanced on the shared share
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, test_idx = next(sgkf.split(df, df["y"], groups=df["merchant"]))
trainB, testB = df.iloc[train_idx], df.iloc[test_idx]

# 6. Sanity checks
def describe(name, train, test):
    seen = test["merchant"].isin(train["merchant"]).mean()
    print(f"{name}: train {len(train)}, test {len(test)}, "
          f"shared in test {test['y'].mean():.1%}, "
          f"test merchants seen in training {seen:.0%}")

print(f"Labeled purchases: {len(df)}, shared: {df['y'].mean():.1%}, merchants: {df['merchant'].nunique()}")
describe("Split A (random)", trainA, testA)
describe("Split B (merchant-held-out)", trainB, testB)
assert not set(testB["merchant"]) & set(trainB["merchant"]), "Merchant leaked across split B"

# =====================================================================
# Part 2: merchant embeddings and ridge models
# =====================================================================
MODEL_NAME = "all-MiniLM-L6-v2"           # small pretrained encoder, 384 numbers per name
PCA_SIZES = [4, 8, 16, 32]                # hyperparameter: embedding dimensions kept
C_VALUES = np.logspace(-3, 2, 10)         # hyperparameter: inverse penalty strength

STORE_TYPES = {
    "st_grocery": "grocery store or supermarket",
    "st_restaurant": "restaurant",
    "st_coffee": "coffee shop or cafe",
    "st_fastfood": "fast food",
    "st_delivery": "food delivery app",
    "st_bar": "bar or nightlife",
    "st_home": "home goods or furniture store",
    "st_hardware": "hardware store",
    "st_utilities": "electricity, gas or internet provider",
    "st_phone": "mobile phone carrier",
    "st_rent": "rent or housing payment",
    "st_pharmacy": "pharmacy or drugstore",
    "st_clothing": "clothing store",
    "st_beauty": "beauty or personal care",
    "st_gym": "gym or fitness class",
    "st_rideshare": "rideshare or taxi",
    "st_transit": "public transit fare",
    "st_travel": "airline, hotel or travel booking",
    "st_streaming": "streaming or entertainment subscription",
    "st_marketplace": "online marketplace",
}
STORE_COLS = list(STORE_TYPES)
CAT_COLS = ["merchant", "cat", "det", "channel"]

# 1. Embed every merchant name once (normalized, so a dot product is cosine similarity).
#    Nothing here is learned from the labels, so it is safe to compute for all rows up front.
encoder = SentenceTransformer(MODEL_NAME)
merchants = sorted(df["merchant"].unique())
merchant_vecs = encoder.encode(merchants, normalize_embeddings=True, show_progress_bar=True)
type_vecs = encoder.encode(list(STORE_TYPES.values()), normalize_embeddings=True)
EMB_COLS = [f"emb_{i}" for i in range(merchant_vecs.shape[1])]

emb_table = pd.DataFrame(merchant_vecs, columns=EMB_COLS)
emb_table[STORE_COLS] = merchant_vecs @ type_vecs.T      # similarity to each store type
emb_table["merchant"] = merchants


def with_embeddings(part):
    """Attach each row's merchant embedding and store-type scores."""
    return part.merge(emb_table, on="merchant", how="left")


# 2. Two model pipelines. PCA sits inside the second one, so during cross-validation
#    it is refit on each training fold and never sees the validation rows.
def make_search(use_embeddings, train):
    steps = [("cats", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)]
    grid = {"clf__C": C_VALUES}
    if use_embeddings:
        steps += [
            ("num", StandardScaler(), ["log_amount"] + STORE_COLS),
            ("emb", Pipeline([("pca", PCA(random_state=SEED)), ("scale", StandardScaler())]), EMB_COLS),
        ]
        grid["pre__emb__pca__n_components"] = PCA_SIZES
    else:
        steps += [("num", StandardScaler(), ["log_amount"])]

    pipe = Pipeline([
        ("pre", ColumnTransformer(steps)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=5000)),
    ])
    # Folds grouped by merchant: hyperparameters are tuned for stores the model has not seen.
    folds = list(GroupKFold(n_splits=5).split(train, train["y"], groups=train["merchant"]))
    return GridSearchCV(pipe, grid, scoring="average_precision", cv=folds, n_jobs=-1)


# 3. Tune on the training data, then score held-out data once.
#    Split A: one random test set.
#    Split B: all five merchant-held-out folds, because one fold alone can be lucky.
def evaluate(train, test):
    """Tune both models on `train`, score them on `test`."""
    tr, te = with_embeddings(train), with_embeddings(test)
    out = {}
    for use_emb in (False, True):
        search = make_search(use_emb, tr).fit(tr, tr["y"])
        p = search.predict_proba(te)[:, 1]
        out[use_emb] = {
            "roc": roc_auc_score(te["y"], p),
            "pr": average_precision_score(te["y"], p),
            "params": search.best_params_,
        }
    return out


runs = {"A (random)": [evaluate(trainA, testA)], "B (new merchants, 5 folds)": []}
for k, (tr_idx, te_idx) in enumerate(sgkf.split(df, df["y"], groups=df["merchant"])):
    print(f"Split B, fold {k + 1} of 5...")
    runs["B (new merchants, 5 folds)"].append(evaluate(df.iloc[tr_idx], df.iloc[te_idx]))


def summarize(values):
    values = np.array(values)
    return f"{values.mean():.3f}" if len(values) == 1 else f"{values.mean():.3f} \u00b1 {values.std():.3f}"


rows = []
for split_name, split_runs in runs.items():
    row = {"split": split_name}
    for metric, label in [("roc", "ROC-AUC"), ("pr", "PR-AUC")]:
        base = [r[False][metric] for r in split_runs]
        emb = [r[True][metric] for r in split_runs]
        row[f"{label} no embeddings"] = summarize(base)
        row[f"{label} + embeddings"] = summarize(emb)
        row[f"{label} change"] = f"{np.mean(emb) - np.mean(base):+.3f}"
    rows.append(row)

summary = pd.DataFrame(rows).set_index("split").T
print("\n=== Final comparison: embeddings vs. no embeddings ===")
print(f"Random guessing: ROC-AUC 0.500, PR-AUC {df['y'].mean():.3f} (share of shared purchases)")
print("Split B shows the average \u00b1 spread across 5 folds.\n")
print(summary.to_string())

chosen = [r[True]["params"].get("pre__emb__pca__n_components") for rs in runs.values() for r in rs]
print(f"\nPCA sizes chosen by cross-validation: {chosen}")
