"""Shared-or-solo model: load the labeled transactions, split them, add merchant
embeddings, and compare ridge logistic regression with and without the embeddings.

Run:  python shared_model.py   (with "Gabriela Transactions.xlsx" in the same folder)
Needs: pip install pandas scikit-learn openpyxl sentence-transformers

Split A (random): purchases at stores the model may have seen before.
Split B (merchant-held-out): every test merchant is absent from training,
which is the case where embeddings should help.
"""
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =====================================================================
# Part 1: load, clean, split
# =====================================================================
PATH = "Gabriela Transactions.xlsx"
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
        ("clf", LogisticRegression(penalty="l2", class_weight="balanced", max_iter=5000)),
    ])
    # Folds grouped by merchant: hyperparameters are tuned for stores the model has not seen.
    folds = list(GroupKFold(n_splits=5).split(train, train["y"], groups=train["merchant"]))
    return GridSearchCV(pipe, grid, scoring="average_precision", cv=folds, n_jobs=-1)


# 3. Tune on the training set, then score the test set once.
results = []
for split_name, train, test in [("A (random)", trainA, testA), ("B (new merchants)", trainB, testB)]:
    tr, te = with_embeddings(train), with_embeddings(test)
    for use_emb in (False, True):
        search = make_search(use_emb, tr).fit(tr, tr["y"])
        p = search.predict_proba(te)[:, 1]
        results.append({
            "split": split_name,
            "model": "ridge + embeddings" if use_emb else "ridge",
            "PR-AUC (test)": round(average_precision_score(te["y"], p), 3),
            "PR-AUC (CV)": round(search.best_score_, 3),
            "best C": round(search.best_params_["clf__C"], 4),
            "best PCA size": search.best_params_.get("pre__emb__pca__n_components", "n/a"),
        })

print(f"\nShare of shared purchases in test (PR-AUC of random guessing): {testA['y'].mean():.3f}")
print(pd.DataFrame(results).to_string(index=False))
