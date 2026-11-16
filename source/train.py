"""
Train a solo-vs-shared classifier and export model.json for the budgeting page.

Usage:
    python source/train.py --data data/processed/train.csv --test data/processed/test.csv --out models/model.json

    (Without --test, 20% of --data is held out for evaluation instead.)

The CSV needs these columns (names are matched loosely, like the page does):
    date, merchant, amount, category, detailed category, label
where label is "solo" or "group"/"shared".

The page scores a purchase as:
    p = sigmoid(intercept + w[merchant=...] + w[det=...] + w[cat=...] + w[log_amount] * z)
    z = (log1p(amount) - mean) / scale
This script fits exactly that form, so the exported weights drop straight in.
"""
import argparse
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

EXCLUDED = {"TRANSFER_IN", "TRANSFER_OUT", "INCOME", "LOAN_PAYMENTS"}  # same as the page


def find_col(cols, *tests):
    for test in tests:
        for c in cols:
            if test(c.lower().strip()):
                return c
    return None


def load(path):
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    cols = list(df.columns)
    c_merch = find_col(cols, lambda h: "merchant" in h, lambda h: h == "name", lambda h: "description" in h)
    c_amt = find_col(cols, lambda h: h.startswith("amount"), lambda h: "amount" in h)
    c_det = find_col(cols, lambda h: "detailed" in h)
    c_cat = find_col(cols, lambda h: h == "category", lambda h: "category" in h and "detailed" not in h)
    c_lab = find_col(cols, lambda h: "solo" in h, lambda h: "shared" in h, lambda h: h == "label")
    missing = [n for n, c in [("merchant", c_merch), ("amount", c_amt), ("label", c_lab)] if c is None]
    if missing:
        raise SystemExit(f"Couldn't find columns: {missing}. Found: {cols}")

    out = pd.DataFrame({
        # IMPORTANT: only strip. The page matches merchant names exactly, so don't lowercase.
        "merchant": df[c_merch].astype(str).str.strip(),
        "amount": pd.to_numeric(df[c_amt].astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce"),
        "det": df[c_det].astype(str).str.strip() if c_det else "",
        "cat": df[c_cat].astype(str).str.strip() if c_cat else "",
        "label": df[c_lab].astype(str).str.strip().str.lower(),
    })
    out.loc[(out.cat == "") & (out.det != ""), "cat"] = out.det.str.split("_").str[:2].str.join("_")
    out["y"] = out.label.map({"group": 1, "shared": 1, "solo": 0})
    out = out.dropna(subset=["amount", "y"])
    out = out[(out.merchant != "") & (~out.cat.isin(EXCLUDED))]
    out["y"] = out.y.astype(int)
    out["log_amount"] = np.log1p(out.amount.clip(lower=0))
    return out.reset_index(drop=True)


def build_vocab(train, min_merchant):
    """Feature names exactly as the page expects them."""
    names = []
    m_counts = train.merchant.value_counts()
    names += [f"merchant={m}" for m in m_counts[m_counts >= min_merchant].index]
    names += [f"det={d}" for d in sorted(train.det[train.det != ""].unique())]
    names += [f"cat={c}" for c in sorted(train.cat[train.cat != ""].unique())]
    return names


def design(df, names, mean, scale):
    idx = {n: i for i, n in enumerate(names)}
    rows, cols = [], []
    for r, (m, d, c) in enumerate(zip(df.merchant, df.det, df.cat)):
        for key in (f"merchant={m}", f"det={d}", f"cat={c}"):
            j = idx.get(key)
            if j is not None:
                rows.append(r)
                cols.append(j)
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(names)))
    z = ((df.log_amount.values - mean) / scale).reshape(-1, 1)
    return sparse.hstack([X, sparse.csr_matrix(z)], format="csr")


def fit(train, min_merchant, C):
    names = build_vocab(train, min_merchant)
    mean, scale = float(train.log_amount.mean()), float(train.log_amount.std() or 1.0)
    X = design(train, names, mean, scale)
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(X, train.y)
    return clf, names, mean, scale


def predict(clf, names, mean, scale, df):
    return clf.predict_proba(design(df, names, mean, scale))[:, 1]


def tune_C(train, min_merchant, grid=(0.1, 0.3, 1, 3, 10)):
    """Pick regularization by cross-validated log loss (probabilities matter, not just accuracy)."""
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    best, best_ll = grid[0], 1e9
    for C in grid:
        losses = []
        for tr, va in skf.split(train, train.y):
            a, b = train.iloc[tr], train.iloc[va]
            clf, names, mean, scale = fit(a, min_merchant, C)
            losses.append(log_loss(b.y, predict(clf, names, mean, scale, b), labels=[0, 1]))
        ll = float(np.mean(losses))
        print(f"  C={C:<5} cv log loss = {ll:.4f}")
        if ll < best_ll:
            best, best_ll = C, ll
    return best


def band_report(y, p, low, high):
    auto_shared, auto_solo = p >= high, p <= low
    review = ~(auto_shared | auto_solo)
    decided = ~review
    acc = ((p >= 0.5) == y)[decided].mean() if decided.any() else float("nan")
    print(f"  band {low:.2f}/{high:.2f}: auto-decided {decided.mean():.0%}, "
          f"asks you about {review.mean():.0%}, accuracy on auto-decided {acc:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="labeled training CSV/XLSX")
    ap.add_argument("--test", help="separate labeled test CSV/XLSX (otherwise 20%% of --data is held out)")
    ap.add_argument("--out", default="model.json")
    ap.add_argument("--min-merchant", type=int, default=2,
                    help="merchants seen fewer times than this fall back to category weights")
    ap.add_argument("--low", type=float, default=0.2)
    ap.add_argument("--high", type=float, default=0.7)
    args = ap.parse_args()

    df = load(args.data)
    print(f"{len(df)} labeled rows, {df.y.mean():.0%} shared, {df.merchant.nunique()} merchants")

    if args.test:
        train, test = df, load(args.test)
        print(f"Using separate test set: {len(test)} rows, {test.y.mean():.0%} shared")
    else:
        train, test = train_test_split(df, test_size=0.2, stratify=df.y, random_state=42)

    print("Tuning regularization...")
    C = tune_C(train, args.min_merchant)
    clf, names, mean, scale = fit(train, args.min_merchant, C)
    p = predict(clf, names, mean, scale, test)

    print(f"\nHold-out results (C={C})")
    print(f"  AUC        {roc_auc_score(test.y, p):.3f}")
    print(f"  log loss   {log_loss(test.y, p):.3f}")
    print(f"  Brier      {brier_score_loss(test.y, p):.3f}")
    print(f"  accuracy   {((p >= 0.5) == test.y).mean():.1%}")
    seen = test.merchant.isin(set(train.merchant))
    if (~seen).any():
        pu = p[~seen.values]
        print(f"  unseen-merchant accuracy {((pu >= 0.5) == test.y[~seen]).mean():.1%} (n={(~seen).sum()})")
    band_report(test.y.values, p, args.low, args.high)

    # Refit for the exported model: on --data only when a separate test set was given
    # (keeps the test set untouched), otherwise on everything.
    clf, names, mean, scale = fit(df, args.min_merchant, C)
    coef = {n: round(float(w), 6) for n, w in zip(names, clf.coef_[0][:-1]) if abs(w) > 1e-6}
    coef["log_amount"] = round(float(clf.coef_[0][-1]), 6)
    model = {
        "version": f"logreg-{date.today().isoformat()}",
        "intercept": round(float(clf.intercept_[0]), 6),
        "coef": coef,
        "scaler": {"log_amount": {"mean": round(mean, 6), "scale": round(scale, 6)}},
        "band": {"low": args.low, "high": args.high},
        "learning": {"lr": 1.0, "lambda": 0.01, "categoryFactor": 0.5, "maxStep": 1.5},
    }
    with open(args.out, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nWrote {args.out} with {len(coef)} weights. Upload it in the page's Settings.")


if __name__ == "__main__":
    main()
