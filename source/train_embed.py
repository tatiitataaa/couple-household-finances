"""
Same solo-vs-shared logistic regression as train.py, plus an OpenAI
text-embedding-3-small embedding of the merchant name. Writes a separate
model file (default models/model_embed.json) and prints the baseline and
embedding models' results side by side on the same test set.

Usage:
    export OPENAI_API_KEY=sk-...
    python source/train_embed.py --data data/processed/train.csv --test data/processed/test.csv \
        --out models/model_embed.json

Scoring form (baseline from train.py plus one dot product):
    p = sigmoid(intercept + w[merchant=...] + w[det=...] + w[cat=...] + w[log_amount] * z
                + dot(embedding.weights, embed(merchant)))
embed(merchant) is the raw, unit-length text-embedding-3-small vector of the stripped
merchant name; the exported weights already include the scaling used during training.

Embeddings are cached in data/processed/embeddings_<model>.json, so each merchant
name is only sent to the API once.
"""
import argparse
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from train import band_report, build_vocab, design, load
from train import fit as fit_base
from train import predict as predict_base
from train import tune_C as tune_C_base

EMBED_MODEL = "text-embedding-3-small"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "processed" / f"embeddings_{EMBED_MODEL}.json"


def load_dotenv(path=ROOT / ".env"):
    """Read KEY=value lines from the gitignored .env (doesn't override real env vars)."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


load_dotenv()


# ---------- embeddings ----------

def get_embeddings(texts, batch=500):
    """Return {text: np.array} for every text, calling the API only for uncached ones."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = sorted({t for t in texts if t not in cache})
    if todo:
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY to compute embeddings.")
        client = OpenAI()
        print(f"Embedding {len(todo)} new merchant names with {EMBED_MODEL}...")
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
            for t, d in zip(chunk, resp.data):
                cache[t] = d.embedding
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
    return {t: np.asarray(cache[t], dtype=np.float32) for t in set(texts)}


def emb_matrix(df, emb):
    return np.vstack([emb[m] for m in df.merchant])


# ---------- model: baseline features + scaled embedding block ----------

def design_embed(df, names, mean, scale, emb, emb_scale):
    X = design(df, names, mean, scale)
    E = sparse.csr_matrix(emb_matrix(df, emb) * emb_scale)
    return sparse.hstack([X, E], format="csr")


def fit(train, min_merchant, C, emb, emb_scale):
    names = build_vocab(train, min_merchant)
    mean, scale = float(train.log_amount.mean()), float(train.log_amount.std() or 1.0)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(design_embed(train, names, mean, scale, emb, emb_scale), train.y)
    return clf, names, mean, scale


def predict(model, df, emb, emb_scale):
    clf, names, mean, scale = model
    return clf.predict_proba(design_embed(df, names, mean, scale, emb, emb_scale))[:, 1]


def tune(train, min_merchant, emb, Cs=(0.1, 0.3, 1, 3, 10), scales=(1, 3, 10)):
    """Pick C and the embedding scale by cross-validated log loss."""
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    folds = list(skf.split(train, train.y))
    best, best_ll = None, 1e9
    for s in scales:
        for C in Cs:
            losses = []
            for tr, va in folds:
                a, b = train.iloc[tr], train.iloc[va]
                m = fit(a, min_merchant, C, emb, s)
                losses.append(log_loss(b.y, predict(m, b, emb, s), labels=[0, 1]))
            ll = float(np.mean(losses))
            print(f"  emb_scale={s:<3} C={C:<5} cv log loss = {ll:.4f}")
            if ll < best_ll:
                best, best_ll = (C, s), ll
    return best


# ---------- reporting ----------

def metrics(name, test, p, seen, low, high):
    y = test.y.values
    print(f"\n{name}")
    print(f"  AUC        {roc_auc_score(y, p):.3f}")
    print(f"  log loss   {log_loss(y, p):.3f}")
    print(f"  Brier      {brier_score_loss(y, p):.3f}")
    print(f"  accuracy   {((p >= 0.5) == y).mean():.1%}")
    if (~seen).any():
        print(f"  unseen-merchant accuracy {((p[~seen] >= 0.5) == y[~seen]).mean():.1%} (n={(~seen).sum()})")
    band_report(y, p, low, high)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="labeled training CSV/XLSX")
    ap.add_argument("--test", help="separate labeled test CSV/XLSX (otherwise 20%% of --data is held out)")
    ap.add_argument("--out", default="models/model_embed.json")
    ap.add_argument("--min-merchant", type=int, default=2)
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

    emb = get_embeddings(list(df.merchant) + list(test.merchant))
    seen = test.merchant.isin(set(train.merchant)).values

    print("\nTuning baseline (train.py)...")
    C0 = tune_C_base(train, args.min_merchant)
    p_base = predict_base(*fit_base(train, args.min_merchant, C0), test)

    print("\nTuning embedding model...")
    C, s = tune(train, args.min_merchant, emb)
    p_emb = predict(fit(train, args.min_merchant, C, emb, s), test, emb, s)

    metrics(f"Baseline hold-out (C={C0})", test, p_base, seen, args.low, args.high)
    metrics(f"Embedding hold-out (C={C}, emb_scale={s})", test, p_emb, seen, args.low, args.high)

    # Refit on --data for export (test set stays untouched when --test is given).
    clf, names, mean, scale = fit(df, args.min_merchant, C, emb, s)
    w = clf.coef_[0]
    n = len(names)
    coef = {k: round(float(v), 6) for k, v in zip(names, w[:n]) if abs(v) > 1e-6}
    coef["log_amount"] = round(float(w[n]), 6)
    emb_w = (w[n + 1:] * s).round(6).tolist()  # fold the scale in: score uses the raw unit vector
    model = {
        "version": f"logreg-embed-{date.today().isoformat()}",
        "intercept": round(float(clf.intercept_[0]), 6),
        "coef": coef,
        "scaler": {"log_amount": {"mean": round(mean, 6), "scale": round(scale, 6)}},
        "embedding": {
            "model": EMBED_MODEL,
            "input": "merchant",
            "dim": len(emb_w),
            "weights": emb_w,
        },
        "band": {"low": args.low, "high": args.high},
        "learning": {"lr": 1.0, "lambda": 0.01, "categoryFactor": 0.5, "maxStep": 1.5},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(model, f)
    print(f"\nWrote {args.out} with {len(coef)} weights + {len(emb_w)}-dim embedding weights.")


if __name__ == "__main__":
    main()
