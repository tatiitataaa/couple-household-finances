"""
P(group | transaction) con regresion logistica L2 (Ridge).

Uso:  python ridge_pipeline.py --data labeled_transactions.csv

Columnas esperadas (como en tu muestra): Date, Merchant, Amount (USD), Type,
Category, Transaction ID, Solo / Group. "Detailed Category" es opcional.
"""
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RS = 42
EXCLUDED = {"TRANSFER_IN", "TRANSFER_OUT", "INCOME", "LOAN_PAYMENTS"}
warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------------------------------------------------ 1. carga + EDA
def find_col(cols, *tests):
    for t in tests:
        for c in cols:
            if t(c.lower().strip()):
                return c
    return None


def load(path):
    raw = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    cols = list(raw.columns)
    c = dict(
        date=find_col(cols, lambda h: h == "date", lambda h: "date" in h),
        merchant=find_col(cols, lambda h: "merchant" in h, lambda h: h == "name"),
        amount=find_col(cols, lambda h: h.startswith("amount"), lambda h: "amount" in h),
        type=find_col(cols, lambda h: h == "type"),
        cat=find_col(cols, lambda h: h == "category"),
        det=find_col(cols, lambda h: "detailed" in h),
        label=find_col(cols, lambda h: "solo" in h, lambda h: h == "label"),
    )
    df = pd.DataFrame({
        "date": pd.to_datetime(raw[c["date"]], errors="coerce"),
        "merchant": raw[c["merchant"]].astype(str).str.strip(),
        "amount": pd.to_numeric(raw[c["amount"]].astype(str).str.replace(r"[$,\s]", "", regex=True),
                                errors="coerce"),
        "cat": raw[c["cat"]].astype(str).str.strip() if c["cat"] else np.nan,
        "det": raw[c["det"]].astype(str).str.strip() if c["det"] else np.nan,
        "type": raw[c["type"]].astype(str).str.strip() if c["type"] else "Charge",
        "y": raw[c["label"]].astype(str).str.strip().str.lower().map({"group": 1, "shared": 1, "solo": 0}),
    })
    # Transaction ID se descarta a proposito: es un identificador, no una senal (riesgo de fuga/ruido).
    return df


def eda(df):
    print("== 1. EDA ==")
    print(f"filas: {len(df)} | y=group: {df.y.mean():.1%} (nulos en y: {df.y.isna().sum()})")
    print("nulos por columna:\n", df.isna().sum().to_string())
    print("tipos:", df.type.value_counts().to_dict())
    print("Fuga: 'Transaction ID' excluido. 'Merchant' es valido (se conoce al cobrar), pero se "
          "agrupan comercios raros dentro del Pipeline para que la CV no los vea de antemano.\n")


# ------------------------------------------------------------------ 2. features
def add_features(df):
    d = df.copy()
    d["log_amount"] = np.log1p(d.amount.clip(lower=0))
    d["is_weekend"] = (d.date.dt.dayofweek >= 5).astype(float)
    d["is_round_amount"] = (np.isclose(d.amount % 1, 0)).astype(float)       # cuotas/suscripciones/cenas divididas
    dom = d.date.dt.day
    d["near_month_boundary"] = ((dom <= 3) | (dom >= 28)).astype(float)       # renta/servicios compartidos
    return d


NUM = ["log_amount", "is_weekend", "is_round_amount", "near_month_boundary"]


# ------------------------------------------------------------------ 3. pipeline
def make_pipeline(cat_cols, C=1.0, balanced=False):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=3,
                                               sparse_output=True)),
                          # with_mean=False mantiene la matriz dispersa; la escala si se normaliza
                          ("sc", StandardScaler(with_mean=False))]), cat_cols),
    ])
    clf = LogisticRegression(C=C,  # L2 es el default (penalty="l2" esta deprecado en sklearn>=1.8)
                              solver="lbfgs", max_iter=5000, random_state=RS,
                             class_weight="balanced" if balanced else None)
    return Pipeline([("pre", pre), ("clf", clf)])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    df = load(args.data)
    eda(df)
    df = df[df.type.str.lower().isin(["charge", "purchase", "sale"])]  # SUPUESTO: excluye refunds/pagos
    df = df[~df.cat.isin(EXCLUDED)].dropna(subset=["y", "amount", "date"])
    df["y"] = df.y.astype(int)
    df = add_features(df)

    cat_cols = ["merchant", "cat"] + (["det"] if df.det.notna().any() else [])
    X, y = df[NUM + cat_cols], df.y

    minority = min(y.mean(), 1 - y.mean())
    balanced = minority < 0.30
    print(f"clase minoritaria: {minority:.1%} -> class_weight={'balanced' if balanced else 'None'}")
    # SUPUESTO: split aleatorio estratificado. Si hay deriva temporal, usa split por fecha.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)

    # ---- 4. entrenamiento
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    grid = GridSearchCV(make_pipeline(cat_cols, balanced=balanced),
                        {"clf__C": np.logspace(-4, 2, 13)}, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(Xtr, ytr)
    C_opt = grid.best_params_["clf__C"]
    print(f"\n== 4. C optimo = {C_opt:.4g} (CV ROC-AUC {grid.best_score_:.3f}) ==")

    best = make_pipeline(cat_cols, C_opt, balanced).fit(Xtr, ytr)

    # class_weight='balanced' sesga las probabilidades hacia arriba -> recalibrar (sigmoide, 5 folds).
    final = (CalibratedClassifierCV(make_pipeline(cat_cols, C_opt, balanced), method="sigmoid", cv=cv)
             .fit(Xtr, ytr) if balanced else best)

    # ---- 6. umbral (SUPUESTO: maximizar F1 de 'group', elegido con predicciones OUT-OF-FOLD del train)
    oof = cross_val_predict(make_pipeline(cat_cols, C_opt, balanced) if not balanced else
                            CalibratedClassifierCV(make_pipeline(cat_cols, C_opt, balanced),
                                                   method="sigmoid", cv=3),
                            Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    pr, rc, th = precision_recall_curve(ytr, oof)
    f1s = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    thr = float(th[np.argmax(f1s)])

    # ---- 5. evaluacion hold-out
    p = final.predict_proba(Xte)[:, 1]
    pred = (p >= thr).astype(int)
    print("\n== 5. Metricas hold-out ==")
    print("| metrica | valor |\n|---|---|")
    for k, v in [("ROC-AUC", roc_auc_score(yte, p)), ("PR-AUC", average_precision_score(yte, p)),
                 ("Log-loss", log_loss(yte, p)), ("Brier", brier_score_loss(yte, p)),
                 (f"F1 (umbral {thr:.2f})", f1_score(yte, pred))]:
        print(f"| {k} | {v:.3f} |")
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    print(f"\nMatriz de confusion @ {thr:.2f}: TN={tn} FP={fp} FN={fn} TP={tp}")

    fr, mp = calibration_curve(yte, p, n_bins=8, strategy="quantile")
    plt.figure(figsize=(4.5, 4.5))
    plt.plot([0, 1], [0, 1], "--", c="grey")
    plt.plot(mp, fr, "o-")
    plt.xlabel("P predicha"); plt.ylabel("Fraccion real group"); plt.title("Calibracion (hold-out)")
    plt.tight_layout(); plt.savefig("calibration.png", dpi=150)
    print("Curva guardada en calibration.png")

    # ---- 7. interpretacion
    names = best.named_steps["pre"].get_feature_names_out()
    coef = best.named_steps["clf"].coef_[0]
    tab = (pd.DataFrame({"variable": names, "coef": coef, "odds_ratio": np.exp(coef)})
           .assign(abs=lambda t: t.coef.abs()).sort_values("abs", ascending=False).drop(columns="abs"))
    print("\n== 7. Coeficientes estandarizados (OR = por +1 desviacion estandar) ==")
    print("| variable | coef | odds ratio |\n|---|---|---|")
    for _, r in tab.head(15).iterrows():
        print(f"| {r.variable} | {r.coef:+.3f} | {r.odds_ratio:.2f} |")

    # ---- comparacion de regularizacion: C optimo vs C/10
    print("\n== Comparacion: C optimo vs C/10 ==")
    m2 = make_pipeline(cat_cols, C_opt / 10, balanced).fit(Xtr, ytr)
    c2 = m2.named_steps["clf"].coef_[0]
    print(f"AUC test  C={C_opt:.3g}: {roc_auc_score(yte, best.predict_proba(Xte)[:, 1]):.3f} | "
          f"C={C_opt/10:.3g}: {roc_auc_score(yte, m2.predict_proba(Xte)[:, 1]):.3f}")
    print(f"||coef||2 : {np.linalg.norm(coef):.2f} -> {np.linalg.norm(c2):.2f} | "
          f"coef no nulos (|w|>0.01): {(abs(coef)>.01).sum()} -> {(abs(c2)>.01).sum()}")
    top = tab.head(5).variable.tolist()
    cmp = pd.DataFrame({"variable": names, "coef_Copt": coef, "coef_C/10": c2}).set_index("variable").loc[top]
    print(cmp.round(3).to_string())


if __name__ == "__main__":
    main()
