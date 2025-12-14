"""
Data validation and VIF diagnostics for the clean churn dataset.

This script:
  - Loads `train_clean.csv` and `test_clean.csv` from local `data/` or Hugging Face
  - Runs basic data validation checks (types, missingness, column consistency)
  - Computes raw and iteratively-pruned VIF tables on numeric features

Outputs:
  - `validation_outputs/validation_summary.txt`
  - `validation_outputs/vif_raw.csv`
  - `validation_outputs/vif_iterative_history.csv`
  - `validation_outputs/vif_final_kept.csv`
  - `validation_outputs/vif_dropped_features.csv`
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

HF_DATASET_REPO = "Saravanan1999/MDBAttrition"

TARGET = "churned"
META_EXCLUDE = {"client_id", "reference_month", TARGET}
MIN_COVERAGE = 0.70          # keep cols with >= 70% non-null
VIF_PRUNE_THRESHOLD = 10.0   # stop when all VIF <= threshold
MAX_VIF_ITERS = 50           # safety cap

# leakage-check config
RANDOM_STATE = 42
WALK_FORWARD_LAST_K = 2      # number of last months for walk-forward splits
TOPK_MI = 8                  # per-feature audit size for feature swap

OUTDIR = Path("validation_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Data access helpers
# ----------------------------------------------------------------------

def get_data_file(filename: str) -> str:
    """
    Return a local path to the requested data file.

    Preference order:
    1) Local 'data/' directory (if present)
    2) Download from Hugging Face dataset repo.
    """
    local_path = os.path.join("data", filename)
    if os.path.exists(local_path):
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Please install `huggingface_hub` to download data from Hugging Face "
            "(pip install huggingface_hub)."
        ) from e

    return hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=filename,
        repo_type="dataset",
    )


def ensure_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "reference_month" in df.columns:
        df["reference_month"] = pd.to_datetime(df["reference_month"])
        df["_month_period"] = df["reference_month"].dt.to_period("M")
    return df


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

def build_numeric_features(df: pd.DataFrame) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    coverage = df[num_cols].notna().mean()
    nonconst = df[num_cols].nunique(dropna=True) > 1
    feats = [
        c
        for c in num_cols
        if (coverage.get(c, 0.0) >= MIN_COVERAGE)
        and nonconst.get(c, False)
        and c not in META_EXCLUDE
    ]
    return feats


def compute_vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIF for columns in X (expects no NaNs).
    """
    X_ = sm.add_constant(X, has_constant="add")
    vifs = []
    for i, col in enumerate(X.columns):
        v = variance_inflation_factor(X_.values, i + 1)  # +1 to skip constant
        vifs.append({"feature": col, "VIF": float(v)})
    return (
        pd.DataFrame(vifs)
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


def basic_validation_checks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """
    Run simple consistency checks between train/test and return a text summary.
    """
    lines = []
    lines.append("=== BASIC DATA VALIDATION ===")
    lines.append(f"Train shape: {train_df.shape}")
    lines.append(f"Test shape:  {test_df.shape}")

    # Column consistency
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    only_in_train = sorted(train_cols - test_cols)
    only_in_test = sorted(test_cols - train_cols)
    common = sorted(train_cols & test_cols)

    lines.append(f"\nCommon columns: {len(common)}")
    if only_in_train:
        lines.append(f"Columns only in train: {only_in_train}")
    if only_in_test:
        lines.append(f"Columns only in test:  {only_in_test}")

    # Missingness
    for name, df in [("Train", train_df), ("Test", test_df)]:
        miss = df.isna().mean().sort_values(ascending=False)
        top_miss = miss[miss > 0].head(10)
        lines.append(f"\n{name} – top missing-value columns:")
        if top_miss.empty:
            lines.append("  (no missing values)")
        else:
            for col, frac in top_miss.items():
                lines.append(f"  {col}: {frac:.3f}")

    # Target checks
    for name, df in [("Train", train_df), ("Test", test_df)]:
        if TARGET in df.columns:
            vc = df[TARGET].value_counts(normalize=True).sort_index()
            lines.append(f"\n{name} – target distribution ({TARGET}):")
            for val, frac in vc.items():
                lines.append(f"  {val}: {frac:.3f}")
        else:
            lines.append(f"\n{name}: target column '{TARGET}' not found.")

    return "\n".join(lines)


def run_vif_diagnostics(train_df: pd.DataFrame) -> None:
    """
    Run raw and iterative VIF diagnostics on the training set.
    """
    df = ensure_month(train_df)

    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET]).copy()

    features = build_numeric_features(df)
    if not features:
        raise RuntimeError(
            "No usable numeric features after filtering. "
            "Check coverage/column types."
        )

    # Prepare design matrix: replace inf, impute, scale
    X_raw = df[features].copy()
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_raw),
        columns=features,
        index=df.index,
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    X = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=features,
        index=df.index,
    )

    # Raw VIF
    vif_raw = compute_vif_table(X)
    vif_raw.to_csv(OUTDIR / "vif_raw.csv", index=False)

    # Iterative pruning
    remaining = list(X.columns)
    history_rows = []

    for step in range(MAX_VIF_ITERS):
        vif_tbl = compute_vif_table(X[remaining])
        max_row = vif_tbl.iloc[0]
        history_rows.append(
            {
                "step": step,
                "dropped": None,
                "max_VIF": float(max_row["VIF"]),
                "max_feature": max_row["feature"],
                "n_features": len(remaining),
            }
        )

        if float(max_row["VIF"]) <= VIF_PRUNE_THRESHOLD:
            break

        drop_feat = max_row["feature"]
        remaining.remove(drop_feat)
        history_rows[-1]["dropped"] = drop_feat

    vif_iter_history = pd.DataFrame(history_rows)
    vif_iter_history.to_csv(OUTDIR / "vif_iterative_history.csv", index=False)

    vif_final = compute_vif_table(X[remaining])
    vif_final.to_csv(OUTDIR / "vif_final_kept.csv", index=False)

    dropped_features = [h["dropped"] for h in history_rows if h["dropped"]]
    pd.Series(dropped_features, name="dropped_feature").to_csv(
        OUTDIR / "vif_dropped_features.csv", index=False
    )

    # Console summary
    print("=== RAW VIF (Top 25) ===")
    print(vif_raw.head(25).to_string(index=False))

    print("\n=== ITERATIVE VIF PRUNING SUMMARY ===")
    print(f"Threshold: VIF ≤ {VIF_PRUNE_THRESHOLD}")
    print(
        f"Initial features: {X.shape[1]} | "
        f"Final kept: {len(remaining)} | Dropped: {len(dropped_features)}"
    )
    if dropped_features:
        print("Dropped (in order):", ", ".join(dropped_features))
    else:
        print("No drops required — all features under threshold.")

    print("\n=== FINAL KEPT FEATURES (with VIF) — Top 25 ===")
    print(vif_final.head(25).to_string(index=False))

    print(f"\nSaved:")
    print(f" - {OUTDIR.resolve()}/vif_raw.csv")
    print(f" - {OUTDIR.resolve()}/vif_iterative_history.csv")
    print(f" - {OUTDIR.resolve()}/vif_final_kept.csv")
    print(f" - {OUTDIR.resolve()}/vif_dropped_features.csv")


# ----------------------------------------------------------------------
# Leakage checks: one-month lag test & feature-swap test
# ----------------------------------------------------------------------

def last_k_month_masks(p: pd.Series, k: int) -> List[Tuple[np.ndarray, np.ndarray, object]]:
    """
    Build walk-forward train/test masks over the last k months.
    """
    months = sorted(p.unique())
    months = months[-k:] if k and k > 0 else months
    masks: List[Tuple[np.ndarray, np.ndarray, object]] = []
    for i, m in enumerate(months):
        tr_months = months[:i]
        if len(tr_months) < 1:
            continue
        te_mask = (p == m).values
        tr_mask = p.isin(tr_months).values
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue
        masks.append((tr_mask, te_mask, m))
    return masks


def fit_auc_over_masks(
    X: pd.DataFrame,
    y: pd.Series,
    masks: List[Tuple[np.ndarray, np.ndarray, object]],
) -> float:
    """
    Fit a simple logistic regression over multiple walk-forward splits
    and return mean ROC-AUC.
    """
    if not masks:
        return float("nan")

    aucs: List[float] = []
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=True, with_std=True)),
                    ]
                ),
                list(X.columns),
            )
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=200,
        random_state=RANDOM_STATE,
    )

    from sklearn.metrics import roc_auc_score

    for tr_mask, te_mask, _ in masks:
        tr_X, te_X = X.loc[tr_mask], X.loc[te_mask]
        tr_y, te_y = y.loc[tr_mask], y.loc[te_mask]
        pipe = Pipeline([("prep", prep), ("lr", model)])
        pipe.fit(tr_X, tr_y)
        p = pipe.predict_proba(te_X)[:, 1]
        aucs.append(roc_auc_score(te_y, p))
    return float(np.mean(aucs)) if aucs else float("nan")


def mutual_info_fast(x: pd.Series, y: pd.Series) -> float:
    z = pd.to_numeric(x, errors="coerce").astype("float32")
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.fillna(z.median()).to_numpy().reshape(-1, 1)
    yv = y.to_numpy().astype(int)
    try:
        return float(
            mutual_info_classif(
                z,
                yv,
                discrete_features=False,
                random_state=RANDOM_STATE,
            )[0]
        )
    except Exception:
        return float("nan")


def run_leakage_checks(train_df: pd.DataFrame) -> None:
    """
    Perform a quick leakage-oriented check using:
      - one-month-lag-only model vs current-features baseline
      - per-feature swap test for top-K mutual-information features
    """
    df = ensure_month(train_df)
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found in training data.")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    features = build_numeric_features(df)
    if not features:
        print("No numeric features available for leakage checks; skipping.")
        return

    # create 1-month lag features within each client
    df = df.sort_values(["client_id", "_month_period"]).copy()
    for f in features:
        df[f"{f}__lag1"] = (
            df.groupby("client_id", observed=True)[f].shift(1).astype("float32")
        )

    X_cur = df[features].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET]
    masks = last_k_month_masks(df["_month_period"], k=WALK_FORWARD_LAST_K)

    baseline_auc = fit_auc_over_masks(X_cur, y, masks)

    lag_feats = [f"{c}__lag1" for c in features if f"{c}__lag1" in df.columns]
    auc_all_lag = float("nan")
    if lag_feats:
        X_lag = df[lag_feats].replace([np.inf, -np.inf], np.nan)
        auc_all_lag = fit_auc_over_masks(X_lag, y, masks)

    # Mutual information to choose top-K features for swap test
    mi_vals = [(f, mutual_info_fast(df[f], y)) for f in features]
    mi_df = (
        pd.DataFrame(mi_vals, columns=["feature", "MI"])
        .sort_values("MI", ascending=False)
        .reset_index(drop=True)
    )
    audit_feats = mi_df["feature"].head(TOPK_MI).tolist()

    rows = []
    for f in audit_feats:
        lag_col = f"{f}__lag1"
        if lag_col in df.columns and df[lag_col].notna().any():
            X_swap = X_cur.copy()
            X_swap[f] = df[lag_col]
            auc_swap = fit_auc_over_masks(X_swap, y, masks)
            delta = baseline_auc - auc_swap
        else:
            auc_swap, delta = float("nan"), float("nan")
        rows.append(
            {
                "feature": f,
                "delta_auc_feature_lag": delta,
                "auc_with_feature_lag": auc_swap,
            }
        )

    per_feature_tbl = (
        pd.DataFrame(rows)
        .sort_values("delta_auc_feature_lag", ascending=False)
        .reset_index(drop=True)
    )
    per_feature_tbl.to_csv(
        OUTDIR / "per_feature_lag_swap_topK.csv",
        index=False,
    )

    print("\n=== LEAKAGE CHECKS (ONE-MONTH LAG & FEATURE SWAP) ===")
    print(f"Baseline WF AUC (last {len(masks)} months): {baseline_auc:.4f}")
    print(f"All-lag-only WF AUC:                       {auc_all_lag:.4f}")
    print(
        "\nPer-feature lag-swap (Top-K by MI):\n",
        per_feature_tbl.to_string(index=False),
    )


def main() -> None:
    # Load clean train/test
    train_df = pd.read_csv(get_data_file("train_clean.csv"))
    test_df = pd.read_csv(get_data_file("test_clean.csv"))

    # Basic validation summary
    summary = basic_validation_checks(train_df, test_df)
    summary_path = OUTDIR / "validation_summary.txt"
    summary_path.write_text(summary)
    print(summary)
    print(f"\nValidation summary saved to: {summary_path.resolve()}")

    # VIF diagnostics on train
    print("\nRunning VIF diagnostics on training data...")
    run_vif_diagnostics(train_df)

    # Leakage checks (one-month lag + feature swap)
    print("\nRunning leakage checks (one-month lag & feature swap)...")
    run_leakage_checks(train_df)


if __name__ == "__main__":
    main()


