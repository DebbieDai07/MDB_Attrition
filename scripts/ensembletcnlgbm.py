"""
Ensemble of TCN + LightGBM on the CLEAN churn dataset.

This is a refactored, notebook-free script that:
  - Loads clean train/test data from local `data/` or Hugging Face
  - Uses the saved clean TCN and LightGBM models
  - Builds a simple stacked ensemble with Logistic Regression
  - Prints precision, recall, F1, PR-AUC, and ROC-AUC on the test set

Requirements (Python):
  - pandas, numpy, scikit-learn, lightgbm, torch, huggingface_hub
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Data access (clean data via local files or Hugging Face)
# ----------------------------------------------------------------------

HF_DATASET_REPO = "Saravanan1999/MDBAttrition"


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


# ----------------------------------------------------------------------
# TCN definitions (aligned with scripts/train_tcn_clean.py)
# ----------------------------------------------------------------------

class ChurnDatasetWithIds(Dataset):
    """
    Sequence dataset that groups rows by client_id and keeps client IDs.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        df = df.sort_values(["client_id", "reference_month"])
        self.sequences: List[torch.Tensor] = []
        self.targets: torch.Tensor
        self.lengths: List[int] = []
        self.client_ids: List[int] = []

        targets = []
        for client_id, group in df.groupby("client_id"):
            features = torch.from_numpy(
                group[feature_cols].to_numpy(dtype=np.float32)
            )
            target = float(group["churned"].values[-1])
            self.sequences.append(features)
            targets.append(target)
            self.lengths.append(len(group))
            self.client_ids.append(client_id)

        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return (
            self.sequences[idx],
            self.targets[idx],
            self.lengths[idx],
            self.client_ids[idx],
        )


def collate_fn_ensemble(batch):
    X_list, y_list, lengths, client_ids = zip(*batch)
    max_len = max(lengths)
    B, F = len(batch), X_list[0].shape[1]
    X_padded = torch.zeros(B, max_len, F, dtype=torch.float32)
    for i, (X, L) in enumerate(zip(X_list, lengths)):
        X_padded[i, :L] = X
    return (
        X_padded,
        torch.stack(list(y_list)),
        torch.tensor(lengths, dtype=torch.int64),
        list(client_ids),
    )


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size, padding=padding, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.padding = padding
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn1(out)
        out = self.relu(self.dropout(out))

        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn2(out)
        out = self.relu(self.dropout(out))

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_dim
            layers.append(
                TemporalBlock(
                    in_ch,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.network(x)
        batch_size = x.size(0)
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        for i, L in enumerate(lengths):
            out[i] = x[i, :, L - 1]
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


def predict_tcn(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (client_ids, probabilities, labels) at client level.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_ensemble,
    )
    model.eval()
    all_probs, all_targets, all_client_ids = [], [], []
    with torch.no_grad():
        for X, y, lengths, client_ids in loader:
            X = X.to(device)
            logits = model(X, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())
            all_client_ids.extend(client_ids)
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    return np.array(all_client_ids), probs, targets


# ----------------------------------------------------------------------
# LightGBM client-level helper
# ----------------------------------------------------------------------

def get_lgb_client_probs(df_raw: pd.DataFrame, model: lgb.Booster):
    """
    Collapse to one row per client (last reference_month) and
    return (client_ids, probabilities, labels).
    """
    feature_cols = model.feature_name()
    missing = [f for f in feature_cols if f not in df_raw.columns]
    if missing:
        raise ValueError(
            f"The following LightGBM features are missing in dataframe: {missing}"
        )

    df_sorted = df_raw.sort_values(["client_id", "reference_month"])
    last_rows = df_sorted.groupby("client_id").tail(1)

    X = last_rows[feature_cols].values
    y = last_rows["churned"].values
    client_ids = last_rows["client_id"].values

    best_iter = model.best_iteration
    if best_iter is None or best_iter <= 0:
        best_iter = model.num_trees()

    probs = model.predict(X, num_iteration=best_iter)
    return np.array(client_ids), probs, y


# ----------------------------------------------------------------------
# Meta-features and ensemble
# ----------------------------------------------------------------------

def build_meta_features(p_tcn: np.ndarray, p_lgb: np.ndarray) -> np.ndarray:
    """
    Build basic meta-features from base model probabilities.
    Columns:
      0: p_tcn
      1: p_lgb
      2: p_avg  = 0.5 * (p_tcn + p_lgb)
      3: p_diff = |p_tcn - p_lgb|
    """
    p_tcn = np.asarray(p_tcn)
    p_lgb = np.asarray(p_lgb)
    p_avg = 0.5 * (p_tcn + p_lgb)
    p_diff = np.abs(p_tcn - p_lgb)
    return np.column_stack([p_tcn, p_lgb, p_avg, p_diff])


@dataclass
class EnsembleConfig:
    min_precision: float = 0.50  # for recall-prioritized threshold


def run_ensemble(cfg: EnsembleConfig | None = None) -> None:
    """
    Run a simple stacked ensemble using the clean TCN & LightGBM models.
    """
    if cfg is None:
        cfg = EnsembleConfig()

    # ------------------------------------------------------------------
    # Load clean data
    # ------------------------------------------------------------------
    train_df = pd.read_csv(get_data_file("train_clean.csv"))
    test_df = pd.read_csv(get_data_file("test_clean.csv"))

    feature_cols = [
        c for c in train_df.columns
        if c not in ["client_id", "reference_month", "churned"]
    ]

    # Normalize (same pattern as training)
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std() + 1e-10
    train_df[feature_cols] = (train_df[feature_cols] - mean) / std
    test_df[feature_cols] = (test_df[feature_cols] - mean) / std

    # ------------------------------------------------------------------
    # Load base models (clean)
    # ------------------------------------------------------------------
    # TCN
    input_size = len(feature_cols)
    tcn_model = TCN(input_size=input_size).to(DEVICE)
    tcn_state = torch.load("saved_models/tcn_clean_model.pt", map_location=DEVICE)
    tcn_model.load_state_dict(tcn_state)
    tcn_model.to(DEVICE)

    # LightGBM
    lgb_model = lgb.Booster(model_file="saved_models/lightgbm_clean_model.txt")

    # ------------------------------------------------------------------
    # TCN client-level predictions
    # ------------------------------------------------------------------
    train_ds_tcn = ChurnDatasetWithIds(train_df, feature_cols)
    test_ds_tcn = ChurnDatasetWithIds(test_df, feature_cols)

    train_cids_tcn, train_p_tcn, train_y = predict_tcn(tcn_model, train_ds_tcn, DEVICE)
    test_cids_tcn, test_p_tcn, test_y = predict_tcn(tcn_model, test_ds_tcn, DEVICE)

    # ------------------------------------------------------------------
    # LightGBM client-level predictions
    # ------------------------------------------------------------------
    train_cids_lgb, train_p_lgb, train_y_lgb = get_lgb_client_probs(train_df, lgb_model)
    test_cids_lgb, test_p_lgb, test_y_lgb = get_lgb_client_probs(test_df, lgb_model)

    # Align by client_id
    train_lgb_dict = {
        cid: (p, y) for cid, p, y in zip(train_cids_lgb, train_p_lgb, train_y_lgb)
    }
    test_lgb_dict = {
        cid: (p, y) for cid, p, y in zip(test_cids_lgb, test_p_lgb, test_y_lgb)
    }

    train_p_lgb_aligned = []
    for cid, y_t in zip(train_cids_tcn, train_y):
        if cid not in train_lgb_dict:
            raise ValueError(f"Client ID {cid} in TCN train set not in LightGBM train set.")
        p_l, y_l = train_lgb_dict[cid]
        train_p_lgb_aligned.append(p_l)
        # labels should match; we ignore any rare mismatches
    train_p_lgb_aligned = np.array(train_p_lgb_aligned)

    test_p_lgb_aligned = []
    for cid, y_t in zip(test_cids_tcn, test_y):
        if cid not in test_lgb_dict:
            raise ValueError(f"Client ID {cid} in TCN test set not in LightGBM test set.")
        p_l, y_l = test_lgb_dict[cid]
        test_p_lgb_aligned.append(p_l)
    test_p_lgb_aligned = np.array(test_p_lgb_aligned)

    # ------------------------------------------------------------------
    # Build meta-features and train Logistic Regression meta-model
    # ------------------------------------------------------------------
    X_train_meta = build_meta_features(train_p_tcn, train_p_lgb_aligned)
    X_test_meta = build_meta_features(test_p_tcn, test_p_lgb_aligned)
    y_train_meta = train_y
    y_test = test_y

    meta_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
    )
    meta_model.fit(X_train_meta, y_train_meta)

    # ------------------------------------------------------------------
    # Evaluate ensemble on TEST
    # ------------------------------------------------------------------
    test_p_meta = meta_model.predict_proba(X_test_meta)[:, 1]

    roc_auc = roc_auc_score(y_test, test_p_meta)
    pr_auc = average_precision_score(y_test, test_p_meta)

    precisions, recalls, thresholds = precision_recall_curve(y_test, test_p_meta)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)

    # Best-F1 threshold
    best_f1_idx = np.argmax(f1_scores)
    if best_f1_idx < len(thresholds):
        best_f1_thresh = thresholds[best_f1_idx]
    else:
        best_f1_thresh = 0.5

    y_pred_best = (test_p_meta >= best_f1_thresh).astype(int)
    precision_best = precision_score(y_test, y_pred_best, zero_division=0)
    recall_best = recall_score(y_test, y_pred_best, zero_division=0)
    f1_best = f1_score(y_test, y_pred_best, zero_division=0)

    # Recall-prioritized threshold (optional)
    recall_prior = 0.0
    precision_prior = 0.0
    f1_prior = 0.0
    thresh_prior = 0.5
    for p, r, t in zip(precisions, recalls, np.append(thresholds, 1.0)):
        if p >= cfg.min_precision and r > recall_prior:
            recall_prior = r
            precision_prior = p
            if p + r > 0:
                f1_prior = 2 * p * r / (p + r)
            else:
                f1_prior = 0.0
            thresh_prior = t

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("ENSEMBLE (TCN + LightGBM CLEAN) - STACKING RESULTS ON TEST")
    print("=" * 70)
    print(f"Overall ROC-AUC (probs): {roc_auc:.4f}")
    print(f"Overall PR-AUC  (probs): {pr_auc:.4f}")

    print("\n--- Best F1 threshold ---")
    print(f"Threshold: {best_f1_thresh:.4f}")
    print(f"Precision: {precision_best:.4f}")
    print(f"Recall:    {recall_best:.4f}")
    print(f"F1:        {f1_best:.4f}")

    print(f"\n--- Recall-prioritized threshold (precision >= {cfg.min_precision:.2f}) ---")
    print(f"Threshold: {thresh_prior:.4f}")
    print(f"Precision: {precision_prior:.4f}")
    print(f"Recall:    {recall_prior:.4f}")
    print(f"F1:        {f1_prior:.4f}")


if __name__ == "__main__":
    run_ensemble()


