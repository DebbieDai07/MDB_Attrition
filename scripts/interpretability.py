"""
Interpretability utilities for the MDB_Attrition project.

This module focuses on:
  - Evaluating the trained TCN model on the clean test set
  - Computing SHAP-based global feature importance
  - Producing per-client top-k SHAP explanations

The code here is a cleaned-up, standalone version of the
TCN interpretability logic that previously lived in
`ensembletcnlgbm.py`.
"""

import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
import shap

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Data access
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
# Dataset & model (same logic as training)
# ----------------------------------------------------------------------

class ChurnDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        df = df.sort_values(["client_id", "reference_month"])
        self.sequences = []
        self.targets = []
        self.lengths = []

        for _, group in df.groupby("client_id"):
            features = torch.from_numpy(group[feature_cols].to_numpy(dtype=np.float32))
            target = float(group["churned"].values[-1])
            self.sequences.append(features)
            self.targets.append(target)
            self.lengths.append(len(group))

        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.targets[idx], self.lengths[idx]


def collate_fn(batch):
    X_list, y_list, lengths = zip(*batch)
    max_len = max(lengths)
    B, F = len(batch), X_list[0].shape[1]
    X_padded = torch.zeros(B, max_len, F)
    for i, (X, L) in enumerate(zip(X_list, lengths)):
        X_padded[i, :L] = X
    return X_padded, torch.stack(list(y_list)), torch.tensor(lengths, dtype=torch.int64)


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
    def __init__(self, input_size, hidden_dim=64, num_layers=3, kernel_size=3, dropout=0.3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_dim
            layers.append(
                TemporalBlock(
                    in_ch,
                    hidden_dim,
                    kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T)
        x = self.network(x)
        batch_size = x.size(0)
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        for i, L in enumerate(lengths):
            out[i] = x[i, :, L - 1]
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


class WrappedTCN(nn.Module):
    """Wrapper so SHAP can pass float lengths."""

    def __init__(self, base_model: TCN):
        super().__init__()
        self.base_model = base_model

    def forward(self, X, lengths):
        lengths_long = lengths.long()
        out = self.base_model(X, lengths_long)
        return out.unsqueeze(-1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)
            targets.extend(y.cpu().numpy())
    return np.array(preds), np.array(targets)


# ----------------------------------------------------------------------
# Main SHAP pipeline
# ----------------------------------------------------------------------

def run_tcn_shap(
    model_path: str = "saved_models/tcn_clean_model.pt",
    output_dir: str = "tcn_interpretability",
    max_background: int = 100,
    max_explain: int = 100,
    top_k_client_reasons: int = 5,
) -> None:
    """
    Run SHAP-based interpretability for the trained TCN model.

    - Loads clean train/test from local data/ or Hugging Face
    - Normalizes using train stats
    - Evaluates TCN on test and prints metrics
    - Computes global SHAP feature importance and per-client reasons
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load clean data
    train_df = pd.read_csv(get_data_file("train_clean.csv"))
    test_df = pd.read_csv(get_data_file("test_clean.csv"))

    feature_cols = [
        c for c in train_df.columns
        if c not in ["client_id", "reference_month", "churned"]
    ]

    # Normalize
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std() + 1e-10
    train_df[feature_cols] = (train_df[feature_cols] - mean) / std
    test_df[feature_cols] = (test_df[feature_cols] - mean) / std

    # Dataset / loader
    test_ds = ChurnDataset(test_df, feature_cols)
    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Load model
    input_size = len(feature_cols)
    model = TCN(input_size=input_size).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    # Evaluate
    probs, y_true = evaluate(model, test_loader, DEVICE)
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    y_pred = (probs >= 0.5).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("=== TCN evaluation on clean test set ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Precision@0.5: {precision:.4f}")
    print(f"Recall@0.5:    {recall:.4f}")
    print(f"F1@0.5:        {f1:.4f}")

    # SHAP setup
    wrapped_model = WrappedTCN(model).to(DEVICE)
    wrapped_model.eval()

    # Build background / explanation sets
    num_bg = min(max_background, len(test_ds))
    num_explain = min(max_explain, len(test_ds))
    all_indices = list(range(len(test_ds)))
    rng = np.random.RandomState(42)
    rng.shuffle(all_indices)
    bg_indices = all_indices[:num_bg]
    explain_indices = all_indices[:num_explain]

    def build_padded_tensors(idxs: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        subset_lengths = [test_ds.lengths[i] for i in idxs]
        max_len = max(subset_lengths)
        F = test_ds.sequences[0].shape[1]
        B = len(idxs)
        X_padded = torch.zeros(B, max_len, F, dtype=torch.float32)
        lengths = torch.zeros(B, dtype=torch.float32)
        for j, ds_idx in enumerate(idxs):
            seq = test_ds.sequences[ds_idx]
            L = test_ds.lengths[ds_idx]
            X_padded[j, :L] = seq
            lengths[j] = float(L)
        return X_padded.to(DEVICE), lengths.to(DEVICE)

    X_bg, L_bg = build_padded_tensors(bg_indices)
    X_ex, L_ex = build_padded_tensors(explain_indices)

    explainer = shap.DeepExplainer(wrapped_model, [X_bg, L_bg])
    shap_values = explainer.shap_values([X_ex, L_ex], check_additivity=False)
    if isinstance(shap_values, list):
        shap_vals_x = shap_values[0]
    else:
        shap_vals_x = shap_values
    if isinstance(shap_vals_x, torch.Tensor):
        shap_vals_x = shap_vals_x.cpu().numpy()
    if shap_vals_x.ndim == 4 and shap_vals_x.shape[-1] == 1:
        shap_vals_x = shap_vals_x[..., 0]

    B_ex, T_ex, F_ex = shap_vals_x.shape
    print("SHAP values shape:", shap_vals_x.shape)

    # Global importance
    global_importance = np.abs(shap_vals_x).mean(axis=(0, 1))
    feat_importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": global_importance}
    ).sort_values("importance", ascending=False)
    feat_importance_path = os.path.join(output_dir, "tcn_shap_global_importance.csv")
    feat_importance_df.to_csv(feat_importance_path, index=False)
    print(f"Saved global SHAP importance to {feat_importance_path}")

    # Per-client reasons
    per_client_importance = np.abs(shap_vals_x).mean(axis=1)  # (B_ex, F)
    test_client_ids = (
        test_df.sort_values(["client_id", "reference_month"])["client_id"]
        .drop_duplicates()
        .tolist()
    )
    explain_client_ids = [test_client_ids[i] for i in explain_indices]

    rows = []
    for row_idx, client_id in enumerate(explain_client_ids):
        imp_vec = per_client_importance[row_idx]
        top_idx = np.argsort(-imp_vec)[:top_k_client_reasons]
        row = {"client_id": client_id}
        for k, fi in enumerate(top_idx, start=1):
            row[f"reason_{k}"] = feature_cols[fi]
            row[f"reason_{k}_score"] = float(imp_vec[fi])
        rows.append(row)

    explanations_df = pd.DataFrame(rows)
    explanations_path = os.path.join(output_dir, "tcn_shap_explanations.csv")
    explanations_df.to_csv(explanations_path, index=False)
    print(f"Saved per-client SHAP explanations to {explanations_path}")


if __name__ == "__main__":
    run_tcn_shap()


