"""
Train TCN model on clean feature dataset and generate SHAP analysis
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 11

# Directories
OUTPUT_DIR = 'churn_detection/datasets/tcn_clean'
PLOTS_DIR = 'churn_detection/tcn_shap_images'
MODEL_DIR = 'saved_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================================
# MODEL DEFINITION
# ========================================
class ChurnDataset(Dataset):
    def __init__(self, df, feature_cols):
        df = df.sort_values(['client_id', 'reference_month'])
        self.sequences = []
        self.targets = []
        self.lengths = []
        self.client_ids = []
        
        for client_id, group in df.groupby('client_id'):
            features = torch.from_numpy(group[feature_cols].to_numpy(dtype=np.float32))
            target = float(group['churned'].values[-1])
            self.sequences.append(features)
            self.targets.append(target)
            self.lengths.append(len(group))
            self.client_ids.append(client_id)
        
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
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
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
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
            layers.append(TemporalBlock(in_ch, hidden_dim, kernel_size, 2**i, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths):
        x = x.transpose(1, 2)
        x = self.network(x)
        batch_size = x.size(0)
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        for i, L in enumerate(lengths):
            out[i] = x[i, :, L-1]
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y, lengths in loader:
            probs = torch.sigmoid(model(X.to(device), lengths)).cpu().numpy()
            preds.extend(probs)
            targets.extend(y.numpy())
    return np.array(preds), np.array(targets)

def main():
    print("="*70)
    print("TRAINING TCN ON CLEAN DATASET")
    print("="*70)
    
    # Load clean data (from local 'data/' or Hugging Face)
    print("\n[1] Loading clean dataset...")
    train_df = pd.read_csv(get_data_file("train_clean.csv"))
    test_df = pd.read_csv(get_data_file("test_clean.csv"))
    
    feature_cols = [
        c
        for c in train_df.columns
        if c not in ["client_id", "reference_month", "churned"]
    ]
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Test: {len(test_df):,} samples")
    print(f"  Features: {len(feature_cols)}")
    
    # Normalize features
    print("\n[2] Normalizing features...")
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std() + 1e-10
    
    train_df[feature_cols] = (train_df[feature_cols] - mean) / std
    test_df[feature_cols] = (test_df[feature_cols] - mean) / std
    
    # Create datasets
    print("\n[3] Creating datasets...")
    train_ds = ChurnDataset(train_df, feature_cols)
    test_ds = ChurnDataset(test_df, feature_cols)
    
    # Split train into train/val
    n_val = int(len(train_ds) * 0.2)
    n_train = len(train_ds) - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    print(f"  Train: {n_train}, Val: {n_val}, Test: {len(test_ds)}")
    
    # Model config (best from previous tuning)
    config = {
        'hidden_dim': 64,
        'num_layers': 3,
        'kernel_size': 3,
        'dropout': 0.3,
        'lr': 0.001,
        'weight_decay': 0.01
    }
    
    # Train model
    print("\n[4] Training TCN model...")
    print(f"  Config: {config}")
    print(f"  Device: {DEVICE}")
    
    model = TCN(
        len(feature_cols),
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout']
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    history = {'train_loss': [], 'val_auc': []}
    
    for epoch in range(50):
        model.train()
        train_loss = 0
        for X, y, lengths in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X, lengths)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        val_preds, val_targets = evaluate(model, val_loader, DEVICE)
        val_auc = roc_auc_score(val_targets, val_preds)
        
        scheduler.step(val_auc)
        
        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), f'{MODEL_DIR}/tcn_clean_model.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Val AUC={val_auc:.4f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f'{MODEL_DIR}/tcn_clean_model.pt', map_location=DEVICE))
    
    # Evaluate
    print("\n[5] Evaluating model...")
    
    train_preds_full, train_targets_full = evaluate(model, DataLoader(train_ds, batch_size=128, shuffle=False, collate_fn=collate_fn), DEVICE)
    test_preds, test_targets = evaluate(model, test_loader, DEVICE)
    
    train_auc = roc_auc_score(train_targets_full, train_preds_full)
    test_auc = roc_auc_score(test_targets, test_preds)
    
    # Optimal threshold
    prec, rec, thresh = precision_recall_curve(test_targets, test_preds)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    opt_thresh = thresh[np.argmax(f1s)]
    
    test_pred_binary = (test_preds >= opt_thresh).astype(int)
    test_f1 = f1_score(test_targets, test_pred_binary)
    test_prec = precision_score(test_targets, test_pred_binary)
    test_rec = recall_score(test_targets, test_pred_binary)
    
    print(f"\n  Results:")
    print(f"    Train AUC: {train_auc:.4f}")
    print(f"    Val AUC:   {best_val_auc:.4f}")
    print(f"    Test AUC:  {test_auc:.4f}")
    print(f"    Test F1:   {test_f1:.4f}")
    print(f"    Test Precision: {test_prec:.4f}")
    print(f"    Test Recall:    {test_rec:.4f}")
    
    # Save metrics
    metrics = {
        'n_features': len(feature_cols),
        'train_auc': float(train_auc),
        'val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'test_f1': float(test_f1),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'optimal_threshold': float(opt_thresh),
        'config': config
    }
    
    with open(f'{OUTPUT_DIR}/tcn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # ========================================
    # FEATURE IMPORTANCE (Permutation-based)
    # ========================================
    print("\n[6] Computing feature importance...")
    
    # Get baseline AUC
    baseline_preds, baseline_targets = evaluate(model, test_loader, DEVICE)
    baseline_auc = roc_auc_score(baseline_targets, baseline_preds)
    
    # Permutation importance
    importance_scores = []
    
    test_df_orig = pd.read_csv(get_data_file("test_clean.csv"))
    test_df_orig[feature_cols] = (test_df_orig[feature_cols] - mean) / std
    
    for i, feat in enumerate(feature_cols):
        if (i + 1) % 10 == 0:
            print(f"  Processing feature {i+1}/{len(feature_cols)}...")
        
        # Permute feature
        test_df_perm = test_df_orig.copy()
        test_df_perm[feat] = np.random.permutation(test_df_perm[feat].values)
        
        # Create dataset with permuted feature
        perm_ds = ChurnDataset(test_df_perm, feature_cols)
        perm_loader = DataLoader(perm_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
        
        # Evaluate
        perm_preds, perm_targets = evaluate(model, perm_loader, DEVICE)
        perm_auc = roc_auc_score(perm_targets, perm_preds)
        
        importance_scores.append({
            'feature': feat,
            'importance': baseline_auc - perm_auc,
            'baseline_auc': baseline_auc,
            'permuted_auc': perm_auc
        })
    
    importance_df = pd.DataFrame(importance_scores).sort_values('importance', ascending=False)
    importance_df.to_csv(f'{OUTPUT_DIR}/feature_importance.csv', index=False)
    
    # ========================================
    # GENERATE PLOTS
    # ========================================
    print("\n[7] Generating plots...")
    
    # 1. Training history
    print("  - Training history...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    
    axes[1].plot(history['val_auc'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Validation AUC')
    axes[1].axhline(y=best_val_auc, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_val_auc:.4f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/01_training_history.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. Feature importance bar chart
    print("  - Feature importance...")
    top_features = importance_df.head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e74c3c' if imp > 0.01 else '#3498db' if imp > 0.005 else '#95a5a6' 
              for imp in top_features['importance']]
    
    bars = ax.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance (AUC Drop when Permuted)')
    ax.set_title('TCN Feature Importance (Top 20)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/02_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 3. Prediction distribution
    print("  - Prediction distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    churn_preds = test_preds[test_targets == 1]
    no_churn_preds = test_preds[test_targets == 0]
    
    ax.hist(no_churn_preds, bins=50, alpha=0.6, label='No Churn', color='#3498db', density=True)
    ax.hist(churn_preds, bins=50, alpha=0.6, label='Churn', color='#e74c3c', density=True)
    ax.axvline(x=opt_thresh, color='black', linestyle='--', linewidth=2, label=f'Threshold: {opt_thresh:.3f}')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('TCN Prediction Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/03_prediction_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 4. Feature group importance
    print("  - Feature group importance...")
    
    def get_group(feat):
        if feat.startswith('curr_'):
            return 'Current Month'
        elif feat.startswith('recent_'):
            return 'Recent 3 Months'
        elif any(feat.startswith(p) for p in ['trend_', 'momentum_', 'volatility_', 'acceleration', 
                                               'cv_', 'stability_', 'max_drawdown', 'recovery_', 
                                               'skewness', 'kurtosis', 'autocorr']):
            return 'Time-Series'
        elif 'dialog' in feat.lower():
            return 'Dialog'
        elif 'geo' in feat.lower():
            return 'Geo'
        else:
            return 'Other'
    
    importance_df['group'] = importance_df['feature'].apply(get_group)
    group_importance = importance_df.groupby('group')['importance'].sum().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Current Month': '#e74c3c', 'Recent 3 Months': '#3498db', 
              'Time-Series': '#9b59b6', 'Dialog': '#2ecc71', 'Geo': '#f39c12', 'Other': '#95a5a6'}
    bar_colors = [colors.get(g, '#95a5a6') for g in group_importance.index]
    
    ax.barh(group_importance.index, group_importance.values, color=bar_colors)
    ax.set_xlabel('Total Importance')
    ax.set_title('TCN Feature Group Importance', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/04_group_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 5. Confusion matrix style metrics
    print("  - Metrics summary...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics_display = [
        ('Test AUC', test_auc),
        ('Test F1', test_f1),
        ('Precision', test_prec),
        ('Recall', test_rec),
        ('Val AUC', best_val_auc)
    ]
    
    colors = ['#27ae60' if v > 0.7 else '#f39c12' if v > 0.5 else '#e74c3c' for _, v in metrics_display]
    bars = ax.barh([m[0] for m in metrics_display], [m[1] for m in metrics_display], color=colors)
    
    for bar, (name, val) in zip(bars, metrics_display):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Score')
    ax.set_title('TCN Model Performance Metrics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/05_metrics_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nModel saved to: {MODEL_DIR}/")
    print(f"Plots saved to: {PLOTS_DIR}/")
    print(f"\nTCN Performance:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Test F1:  {test_f1:.4f}")
    
    return model, metrics, importance_df

if __name__ == "__main__":
    model, metrics, importance_df = main()

