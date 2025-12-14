"""
Train LightGBM with cleaned feature set (removing misleading features)
Then generate SHAP analysis report
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 11

# Output directories
OUTPUT_DIR = 'churn_detection/datasets/features_clean'
PLOTS_DIR = 'churn_detection/shap_clean_images'
MODEL_DIR = 'saved_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("="*70)
    print("TRAINING CLEAN MODEL WITH MEANINGFUL FEATURES ONLY")
    print("="*70)
    
    # Load original data
    print("\n[1] Loading data...")
    train_df = pd.read_csv('churn_detection/datasets/features_v3/train_enhanced.csv')
    test_df = pd.read_csv('churn_detection/datasets/features_v3/test_enhanced.csv')
    
    with open('churn_detection/datasets/features_v3/feature_columns_enhanced.txt', 'r') as f:
        all_features = [line.strip() for line in f if line.strip()]
    all_features = [c for c in all_features if c in train_df.columns]
    
    print(f"  Original features: {len(all_features)}")
    print(f"  Train samples: {len(train_df):,}")
    print(f"  Test samples: {len(test_df):,}")
    
    # ========================================
    # REMOVE MISLEADING FEATURES
    # ========================================
    print("\n[2] Removing misleading features...")
    
    # Features to remove (weak direct effect / misleading SHAP)
    remove_features = [
        'curr_trx_amount_std',
        'curr_trx_amount_sum', 
        'recent_trx_amount_std',
        'recent_trx_amount_sum',
        'recent_trx_amount_max',
        'recent_trx_amount_min',
        'curr_trx_amount_mean',
        'recent_trx_amount_mean',
        'recent_trx_amount_cv',  # derived from std/mean
    ]
    
    # Keep only meaningful features
    clean_features = [f for f in all_features if f not in remove_features]
    
    print(f"  Removed features: {len(remove_features)}")
    print(f"  Clean features: {len(clean_features)}")
    print(f"\n  Removed:")
    for f in remove_features:
        if f in all_features:
            print(f"    - {f}")
    
    # ========================================
    # PREPARE CLEAN DATASETS
    # ========================================
    print("\n[3] Preparing clean datasets...")
    
    X_train = train_df[clean_features].values
    X_test = test_df[clean_features].values
    y_train = train_df['churned'].values
    y_test = test_df['churned'].values
    
    # Save clean datasets
    train_clean = train_df[['client_id', 'reference_month', 'churned'] + clean_features].copy()
    test_clean = test_df[['client_id', 'reference_month', 'churned'] + clean_features].copy()
    
    train_clean.to_csv(f'{OUTPUT_DIR}/train_clean.csv', index=False)
    test_clean.to_csv(f'{OUTPUT_DIR}/test_clean.csv', index=False)
    
    with open(f'{OUTPUT_DIR}/feature_columns_clean.txt', 'w') as f:
        for feat in clean_features:
            f.write(feat + '\n')
    
    print(f"  Saved to {OUTPUT_DIR}/")
    
    # ========================================
    # TRAIN MODEL
    # ========================================
    print("\n[4] Training LightGBM model...")
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=clean_features)
    
    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum()
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Save model
    model.save_model(f'{MODEL_DIR}/lightgbm_clean_model.txt')
    print(f"  Model saved to {MODEL_DIR}/lightgbm_clean_model.txt")
    
    # ========================================
    # EVALUATE
    # ========================================
    print("\n[5] Evaluating model...")
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_auc = roc_auc_score(y_train, train_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    
    # Optimal threshold
    prec, rec, thresh = precision_recall_curve(y_test, test_preds)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    opt_thresh = thresh[np.argmax(f1s)]
    
    test_pred_binary = (test_preds >= opt_thresh).astype(int)
    test_f1 = f1_score(y_test, test_pred_binary)
    test_prec = precision_score(y_test, test_pred_binary)
    test_rec = recall_score(y_test, test_pred_binary)
    
    print(f"\n  Results:")
    print(f"    Train AUC: {train_auc:.4f}")
    print(f"    Test AUC:  {test_auc:.4f}")
    print(f"    Test F1:   {test_f1:.4f}")
    print(f"    Test Precision: {test_prec:.4f}")
    print(f"    Test Recall:    {test_rec:.4f}")
    print(f"    Optimal Threshold: {opt_thresh:.3f}")
    
    # ========================================
    # SHAP ANALYSIS
    # ========================================
    print("\n[6] Computing SHAP values...")
    
    sample_size = min(1000, len(X_test))
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[sample_idx]
    y_sample = y_test[sample_idx]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    shap_df = pd.DataFrame(shap_values, columns=clean_features)
    X_df = pd.DataFrame(X_sample, columns=clean_features)
    
    # ========================================
    # GENERATE PLOTS
    # ========================================
    print("\n[7] Generating SHAP plots...")
    
    # 1. Summary Beeswarm
    print("  - Summary beeswarm...")
    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=clean_features, 
                      max_display=20, show=False)
    plt.title('Feature Importance (SHAP Values)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/01_summary_beeswarm.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. Summary Bar
    print("  - Summary bar...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=clean_features, 
                      plot_type="bar", max_display=20, show=False)
    plt.title('Mean |SHAP| Feature Importance', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/02_summary_bar.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 3-7. Dependence plots for top 5 features
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    top_features = mean_abs_shap.head(5).index.tolist()
    
    for i, feat in enumerate(top_features, 1):
        print(f"  - Dependence: {feat}...")
        feat_idx = clean_features.index(feat)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feat_idx, shap_values, X_sample, 
                            feature_names=clean_features, 
                            interaction_index='auto',
                            show=False)
        plt.title(f'SHAP Dependence: {feat}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/0{i+2}_dependence_{feat}.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # 8. Waterfall - High Churn
    print("  - Waterfall high churn...")
    test_preds_sample = model.predict(X_sample)
    churn_idx = np.where(y_sample == 1)[0]
    if len(churn_idx) > 0:
        high_idx = churn_idx[np.argmax(test_preds_sample[churn_idx])]
        
        shap_exp = shap.Explanation(
            values=shap_values[high_idx],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_sample[high_idx],
            feature_names=clean_features
        )
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_exp, max_display=15, show=False)
        plt.title('High Churn Risk Customer', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/08_waterfall_high_churn.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # 9. Waterfall - Low Churn
    print("  - Waterfall low churn...")
    no_churn_idx = np.where(y_sample == 0)[0]
    if len(no_churn_idx) > 0:
        low_idx = no_churn_idx[np.argmin(test_preds_sample[no_churn_idx])]
        
        shap_exp = shap.Explanation(
            values=shap_values[low_idx],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_sample[low_idx],
            feature_names=clean_features
        )
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_exp, max_display=15, show=False)
        plt.title('Low Churn Risk Customer', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/09_waterfall_low_churn.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # 10. Feature Group Contribution
    print("  - Feature group contribution...")
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
    
    group_contrib = shap_df.abs().sum().groupby([get_group(f) for f in clean_features]).sum().sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    colors = {'Current Month': '#e74c3c', 'Recent 3 Months': '#3498db', 
              'Time-Series': '#9b59b6', 'Dialog': '#2ecc71', 'Geo': '#f39c12', 'Other': '#95a5a6'}
    bar_colors = [colors.get(g, '#95a5a6') for g in group_contrib.index]
    
    plt.barh(group_contrib.index, group_contrib.values, color=bar_colors)
    plt.xlabel('Total |SHAP| Contribution', fontsize=12)
    plt.title('SHAP Contribution by Feature Group', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/10_group_contribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ========================================
    # SAVE METRICS
    # ========================================
    print("\n[8] Saving results...")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': clean_features,
        'mean_abs_shap': mean_abs_shap[clean_features].values,
        'lgb_importance': model.feature_importance('gain')
    }).sort_values('mean_abs_shap', ascending=False)
    
    importance_df.to_csv(f'{OUTPUT_DIR}/feature_importance.csv', index=False)
    
    # Model metrics
    metrics = {
        'n_features': len(clean_features),
        'train_auc': train_auc,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'optimal_threshold': opt_thresh
    }
    
    pd.DataFrame([metrics]).to_csv(f'{OUTPUT_DIR}/model_metrics.csv', index=False)
    
    print(f"\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nClean dataset: {OUTPUT_DIR}/")
    print(f"SHAP plots: {PLOTS_DIR}/")
    print(f"\nModel Performance:")
    print(f"  Features: {len(clean_features)}")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Test F1:  {test_f1:.4f}")
    
    return clean_features, metrics, importance_df

if __name__ == "__main__":
    clean_features, metrics, importance_df = main()

