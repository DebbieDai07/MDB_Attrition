================================================================================
CHURN PREDICTION MODEL PACKAGE
================================================================================

OVERVIEW
--------
This package contains the clean churn prediction models (LightGBM and TCN)
trained on 62 features after removing 9 redundant/misleading features.

Removed features (based on SHAP analysis showing weak direct effects):
- curr_trx_amount_std, curr_trx_amount_sum, curr_trx_amount_mean
- recent_trx_amount_std, recent_trx_amount_sum, recent_trx_amount_mean
- recent_trx_amount_max, recent_trx_amount_min, recent_trx_amount_cv


MODEL PERFORMANCE
-----------------
                    LightGBM    TCN
Test AUC:           0.914       0.912
Test F1:            0.579       0.609
Precision:          49%         54%
Recall:             70%         70%


FILES INCLUDED
--------------

DATASET:
- data/train_clean.csv           : Training data (202,357 samples, 62 features)
- data/test_clean.csv            : Test data (34,160 samples, 62 features)
- feature_columns_clean.txt : List of 62 feature column names

LIGHTGBM MODEL:
- saved_models/lightgbm_clean_model.txt      : Trained LightGBM model (text format)
- metrics/lightgbm_metrics.csv          : Model performance metrics
- saved_models/lightgbm_feature_importance.csv: Feature importance scores

TCN MODEL:
- saved_models/tcn_clean_model.pt            : Trained TCN model (PyTorch format)
- metrics/tcn_metrics.json              : Model performance metrics
- saved_models/tcn_feature_importance.csv    : Permutation importance scores

TRAINING SCRIPTS:
- scripts/train_clean_model.py          : Script to train LightGBM on clean data
- scripts/train_tcn_clean.py            : Script to train TCN on clean data


USAGE
-----

1. Load LightGBM model:
   import lightgbm as lgb
   model = lgb.Booster(model_file='saved_models/lightgbm_clean_model.txt')
   predictions = model.predict(X_test)

2. Load TCN model:
   import torch
   model = TCN(input_size=62, hidden_dim=64, num_layers=3, kernel_size=3, dropout=0.3)
   model.load_state_dict(torch.load('saved_models/tcn_clean_model.pt'))
   model.eval()

3. Feature columns:
   with open('feature_columns_clean.txt', 'r') as f:
       feature_cols = [line.strip() for line in f if line.strip()]


TOP FEATURES
------------
LightGBM (by SHAP importance):
1. curr_trx_days_active
2. curr_vs_recent_ratio
3. curr_trx_count
4. recent_evening_ratio
5. recent_trx_event_types

TCN (by permutation importance):
1. recent_trx_months_active (5.0% AUC drop)
2. curr_vs_recent_ratio (4.5% AUC drop)
3. curr_trx_days_active (1.2% AUC drop)
4. recent_trx_days_active (0.8% AUC drop)
5. recent_trx_event_types (0.5% AUC drop)


REQUIREMENTS
------------
- Python 3.8+
- lightgbm
- torch
- pandas
- numpy
- scikit-learn

================================================================================

