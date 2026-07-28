# E-Commerce Transaction Fraud Detection (Cost-Weighted)

## Project Overview

This project builds an interpretable, cost-aware fraud detection system using the IEEE-CIS Fraud Detection dataset (Kaggle). The system outputs a ranked investigation queue with capacity-aware precision/recall metrics and a cost-weighted threshold optimizer.

**Dataset:** IEEE-CIS Fraud Detection — `train_transaction.csv` (590,540 rows × 394 cols) + `train_identity.csv` (590,540 rows × 41 cols), merged on `TransactionID`. Target: `isFraud` (3.5% positive class).

**Model:** Random Forest (200 trees, max_depth=12, class_weight='balanced'), trained on a 20k stratified subsample due to local memory constraints (4 GB RAM).

---

## Model Performance — Random Forest (Selected Model)

All numbers below computed on the **full 118k test set** (`test_rf_split.parquet`).

### Cost-Optimal Threshold (from `rf_cost_analysis.parquet`)

| Metric | Value |
|--------|-------|
| **Cost-optimal threshold** | 0.07 |
| **Cost at that threshold** | $300,350 |
| **Cost at default 0.5 threshold** | $561,700 |
| **Savings vs default** | 46.5% |
| **Recall at cost-optimal threshold** | 84.6% |
| **Precision at cost-optimal threshold** | 7.1% |

The cost matrix used: **FN = $2,000, FP = $50** (assumed business figure, not fitted from data).

### Investigator Capacity Metrics (Dashboard, 10k Held-Out Sample)

The live dashboard evaluates a fixed investigator-capacity constraint on the **10k working test sample** (`test_working_rf.parquet`), which is a different metric than the cost-minimizing threshold above.

| Capacity (k) | True Positives Caught | Precision | Recall |
|--------------|----------------------|-----------|--------|
| 300 | 112 | 37.3% | 32.0% |

This reflects a fixed-capacity investigation queue, not the cost-optimal threshold.

---

## Logistic Regression (Rejected)

The LR artifacts (lr_model_v2.pkl, lr_feature_cols.txt, lr_v2_cost_analysis.parquet, and the LR train/test parquet files) are kept in the repository for transparency and reproducibility only, and are not intended to be reused or deployed.

Logistic Regression was trained and evaluated on the same data. Its best-case cost (threshold 0.37) was **$3,765,800** — roughly 7× worse than Random Forest — so it was rejected in favor of RF.

The original `lr_model.pkl` was discarded because it had no recoverable feature list and could no longer be run. It was replaced by `lr_model_v2.pkl` with feature names properly saved (722 features after dropping 34 constant columns).

---

## Limitations

1. **Training data subsample:** Models were trained on a 20k stratified subsample due to local memory constraints (4 GB RAM), not the full 472k training set.
2. **Assumed cost matrix:** The cost matrix (FN=$2,000, FP=$50) is an assumed business figure, not a fitted one.
3. **Low precision:** The majority of flagged transactions are false positives (precision 7.1% at cost-optimal threshold).
4. **No temporal validation:** Train/test split is random stratified, not time-based. Real fraud patterns drift; this overstates performance.
5. **No calibration:** Predicted probabilities are not isotonic/Platt calibrated. Cost optimization assumes well-calibrated scores.
6. **Small-sample artifacts:** Precision@k on the 10k working test shows high variance at low k (e.g., k=10).
7. **Single seed:** All splits use `random_state=42`. No confidence intervals or bootstrap variance estimates.

---

## Repository Structure

```
Data/ieee-cis/
├── train_engineered.parquet        # Full engineered dataset (590k × 386)
├── train_rf_split.parquet          # RF train split (472k × 385)
├── test_rf_split.parquet           # RF test split (118k × 385)
├── train_working_rf.parquet        # RF working train (30k × 385)
├── test_working_rf.parquet         # RF working test (10k × 385)
├── train_lr_split.parquet          # LR train split (472k × 758)
├── test_lr_split.parquet           # LR test split (118k × 758)
├── train_working_lr.parquet        # LR working train (30k × 758)
├── test_working_lr.parquet         # LR working test (10k × 758)
├── rf_model.pkl                    # Trained Random Forest (20k sample)
├── lr_model_v2.pkl                 # Retrained Logistic Regression (20k sample)
├── lr_feature_cols.txt             # Feature list for LR model (722 cols)
├── rf_cost_analysis.parquet        # Cost curve for RF (99 thresholds)
├── lr_v2_cost_analysis.parquet     # Cost curve for LR v2 (99 thresholds)
├── dashboard.py                    # Streamlit dashboard
└── README.md
```

---

## Running the Dashboard

```bash
cd Data/ieee-cis
streamlit run dashboard.py
```

The dashboard loads `rf_model.pkl` and `test_working_rf.parquet`, scores the 10k held-out sample, and provides:

1. **Ranked Fraud Queue** — Transactions sorted by fraud probability with actual labels
2. **Investigator Capacity Slider** — Live precision/recall at selected capacity (10–2000)
3. **Cost Curve Chart** — Expected cost vs threshold from `rf_cost_analysis.parquet`, with optimal threshold marked and savings vs 0.5 shown