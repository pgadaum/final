# Telco Customer Churn Prediction

**Authors:** Pawan Gadaum, Sami Kebebe  
**Course:** Data Science Capstone

---

## Overview

This project builds a machine learning pipeline to predict customer churn for a telecom provider using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle. The goal is to identify customers at high risk of leaving, uncover the key drivers of churn, and deliver actionable business recommendations.

---

## Dataset

| Attribute | Details |
|---|---|
| **Source** | Kaggle — `blastchar/telco-customer-churn` |
| **File** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Target** | `Churn` (Yes / No) |
| **Features** | Demographics, account info, services subscribed |
| **Size** | ~7,043 rows × 21 columns (after cleaning) |

Key feature groups:
- **Demographics:** Gender, Senior Citizen, Partner, Dependents
- **Account info:** Tenure, Contract type, Payment method, Monthly & Total charges
- **Services:** Phone, Internet, Online Security, Tech Support, Streaming

---

## Project Structure

```
ProjectChurn_finalCoding.ipynb   # Main notebook (analysis + models)
churn_risk_table_test.csv        # Full test-set risk scores (output)
top10_at_risk_customers.csv      # Top 10 highest-risk customers (output)
telco_churn_with_predictions.csv # Full dataset with predicted labels (output)
README.md
```

---

## Pipeline

```
Raw Data
   └── Data Cleaning (TotalCharges coercion, drop NaNs)
       └── EDA (churn distribution, contract, charges, tenure)
           └── Preprocessing (OneHotEncoder + StandardScaler via ColumnTransformer)
               └── Modeling
                   ├── Logistic Regression (baseline)
                   ├── Random Forest
                   ├── XGBoost
                   └── CatBoost
                       └── Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
                           └── Risk Table → Top 10 At-Risk Customers → CSV Outputs
```

---

## Models

| Model | Notes |
|---|---|
| **Logistic Regression** | Baseline; interpretable via coefficients; `class_weight='balanced'` |
| **Random Forest** | Tree ensemble; `class_weight='balanced_subsample'`, 300 trees |
| **XGBoost** | Gradient boosting; 500 estimators, `learning_rate=0.05` |
| **CatBoost** | Gradient boosting; handles categoricals well out of the box |

All models are wrapped in `sklearn` `Pipeline` objects to ensure consistent preprocessing across train and test sets.

---

## Evaluation Metrics

Given the class imbalance (fewer churners than non-churners), the project prioritizes:

- **Recall** — catching as many actual churners as possible
- **F1-score** — balance between precision and recall
- **ROC-AUC** — overall discriminative power

Confusion matrices and ROC curves are plotted for each model.

---

## Key Findings

| Driver | Direction |
|---|---|
| Month-to-month contract | ↑ Churn risk |
| Fiber optic internet | ↑ Churn risk |
| High monthly charges | ↑ Churn risk |
| No Online Security / Tech Support | ↑ Churn risk |
| Longer tenure | ↓ Churn risk (protective) |
| 1- or 2-year contracts | ↓ Churn risk (protective) |

---

## Business Recommendations

1. **Contract migration** — Incentivize month-to-month customers with loyalty pricing or bundles to move to annual contracts.
2. **Early-life interventions** — Focus retention efforts on customers in their first 3–6 months (churn risk is highest at low tenure).
3. **Price/plan optimization** — Offer targeted discounts or right-sized plans to high-charge segments showing churn signals.
4. **Support add-ons** — Promote Online Security and Tech Support to at-risk cohorts; these services correlate with lower churn.
5. **Targeted outreach** — Use model probabilities to contact the **top decile** of at-risk customers each month and measure uplift.

---

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost catboost kagglehub
```

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualizations |
| `scikit-learn` | Preprocessing, modeling, evaluation |
| `imbalanced-learn` | SMOTE / imbalanced pipeline support |
| `xgboost` | XGBoost classifier |
| `catboost` | CatBoost classifier |
| `kagglehub` | Dataset download from Kaggle |

---

## How to Run

1. Clone or download this repository.
2. Install dependencies (see above).
3. Open `ProjectChurn_finalCoding.ipynb` in Jupyter or VS Code.
4. Run all cells top to bottom. The dataset is auto-downloaded via `kagglehub` on first run.
5. Output CSV files will be saved to the working directory.

> **Note:** If you change the data path or environment, only the data loading cell needs to be updated — the rest of the pipeline is self-contained.

---

## Reproducibility

Random seeds are fixed (`random_state=42`) throughout to ensure consistent results across runs. Train/test split uses stratification on the `Churn` label to preserve class proportions.
