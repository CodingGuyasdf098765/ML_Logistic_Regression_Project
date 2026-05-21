# Logistic Regression Classifier — Bank Marketing Campaign

> Binary classification pipeline on 41,000+ bank customer records: full EDA with feature engineering, chi-square feature selection, and a Logistic Regression model tuned via GridSearchCV across 1,400 model combinations — achieving 92.2% accuracy.

---

## Problem

A Portuguese bank wants to identify which existing customers are likely to subscribe to a long-term deposit following a phone marketing campaign. Focusing outreach on high-probability customers reduces wasted calls and improves campaign ROI. This is a binary classification problem: did the customer subscribe (`yes`) or not (`no`)?

## Dataset

- **Source:** Bank Marketing Campaign dataset (UCI / 4Geeks)
- **Size:** 41,188 rows × 21 columns → 41,176 after removing 12 duplicates
- **Target:** `y` — subscribed to long-term deposit (yes/no)
- **Class balance:** ~89% no / ~11% yes — heavily imbalanced

**Column types:**

| Type | Columns |
|---|---|
| Numerical | age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed |
| Categorical | job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome, y |

## EDA & Preprocessing Pipeline

| Step | Action |
|---|---|
| Duplicates | 12 removed → 41,176 rows |
| Feature engineering | `pdays=999` (sentinel "never contacted") → binary `was_previously_contacted` column |
| Dropped columns | `default` (near-zero variance, ~99% "no") and `pdays` (replaced by engineered feature) |
| Outlier removal | `duration` capped at 644.5s; `campaign` capped at 6 calls → 35,951 rows |
| Scaling | MinMaxScaler on all feature columns |
| Feature selection | SelectKBest (chi², k=7) — applied to training set only to prevent data leakage |

**Selected features:** `month`, `was_previously_contacted`, `previous`, `poutcome`, `emp.var.rate`, `euribor3m`, `nr.employed`

**Key EDA finding:** `duration` (call length) is the single strongest predictor of subscription — but it constitutes data leakage since it's only known after the call ends. Dropped from the final feature set for a deployment-realistic model.

## Model Results

| Model | Accuracy |
|---|---|
| Baseline (default hyperparameters) | **92.07%** |
| Optimised (GridSearchCV) | **92.16%** |

**GridSearchCV search space:** C (7 values) × penalty (4 options) × solver (5 options) = 140 combinations × 10-fold CV = **1,400 models trained**

**Best hyperparameters:** `C=0.1`, `penalty=l2`, `solver=liblinear`

The accuracy improvement is modest (+0.08%) because the baseline model with default settings was already well-configured for this dataset. The value of grid search here is principled confirmation, not a dramatic lift.

## Key Takeaways

- **Feature engineering beats feature selection:** Converting the sentinel `pdays=999` into a meaningful binary flag captures real information that raw numbers hide.
- **Data leakage is subtle:** `duration` seems like a legitimate feature but knowing call length before making the call is impossible — including it would inflate training accuracy while producing a model that cannot be deployed.
- **High accuracy ≠ good model when classes are imbalanced:** 92% sounds strong, but a naive model predicting "no" for every customer would achieve ~89%. Precision and recall on the minority "yes" class tell the real story.

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Run It Locally

```bash
git clone https://github.com/matthewkane-ml/ML_LogisticRegression_MTK.git
cd ML_LogisticRegression_MTK
pip install -r requirements.txt
jupyter notebook src/app.ipynb
```

The trained model is saved to `models/` via `pickle`.

## What I'd Do Next

- Evaluate on **precision, recall, and F1** for the minority class rather than overall accuracy — the business cost of missing a likely subscriber is much higher than a wasted call
- Try **class_weight="balanced"** or **SMOTE** oversampling to improve recall on the "yes" class
- Build a **calibration curve** to check whether the model's predicted probabilities (not just the binary predictions) are reliable enough to use as a customer ranking score

---

**Author:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [GitHub portfolio](https://github.com/matthewkane-ml)
