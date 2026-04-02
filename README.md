# Predicting Workplace Accommodation Outcomes for Employees with Disabilities

> **Machine Learning Modeling & Pipeline — Final Submission**
> Jem Millett | Spring 2026

---

## BLUF (Bottom Line Up Front)

This project builds a machine learning classification model to predict whether a requested workplace accommodation for an employee with a disability will result in a positive employment outcome — defined as the employee being retained **and** the accommodation rated effective by the employer.

The final model is a **tuned Random Forest classifier** embedded in a full scikit-learn Pipeline that handles feature engineering, preprocessing, and prediction end-to-end. On the held-out test set it achieves a **false denial rate of 6.1%** (well under the 10% target) and an **approval accuracy of 83.3%** (above the 80% target), making it practically deployable as an HR decision-support tool.

The strongest predictive drivers — accommodation type, cost tier, and implementation speed — directly challenge common employer misconceptions: the most successful accommodations tend to be the fastest and cheapest to implement.

---

## Project Overview

Employers frequently make accommodation decisions based on anecdotal experience or unfounded cost concerns, leading to unnecessary denials and avoidable turnover. This project applies ML to surface the patterns that predict accommodation success, giving HR teams an evidence-based foundation for more equitable, consistent decisions.

**Dataset:** Synthetically generated dataset calibrated to Job Accommodation Network (JAN) Annual Survey statistics (U.S. Dept. of Labor — askjan.org). JAN was contacted directly to request individual case records; the simulation reflects published aggregate distributions pending that response. If real data is received, only the data-loading cell requires updating.

---

## Key Results

| Metric | Score |
|---|---|
| **Final Model** | Tuned Random Forest Classifier |
| **Test F1 Score (Macro)** | 0.61 |
| **Test ROC-AUC** | 0.63 |
| **False Denial Rate** | 6.1% ✅ (target: <10%) |
| **Approval Accuracy** | 83.3% ✅ (target: >80%) |

**Top predictive drivers (SHAP):** accommodation type, cost tier, days to implement, employee tenure, employer size, zero_cost flag (engineered)

**Key insight:** Flexible scheduling, remote work, and policy change accommodations achieve 82–90% model accuracy. Leave of absence cases show the lowest performance (~46%) — suggesting these warrant individualised human review rather than model-guided decisions.

---

## CRISP-DM Framework

| Phase | Notebook Section | Description |
|---|---|---|
| Business Understanding | Section 1 | Problem, stakeholders, success metrics, ML justification |
| Data Understanding | Section 2 | Dataset generation, data dictionary, EDA (6 figures), chi-square tests |
| Data Preparation | Section 3 | Custom FeatureEngineer transformer, ColumnTransformer, stratified split |
| Modeling | Section 4 | Dummy baseline → LR → RF → XGB → SVM; 5-fold CV comparison |
| Evaluation | Section 5 | RandomizedSearchCV tuning for RF and XGBoost; model selection |
| (Class Imbalance) | Section 6 | SMOTE via imblearn Pipeline; comparison with class_weight baseline |
| Evaluation | Section 7 | Test set metrics, confusion matrix, ROC, SHAP, business KPIs |
| Deployment | Section 8 | joblib serialization; reload and verify |
| Conclusions | Section 9 | Findings, business recommendations, limitations, next steps |

---

## Repository Structure

```
accommodation-ml-project/
│
├── accommodation_ml_project.ipynb   # Final notebook — fully executed, all outputs visible
│
├── data/
│   └── jan_accommodation_data.csv   # Synthetic dataset (1,500 records, 10 columns)
│
├── images/                          # All EDA and evaluation figures (auto-generated)
│   ├── fig1_class_balance.png
│   ├── fig2_outcome_by_category.png
│   ├── fig3_cost_and_size.png
│   ├── fig4_numeric_distributions.png
│   ├── fig5_numeric_by_outcome.png
│   ├── fig6_correlation_heatmap.png
│   ├── fig7_cv_comparison.png
│   ├── fig8_evaluation.png
│   ├── fig9_feature_importance.png
│   └── fig10_shap_summary.png
│
├── README.md                        # This file
└── .gitignore                       # Excludes .pkl model files and local artifacts
```

> **Note:** `final_accommodation_model.pkl` is excluded from version control (see `.gitignore`).
> Re-run the notebook to regenerate it — all pipeline steps are fully reproducible with `SEED = 42`.

---

## Data Usage Guide

### Loading the dataset

```python
import pandas as pd
df = pd.read_csv('data/jan_accommodation_data.csv')
```

### Column reference

| Column | Type | Description |
|---|---|---|
| `disability_category` | Categorical (6) | Type of disability (Psychiatric, Musculoskeletal, etc.) |
| `accommodation_type` | Categorical (7) | Type of accommodation requested |
| `functional_limitation` | Categorical (7) | Functional area the accommodation addresses |
| `cost_tier` | Ordinal (4) | Cost band: $0 / $1–$500 / $501–$2K / $2K+ |
| `employer_size` | Ordinal (3) | Small (<50) / Mid-size (50–499) / Large (500+) |
| `industry` | Categorical (7) | Employer industry sector |
| `employee_tenure_years` | Numeric | Years of tenure at request time |
| `days_to_implement` | Numeric | Days taken to implement the accommodation |
| `prior_requests` | Numeric (int) | Number of prior accommodation requests (0–5) |
| `outcome_positive` | Binary (target) | 1 = retained + effective / 0 = separated or ineffective |

### Calibration sources
- ~49% zero-cost accommodations (JAN 2023 Costs & Benefits Report)
- ~75% overall positive outcome rate (JAN employer surveys)
- Disability category and accommodation type proportions from JAN Annual Accommodation Series

---

## Code Highlights

### Custom Feature Engineering (inside the Pipeline)

```python
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds quick_implementation, high_tenure, and zero_cost binary features."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['quick_implementation'] = (X['days_to_implement'] < 14).astype(int)
        X['high_tenure']          = (X['employee_tenure_years'] > 5).astype(int)
        X['zero_cost']            = (X['cost_tier'] == '$0 (No Cost)').astype(int)
        return X
```

### Full scikit-learn Pipeline (leak-proof)

```python
pipeline = Pipeline([
    ('feature_engineer', FeatureEngineer()),           # Domain-driven feature engineering
    ('preprocessor',     ColumnTransformer([           # OHE + Ordinal + StandardScaler
        ('cat', OneHotEncoder(...), categorical_features),
        ('ord', OrdinalEncoder(...), ordinal_features),
        ('num', StandardScaler(),   numeric_features),
    ])),
    ('classifier', RandomForestClassifier(class_weight='balanced', ...)),
])
```

### SMOTE with imbalanced-learn Pipeline

```python
from imblearn.pipeline import Pipeline as ImbPipeline
smote_pipeline = ImbPipeline([
    ('feature_engineer', FeatureEngineer()),
    ('preprocessor',     preprocessor),
    ('smote',            SMOTE(random_state=42)),     # Applied only inside CV folds
    ('classifier',       RandomForestClassifier(...)),
])
```

### SHAP Explainability

```python
explainer   = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_proc)
shap.summary_plot(shap_values[1], X_test_proc, feature_names=all_feature_names)
```

---

## Dependencies

Python 3.8+ required.

| Package | Purpose |
|---|---|
| `numpy` | Numerical operations, random seed control |
| `pandas` | Data manipulation and EDA |
| `matplotlib` | Visualizations |
| `seaborn` | Correlation heatmap |
| `scipy` | Chi-square statistical tests |
| `scikit-learn` | Pipeline, preprocessing, modeling, CV, evaluation |
| `xgboost` | Gradient boosting candidate model |
| `imbalanced-learn` | SMOTE for class imbalance handling |
| `shap` | Feature importance and model explainability |
| `joblib` | Pipeline serialization (bundled with scikit-learn) |
| `jupyter` | Notebook environment |

### Install all at once

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost imbalanced-learn shap jupyter
```

### Run the notebook

```bash
git clone https://github.com/jemillett/accommodation-ml-project.git
cd accommodation-ml-project
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost imbalanced-learn shap jupyter
jupyter notebook accommodation_ml_project.ipynb
```

All cells are pre-executed — outputs are visible without running anything.

---

*Submitted for: Machine Learning Modeling & Pipeline Capstone | Spring 2026*
