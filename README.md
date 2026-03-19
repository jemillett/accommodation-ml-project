# Predicting Workplace Accommodation Outcomes for Employees with Disabilities

> **ML Modeling & Pipeline Capstone — MVP Version 1**
> Jem Millett | Spring 2026

---

## Project Overview

Employers and HR teams often make accommodation decisions for employees with disabilities based on gut feeling, inconsistent policies, or fear of cost — leading to unnecessary denials and avoidable turnover. This project applies machine learning to predict whether a requested workplace accommodation will result in a **positive employment outcome** (employee retention + accommodation rated effective), using data calibrated to the Job Accommodation Network (JAN) Annual Survey published by the U.S. Department of Labor.

The core insight driving this project: accommodation success isn't random. Features like accommodation type, cost tier, and time to implement carry real predictive signal — and a model can surface that signal in a way that helps HR professionals make more equitable, evidence-based decisions.

---

## CRISP-DM Framework

| Phase | Description |
|---|---|
| 1. Business Understanding | Problem definition, stakeholders, ML justification |
| 2. Data Understanding | EDA, distributions, correlation analysis |
| 3. Data Preparation | Feature engineering, scikit-learn Pipeline construction |
| 4. Modeling | 4 candidate algorithms, cross-validation, hyperparameter tuning |
| 5. Evaluation | F1, ROC-AUC, SHAP feature importance, business metrics |
| 6. Deployment Notes | Pipeline export, integration pathway, known limitations |

---

## Key Results (MVP v1)

| Metric | Score |
|---|---|
| **Final Model** | Tuned Random Forest Classifier |
| **Test F1 Score (Macro)** | 0.60 |
| **Test ROC-AUC** | 0.64 |
| **False Denial Rate** | 7.3% ✅ (target: <10%) |
| **Approval Accuracy** | 82.5% ✅ (target: >80%) |

**Top predictive drivers (SHAP):** accommodation type, cost tier, days to implement, employee tenure, employer size

**Key insight:** Flexible scheduling and policy change/exemption accommodations have the highest predicted positive outcome probabilities (~0.87–0.88). Leave of absence accommodations are the hardest to predict and show the weakest model performance — suggesting this category may require case-by-case review rather than model-guided decisions.

---

## Code Highlights

### scikit-learn Pipeline (no data leakage)
All preprocessing is bundled with the estimator in a single Pipeline, so the model can take raw input records directly:

```python
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('ord', OrdinalEncoder(categories=ordinal_categories),               ordinal_features),
    ('num', StandardScaler(),                                             numeric_features),
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])
```

### Stratified K-Fold Cross-Validation
Preserves class ratio across all 5 folds — critical given the ~79%/21% class imbalance:

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1_macro')
```

### SHAP Feature Importance
Model predictions are explained at the feature level, not just reported as black-box probabilities:

```python
explainer   = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_proc)
shap.summary_plot(shap_values[1], X_test_proc, feature_names=all_feature_names)
```

### Business Metric Translation
Raw model outputs are translated into HR-relevant KPIs:

```python
false_denial_rate = 1 - recall_score(y_test, y_pred, pos_label=1)
approval_accuracy = precision_score(y_test, y_pred, pos_label=1)
```

---

## Repo Structure

```
accommodation-ml-project/
│
├── accommodation_ml_project.ipynb   # Main notebook — fully executed, all outputs visible
│                                    # Sections: Business Understanding → Deployment Notes
│
├── README.md                        # This file
│
└── .gitignore                       # Excludes checkpoints, .pkl files, and local artifacts
```

> **Note:** The trained pipeline is exported to `final_accommodation_model.pkl` (excluded from version control via `.gitignore`). Re-run the notebook to regenerate it.

---

## Dependencies

All packages installable via pip. Python 3.8+ required.

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.23 | Numerical operations, random seed control |
| `pandas` | ≥1.5 | Data manipulation and EDA |
| `matplotlib` | ≥3.6 | Visualizations (bar charts, ROC curves, histograms) |
| `seaborn` | ≥0.12 | Correlation heatmap |
| `scikit-learn` | ≥1.2 | Pipeline, preprocessing, modeling, cross-validation |
| `xgboost` | ≥1.7 | Gradient boosting candidate model |
| `shap` | ≥0.41 | Feature importance and model explainability |
| `joblib` | (bundled with scikit-learn) | Pipeline serialization |
| `jupyter` | any | Running the notebook |

### Install all at once

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap jupyter
```

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/accommodation-ml-project.git
cd accommodation-ml-project

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap jupyter

# 3. Launch the notebook
jupyter notebook accommodation_ml_project.ipynb
```

All cells are pre-executed and outputs are visible — you can read through it without running anything.

---

## Data Note

The dataset is synthetically generated and calibrated to published JAN aggregate statistics (askjan.org), since individual-level case records were not confirmed publicly available at the time of development. The simulation reflects documented JAN distributions: ~49% of accommodations cost $0, ~75%+ positive outcome rate, and disability/accommodation type proportions from the JAN 2023 report. If case-level JAN data is obtained, only the data loading cell requires replacement — all pipeline, modeling, and evaluation code remains valid.

---

## Author

**Jem Millett** | Spring 2026 | Machine Learning Modeling & Pipeline Capstone
