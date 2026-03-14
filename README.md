# Predicting Workplace Accommodation Outcomes for Employees with Disabilities

## Project Overview
A machine learning classification project that predicts whether a workplace accommodation will result in a **positive employment outcome** (employee retention + accommodation effectiveness) for employees with disabilities.

**Dataset:** Synthetically generated dataset calibrated to Job Accommodation Network (JAN) Annual Survey statistics (U.S. Dept. of Labor)

## CRISP-DM Framework
1. Business Understanding — Problem definition, stakeholders, ML justification
2. Data Understanding — EDA, distributions, correlations
3. Data Preparation — Feature engineering, scikit-learn Pipeline
4. Modeling — Logistic Regression, Random Forest, XGBoost, SVM
5. Evaluation — F1, ROC-AUC, SHAP feature importance, business metrics
6. Deployment Notes — Pipeline export, integration pathway

## Key Results (MVP v1)
| Metric | Score |
|---|---|
| Test F1 Score (Macro) | See notebook |
| Test ROC-AUC | See notebook |
| False Denial Rate | See notebook |

## Setup & Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap nbformat jupyter
jupyter notebook accommodation_ml_project.ipynb
```

## Files
- `accommodation_ml_project.ipynb` — Main notebook (fully executed)
- `README.md` — This file

## Author
Jem Millett | Spring 2026
