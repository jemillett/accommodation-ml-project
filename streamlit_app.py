"""
Workplace Accommodation Outcome Predictor
==========================================
Interactive dashboard for the ML capstone project:
"Predicting Workplace Accommodation Outcomes for Employees with Disabilities"

Author: Jem Millett | Spring 2026
Data:   Synthetic dataset calibrated to JAN Annual Survey statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, f1_score, precision_score, recall_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Accommodation Outcome Predictor",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEED = 42
OPTIMAL_THRESHOLD = 0.54

# ═══════════════════════════════════════════════════════
# DATA & MODEL (cached so they only run once)
# ═══════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df = pd.read_csv("data/jan_accommodation_data.csv")
    return df

@st.cache_resource
def train_model(df):
    """Train the full sklearn Pipeline and return model + split data."""

    class FeatureEngineer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): return self
        def transform(self, X):
            X = X.copy()
            X['quick_implementation'] = (X['days_to_implement'] < 14).astype(int)
            X['high_tenure']          = (X['employee_tenure_years'] > 5).astype(int)
            X['zero_cost']            = (X['cost_tier'] == '$0 (No Cost)').astype(int)
            return X

    categorical_features = ['disability_category', 'accommodation_type',
                             'functional_limitation', 'industry']
    ordinal_features     = ['cost_tier', 'employer_size']
    ordinal_categories   = [
        ['$0 (No Cost)', '$1–$500', '$501–$2,000', '$2,001+'],
        ['Small (<50)', 'Mid-size (50–499)', 'Large (500+)'],
    ]
    numeric_features = ['employee_tenure_years', 'days_to_implement', 'prior_requests',
                        'quick_implementation', 'high_tenure', 'zero_cost']

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('ord', OrdinalEncoder(categories=ordinal_categories),               ordinal_features),
        ('num', StandardScaler(),                                             numeric_features),
    ], remainder='drop')

    pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor',     preprocessor),
        ('classifier',       RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=2,
            max_features='sqrt', class_weight='balanced', random_state=SEED
        )),
    ])

    X = df.drop(columns=['outcome_positive'])
    y = df['outcome_positive']
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=SEED, stratify=y_temp)

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= OPTIMAL_THRESHOLD).astype(int)

    metrics = {
        'f1_macro':      f1_score(y_test, y_pred, average='macro'),
        'roc_auc':       roc_auc_score(y_test, y_prob),
        'fdr':           1 - recall_score(y_test, y_pred, pos_label=1),
        'approval_acc':  precision_score(y_test, y_pred, pos_label=1),
        'recall_neg':    recall_score(y_test, y_pred, pos_label=0),
        'accuracy':      (y_pred == y_test.values).mean(),
    }
    return pipeline, X_test, y_test, y_prob, y_pred, metrics

# ── Load ─────────────────────────────────────────────────────────
df = load_data()
pipeline, X_test, y_test, y_prob, y_pred, metrics = train_model(df)

COST_ORDER = ['$0 (No Cost)', '$1–$500', '$501–$2,000', '$2,001+']
SIZE_ORDER  = ['Small (<50)', 'Mid-size (50–499)', 'Large (500+)']

# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/International_Symbol_of_Access.svg/240px-International_Symbol_of_Access.svg.png", width=60)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["📊 Overview", "🔍 EDA", "🤖 Model Performance", "🎯 Predict Outcome"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project:** ML Capstone — Spring 2026
**Author:** Jem Millett
**Model:** Tuned Random Forest
**Data:** JAN Survey (synthetic)
[GitHub Repo](https://github.com/jemillett/accommodation-ml-project)
""")

# ═══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("♿ Workplace Accommodation Outcome Predictor")
    st.markdown("""
    This dashboard presents findings from a machine learning project that predicts whether a
    requested workplace accommodation for an employee with a disability will result in a
    **positive employment outcome** — employee retained and accommodation rated effective.

    > **Data note:** Dataset is synthetically generated and calibrated to
    > [JAN Annual Survey](https://askjan.org/topics/costs.cfm) aggregate statistics.
    > JAN was contacted directly for individual case records.
    """)

    st.markdown("### Key Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1 Score (Macro)", f"{metrics['f1_macro']:.3f}", help="Balance of precision and recall across both classes")
    c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", help="Model discrimination ability")
    c3.metric("False Denial Rate", f"{metrics['fdr']:.1%}", delta=f"Target <10% {'✅' if metrics['fdr'] < 0.10 else '❌'}", delta_color="inverse")
    c4.metric("Approval Accuracy", f"{metrics['approval_acc']:.1%}", delta=f"Target >80% {'✅' if metrics['approval_acc'] > 0.80 else '❌'}")

    st.markdown("---")
    st.markdown("### Key Business Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ **Low-cost accommodations succeed most often.** ~49% of accommodations cost $0 — and these are among the highest-performing categories. Cost is not a reliable predictor of poor outcomes.")
        st.info("📅 **Speed matters.** Accommodations implemented in under 14 days have meaningfully higher predicted success probabilities. HR teams should prioritise fast approvals.")
    with col2:
        st.warning("⚠️ **Leave of Absence is the hardest to predict.** This category shows the lowest model accuracy (~46%) — these cases warrant individualised human review rather than model-guided decisions.")
        st.info("🏢 **Larger employers show modestly better outcomes.** Likely reflecting more established HR infrastructure and accommodation processes.")

    st.markdown("---")
    st.markdown("### About This Project")
    st.markdown("""
    | Item | Detail |
    |---|---|
    | **Algorithm** | Tuned Random Forest Classifier |
    | **Pipeline** | FeatureEngineer → ColumnTransformer → RandomForest |
    | **Class imbalance** | Addressed via `class_weight='balanced'` + threshold tuning (t=0.54) |
    | **Dataset** | 1,500 synthetic records, 9 raw features + 3 engineered |
    | **Framework** | CRISP-DM (Business Understanding → Deployment) |
    """)

# ═══════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    overall_rate = df['outcome_positive'].mean()

    tab1, tab2, tab3 = st.tabs(["Outcome by Category", "Cost & Employer Size", "Numeric Features"])

    with tab1:
        st.markdown("#### Positive Outcome Rate by Feature")
        col_choice = st.selectbox("Select feature", ['accommodation_type', 'disability_category',
                                                       'functional_limitation', 'industry'])
        rates = df.groupby(col_choice)['outcome_positive'].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(4, len(rates) * 0.55)))
        colors = ['#2E75B6' if v >= overall_rate else '#ED7D31' for v in rates.values]
        ax.barh(rates.index, rates.values, color=colors, edgecolor='white')
        ax.axvline(overall_rate, color='gray', linestyle='--', linewidth=1.5,
                   label=f'Overall avg ({overall_rate:.0%})')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_xlabel('Positive Outcome Rate')
        ax.set_title(f'Outcome Rate by {col_choice.replace("_", " ").title()}', fontweight='bold')
        ax.legend()
        st.pyplot(fig)
        plt.close()

        n_records = df.groupby(col_choice)['outcome_positive'].count()
        st.dataframe(
            pd.DataFrame({'Positive Rate': rates.map('{:.1%}'.format),
                          'Record Count': n_records}).sort_values('Positive Rate', ascending=False),
            use_container_width=True
        )

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Accommodation Cost Distribution")
            cost_counts = df['cost_tier'].value_counts().reindex(COST_ORDER)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(COST_ORDER, cost_counts.values, color='#2E75B6', alpha=0.85, edgecolor='white')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=20)
            ax.set_title('Cost Tier Distribution', fontweight='bold')
            st.pyplot(fig); plt.close()

        with c2:
            st.markdown("#### Outcome Rate by Cost Tier")
            cost_rates = df.groupby('cost_tier')['outcome_positive'].mean().reindex(COST_ORDER)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(COST_ORDER, cost_rates.values, color='#70AD47', alpha=0.85, edgecolor='white')
            ax.axhline(overall_rate, color='gray', linestyle='--', linewidth=1.3)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            ax.tick_params(axis='x', rotation=20)
            ax.set_title('Outcome Rate by Cost Tier', fontweight='bold')
            st.pyplot(fig); plt.close()

        st.markdown("#### Outcome Rate by Employer Size")
        size_rates = df.groupby('employer_size')['outcome_positive'].mean().reindex(SIZE_ORDER)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(SIZE_ORDER, size_rates.values, color='#ED7D31', alpha=0.85, edgecolor='white')
        ax.axhline(overall_rate, color='gray', linestyle='--', linewidth=1.3, label='Overall avg')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.legend(); ax.set_title('Outcome Rate by Employer Size', fontweight='bold')
        st.pyplot(fig); plt.close()

    with tab3:
        st.markdown("#### Numeric Feature Distributions by Outcome")
        num_col = st.selectbox("Select numeric feature",
                               ['employee_tenure_years', 'days_to_implement', 'prior_requests'])
        fig, ax = plt.subplots(figsize=(9, 4))
        for outcome, label, color in [(1, 'Positive', '#2E75B6'), (0, 'Negative', '#ED7D31')]:
            ax.hist(df[df['outcome_positive'] == outcome][num_col],
                    bins=25, alpha=0.6, color=color, label=label,
                    edgecolor='white', density=True)
        ax.set_xlabel(num_col.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'{num_col.replace("_", " ").title()} by Outcome', fontweight='bold')
        ax.legend()
        st.pyplot(fig); plt.close()

        st.markdown("#### Summary Statistics")
        st.dataframe(df.groupby('outcome_positive')[num_col].describe().T
                     .rename(columns={0: 'Negative', 1: 'Positive'}),
                     use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")
        cm  = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive']).plot(
            ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'Confusion Matrix (threshold={OPTIMAL_THRESHOLD})', fontweight='bold')
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='#2E75B6', lw=2.5, label=f'Model (AUC={metrics["roc_auc"]:.3f})')
        ax.plot([0,1],[0,1], color='#BDC3C7', linestyle='--', lw=1.5, label='Random')
        ax.fill_between(fpr, tpr, alpha=0.08, color='#2E75B6')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontweight='bold'); ax.legend()
        st.pyplot(fig); plt.close()

    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred,
                                   target_names=['Negative (0)', 'Positive (1)'],
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.markdown("#### Accuracy by Accommodation Type")
    res = X_test.copy()
    res['actual'] = y_test.values; res['predicted'] = y_pred
    accom_acc = (res.groupby('accommodation_type')
                 .apply(lambda g: (g['actual'] == g['predicted']).mean())
                 .sort_values(ascending=True))
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['#2E75B6' if v >= 0.75 else '#ED7D31' for v in accom_acc.values]
    ax.barh(accom_acc.index, accom_acc.values, color=colors, edgecolor='white')
    ax.axvline(0.75, color='gray', linestyle='--', linewidth=1.3, label='75% reference')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title('Model Accuracy by Accommodation Type', fontweight='bold')
    ax.legend(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════
# PAGE 4 — PREDICT
# ═══════════════════════════════════════════════════════
elif page == "🎯 Predict Outcome":
    st.title("🎯 Predict Accommodation Outcome")
    st.markdown("""
    Enter the details of a workplace accommodation request below.
    The model will return a **predicted probability** of a positive outcome
    (employee retained + accommodation effective).

    > ⚠️ This tool is a proof-of-concept trained on synthetic data.
    > Do not use for real HR decisions without validation on actual case data.
    """)

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            disability = st.selectbox("Disability Category", [
                'Psychiatric', 'Musculoskeletal', 'Neurological',
                'Sensory (Vision/Hearing)', 'Chronic Illness', 'Cognitive'])

            accommodation = st.selectbox("Accommodation Type", [
                'Flexible Scheduling', 'Remote Work / Telework',
                'Assistive Technology', 'Physical Modification',
                'Leave of Absence', 'Policy Change / Exemption', 'Ergonomic Equipment'])

            func_limit = st.selectbox("Functional Limitation", [
                'Concentration/Focus', 'Mobility', 'Communication',
                'Stamina/Endurance', 'Dexterity', 'Vision', 'Hearing'])

            industry = st.selectbox("Industry", [
                'Healthcare', 'Education', 'Retail/Hospitality',
                'Manufacturing', 'Finance/Insurance',
                'Government/Public Sector', 'Technology'])

        with c2:
            cost_tier  = st.selectbox("Accommodation Cost", COST_ORDER)
            emp_size   = st.selectbox("Employer Size", SIZE_ORDER)
            tenure     = st.slider("Employee Tenure (years)", 0.5, 30.0, 4.0, 0.5)
            impl_days  = st.slider("Days to Implement", 1, 180, 20)
            prior_req  = st.slider("Prior Accommodation Requests", 0, 5, 0)

        submitted = st.form_submit_button("Predict Outcome", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            'disability_category':   disability,
            'accommodation_type':    accommodation,
            'functional_limitation': func_limit,
            'cost_tier':             cost_tier,
            'employer_size':         emp_size,
            'industry':              industry,
            'employee_tenure_years': tenure,
            'days_to_implement':     impl_days,
            'prior_requests':        prior_req,
        }])

        prob = pipeline.predict_proba(input_df)[0][1]
        pred = int(prob >= OPTIMAL_THRESHOLD)

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"### Predicted Positive Outcome Probability")
            st.progress(float(prob))
            st.markdown(f"<h1 style='text-align:center; color:{'#2E75B6' if prob >= 0.5 else '#ED7D31'}'>"
                        f"{prob:.1%}</h1>", unsafe_allow_html=True)

            if pred == 1:
                st.success(f"✅ **Predicted: Positive Outcome** (above threshold {OPTIMAL_THRESHOLD})")
                st.markdown("The model predicts this accommodation is likely to result in employee retention and be rated effective.")
            else:
                st.warning(f"⚠️ **Predicted: Negative Outcome** (below threshold {OPTIMAL_THRESHOLD})")
                st.markdown("The model predicts this accommodation may not result in a positive outcome. Consider additional review or alternative accommodation types.")

        st.markdown("#### How this compares to similar cases:")
        similar = df[
            (df['accommodation_type'] == accommodation) &
            (df['disability_category'] == disability)
        ]
        if len(similar) > 5:
            rate = similar['outcome_positive'].mean()
            st.info(f"In the dataset, **{accommodation}** accommodations for **{disability}** "
                    f"have a historical positive outcome rate of **{rate:.0%}** "
                    f"(n={len(similar)} cases).")
        else:
            rate = df[df['accommodation_type'] == accommodation]['outcome_positive'].mean()
            st.info(f"**{accommodation}** accommodations overall have a positive outcome rate of **{rate:.0%}**.")
