# 📉 ML Customer Churn Prediction

> **Predict who's leaving — before they do.**  
> A machine learning pipeline that identifies at-risk customers using gradient boosting models (CatBoost) and explains *why* they're predicted to churn using SHAP (SHapley Additive exPlanations).

---

## 🧭 Table of Contents

- [Business Overview](#-business-overview)
- [Technical Overview](#-technical-overview)
- [Project Structure](#-project-structure)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Models Used](#-models-used)
- [Model Interpretability — SHAP](#-model-interpretability--shap)
- [Key Business Insights](#-key-business-insights)
- [How to Run](#-how-to-run)
- [Dependencies](#-dependencies)
- [License](#-license)

---

## 💼 Business Overview

**Customer churn** — when a customer stops doing business with you — is one of the most costly problems in subscription-based industries (telecom, SaaS, banking, streaming). Acquiring a new customer typically costs **5–7× more** than retaining an existing one.

This project solves a critical business question:

> *"Which of our current customers are most likely to leave in the near future, and what is driving that behavior?"*

### What This Model Delivers

| Business Need | What the Model Provides |
|---|---|
| Proactive retention | Ranked list of at-risk customers before they churn |
| Root-cause understanding | Per-customer explanation of churn drivers (SHAP) |
| Resource prioritisation | Focus retention spend on high-risk, high-value segments |
| Strategy input | Identifies product/service factors linked to churn at scale |

### Who Should Use This

- **Customer Success / Retention teams** — act on SHAP-explained risk scores per customer
- **Marketing** — personalise outreach to high-risk cohorts
- **Product** — identify service pain points correlated with churn
- **Executives / Analysts** — understand portfolio-level churn risk

---

## 🔬 Technical Overview

This is an end-to-end supervised binary classification pipeline built in Python. The target variable is whether a customer **churned (1)** or **stayed (0)**.

**Core technical components:**

- **Data Processing** — cleaning, encoding, feature engineering
- **Model Training** — multiple classifiers including CatBoost
- **Hyperparameter Tuning** — fine-tuning for optimal generalisation
- **Evaluation** — metrics suited to imbalanced classification (AUC-ROC, F1, Precision/Recall)
- **Interpretability** — global and local SHAP analysis for model explainability

**Tech stack:** `Python` · `CatBoost` · `scikit-learn` · `SHAP` · `pandas` · `numpy`

---

## 📁 Project Structure

```
ML_Customer-Churn-Prediction/
│
├── run.py
│   ├── EDA.py                 # Learn data characteristics
│   ├── normalization_encoding.py            # Normalization to prevent bias and skewed figures and encoding for catergorical data
│   └── train_test_split.py            # Define training and testing datasets                    
│
├── data_processing/              # Data ingestion, cleaning & feature engineering
│   └── ...
│
├── model/                        # Model training, tuning, and evaluation
│   ├── models.py                 # Model definitions and training logic
│   ├── fine_tuning.py            # Hyperparameter optimisation
│   └── evaluation.py            # Metrics: AUC, F1, Precision, Recall, etc.
│
├── SHAP_interpretation/          # Explainability layer — global & local SHAP analysis
│   └── SHAP.py
│
├── catboost_info/                # CatBoost training logs and metadata
│
├── output_storage/              # Saved models, predictions, and SHAP outputs
│
├── README.md
└── LICENSE
```

---

## 🔄 Pipeline Walkthrough

```
Raw Customer Data
       │
       ▼
┌─────────────────────┐
│   data_processing   │  ← Clean nulls, encode categoricals,
│                     │    engineer features (tenure buckets,
└─────────────────────┘    contract type flags, charge ratios)
       │
       ▼
┌─────────────────────┐
│   model/models.py   │  ← Train candidate models
│                     │    (CatBoost, others)
└─────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  model/fine_tuning.py   │  ← Hyperparameter search
│                         │    (grid/random/Bayesian search)
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  model/evaluation.py    │  ← AUC-ROC, F1, Precision,
│                         │    Recall, Confusion Matrix
└─────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  SHAP_interpretation/SHAP   │  ← Global feature importance
│                             │    + per-customer explanations
└─────────────────────────────┘
       │
       ▼
  output_storage/
  (predictions + SHAP plots + model artefacts)
```

---

## 🤖 Models Used

### CatBoost (Primary)

CatBoost is a gradient boosting algorithm developed by Yandex, particularly effective for:

- **Categorical features** — handles them natively without manual encoding
- **Imbalanced datasets** — robust performance without heavy preprocessing
- **SHAP integration** — first-class support for fast, exact SHAP value computation

```python
# Simplified example of what model training looks like
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    eval_metric='AUC',
    early_stopping_rounds=50,
    verbose=100
)
model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_val, y_val))
```

### Evaluation Metrics

| Metric | Why It Matters for Churn |
|---|---|
| **AUC-ROC** | Measures overall ranking power — how well the model separates churners from stayers |
| **Precision** | Of those predicted to churn, how many actually did? (Cost of false alarms) |
| **Recall** | Of those who actually churned, how many did we catch? (Cost of missing churners) |
| **F1 Score** | Harmonic mean of Precision & Recall — balanced performance measure |
| **Confusion Matrix** | Full breakdown of TP, FP, TN, FN predictions |

> **Business note:** In churn problems, **Recall is often prioritised** — it's worse to miss a churner than to flag a loyal customer unnecessarily.

---

## 🔍 Model Interpretability — SHAP

The `SHAP_interpretation/` module provides the "why" behind every prediction.

### What SHAP Does

SHAP assigns each feature a contribution score to the model's output for a specific prediction. Based on Shapley values from cooperative game theory, it is mathematically consistent and model-agnostic.

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test)

# Individual customer explanation
shap.waterfall_plot(shap.Explanation(
    values=shap_values[i],
    base_values=explainer.expected_value,
    data=X_test.iloc[i]
))
```

### Types of SHAP Analysis in This Project

| Analysis Type | Plot | Business Use |
|---|---|---|
| **Global importance** | Beeswarm / bar chart | Which features drive churn most across all customers? |
| **Local explanation** | Waterfall / force plot | Why is *this specific customer* flagged as at risk? |
| **Feature interaction** | Dependency plot | How do two features interact (e.g. Tenure × Contract type)? |

### Example Insight (Typical Churn Patterns)

```
Customer #4821 — Churn Probability: 87%

Base rate:           +0.23  (population avg churn rate)
Contract: Month-to-Month  +0.31  ↑ increases churn risk
Tenure: 2 months     +0.18  ↑ new customer, high risk
MonthlyCharges: $89  +0.12  ↑ high bill relative to usage
Has TechSupport: No  +0.09  ↑ no support = frustration risk
─────────────────────────────
Predicted probability: 0.87
```

> **Business action:** Contact this customer with a contract upgrade offer or loyalty discount before next billing cycle.

---

## 📈 Key Business Insights

Based on typical findings in churn models of this type:

1. **Contract type is the strongest signal** — month-to-month customers churn at dramatically higher rates than annual/biennial contract holders

2. **Early tenure is the danger zone** — customers in their first 3–6 months are disproportionately likely to leave; early onboarding interventions are high-ROI

3. **High monthly charges without perceived value** — customers paying more without premium services (e.g. no Tech Support, no Online Security) show elevated churn

4. **Service quality gaps drive churn** — lack of support services is consistently associated with higher churn probability regardless of spend level

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Grace-VN/ML_Customer-Churn-Prediction.git
cd ML_Customer-Churn-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install catboost scikit-learn shap pandas numpy matplotlib
```

### 3. Run the full pipeline

```bash
python run.py
```

The entry point (`run.py`) orchestrates the full flow:

```python
# run.py
from model import models, fine_tuning, evaluation
from SHAP_interpretation import SHAP
```

Outputs (predictions, model files, SHAP plots) are saved to `output_storage/`.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `catboost` | Primary gradient boosting model |
| `scikit-learn` | Preprocessing, model selection, evaluation metrics |
| `shap` | Model interpretability and feature attribution |
| `pandas` | Data manipulation and feature engineering |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualisation of SHAP plots and evaluation curves |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

**Grace-VN** · [GitHub Profile](https://github.com/Grace-VN)

---

*Built to turn raw customer data into actionable retention intelligence.*
