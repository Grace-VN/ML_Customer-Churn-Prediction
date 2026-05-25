from sklearn.metrics import classification_report, roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from model.fine_tuning import best_params
from model.models import baseline_models, preprocessor, X_temp, y_temp, X_test, y_test, skf
from catboost import CatBoostClassifier
from data_processing.train_test_split import X_temp, y_temp, X_test, y_test
from sklearn.pipeline import Pipeline

# ==============================
# Output Paths
# ==============================
CSV_DIR   = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\output_storage\csv_files"
IMAGE_DIR = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\output_storage\images"
os.makedirs(CSV_DIR,   exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ==============================
# Final Model Selection
# ==============================
try:
    from model.fine_tuning import best_params
    print(">>> Using Optimized Hyperparameters from Optuna Tuning")
    final_model = CatBoostClassifier(
        **best_params,
        random_state=42,
        verbose=0,
        eval_metric='AUC'
    )
except (ImportError, AttributeError, NameError):
    print(">>> Optuna tuning skipped. Using Baseline CatBoost parameters")
    final_model = baseline_models['CatBoost']

# ==============================
# Final Fit on Full Temp Set
# ==============================
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', final_model)
])

final_pipeline.fit(X_temp, y_temp)

# ==============================
# Predictions on Held-Out Test Set
# ==============================
probs = final_pipeline.predict_proba(X_test)[:, 1]
preds = final_pipeline.predict(X_test)

# ==============================
# Evaluation
# ==============================
final_auc = roc_auc_score(y_test, probs)
print(f"\nFinal Test AUC-ROC: {final_auc:.5f}")

# --- Classification Report → CSV ---
report_dict = classification_report(y_test, preds, output_dict=True)
report_df   = pd.DataFrame(report_dict).transpose().round(4)
report_path = os.path.join(CSV_DIR, "classification_report.csv")
report_df.to_csv(report_path)
print(f"\nClassification Report:\n{report_df}")
print(f"[Saved] Classification report → {report_path}")

# --- Prediction Probabilities & Labels → CSV ---
preds_df = pd.DataFrame({
    "y_true":      y_test.values if hasattr(y_test, "values") else y_test,
    "y_pred":      preds,
    "prob_churn":  probs,
})
preds_path = os.path.join(CSV_DIR, "test_predictions.csv")
preds_df.to_csv(preds_path, index=False)
print(f"[Saved] Test predictions       → {preds_path}")

# --- Summary Metrics → CSV ---
metrics_df = pd.DataFrame([{"metric": "AUC-ROC", "value": round(final_auc, 5)}])
metrics_path = os.path.join(CSV_DIR, "summary_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"[Saved] Summary metrics        → {metrics_path}")

# ==============================
# ROC Curve Plot → Image
# ==============================
fpr, tpr, thresholds = roc_curve(y_test, probs)

# --- ROC Data → CSV ---
roc_df   = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
roc_path = os.path.join(CSV_DIR, "roc_curve_data.csv")
roc_df.to_csv(roc_path, index=False)
print(f"[Saved] ROC curve data         → {roc_path}")

roc_display = RocCurveDisplay(
    fpr=fpr,
    tpr=tpr,
    roc_auc=final_auc,
    estimator_name='Optimized CatBoost Pipeline'
)

fig, ax = plt.subplots(figsize=(8, 6))
roc_display.plot(ax=ax, color='darkorange', lw=2)
ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')
ax.set_title('Receiver Operating Characteristic (ROC) — CatBoost', fontsize=14)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()

roc_img_path = os.path.join(IMAGE_DIR, "roc_curve.png")
fig.savefig(roc_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] ROC curve image        → {roc_img_path}")
plt.show()