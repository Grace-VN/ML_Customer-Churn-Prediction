from sklearn.metrics import classification_report, roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from model.fine_tuning import best_params
from model.models import baseline_models, preprocessor, X_temp, y_temp, X_test, y_test, skf
from data_processing.train_test_split import X_temp, y_temp, X_test, y_test
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
fpr, tpr, thresholds_roc = roc_curve(y_test, probs)

# --- ROC Data → CSV ---
roc_df   = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds_roc})
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

# ==============================
# Confusion Matrix
# ==============================
cm = confusion_matrix(y_test, preds)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'='*60}")
print(f"CONFUSION MATRIX")
print(f"{'='*60}")
print(f"True Negatives (TN):   {tn:,}")
print(f"False Positives (FP):  {fp:,}")
print(f"False Negatives (FN):  {fn:,}")
print(f"True Positives (TP):   {tp:,}")
print(f"Total samples:         {len(y_test):,}")

# --- Confusion Matrix → CSV ---
cm_df = pd.DataFrame(
    cm,
    index=["Actual: No Churn", "Actual: Churn"],
    columns=["Predicted: No Churn", "Predicted: Churn"]
)
cm_path = os.path.join(CSV_DIR, "confusion_matrix.csv")
cm_df.to_csv(cm_path)
print(f"\n[Saved] Confusion matrix       → {cm_path}")

# --- Detailed metrics from confusion matrix → CSV ---
cm_metrics_df = pd.DataFrame([
    {"metric": "True Negatives (TN)",  "value": tn},
    {"metric": "False Positives (FP)", "value": fp},
    {"metric": "False Negatives (FN)", "value": fn},
    {"metric": "True Positives (TP)",  "value": tp},
])
cm_metrics_path = os.path.join(CSV_DIR, "confusion_matrix_components.csv")
cm_metrics_df.to_csv(cm_metrics_path, index=False)
print(f"[Saved] CM components         → {cm_metrics_path}")

# ==============================
# Confusion Matrix Heatmap Plot
# ==============================
fig, ax = plt.subplots(figsize=(8, 6))

# Create heatmap
sns.heatmap(
    cm_df,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar_kws={'label': 'Count'},
    ax=ax,
    linewidths=1.5,
    linecolor='white',
    square=True,
    annot_kws={'size': 14, 'weight': 'bold'}
)

ax.set_title('Confusion Matrix — CatBoost Churn Prediction', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
plt.tight_layout()

cm_img_path = os.path.join(IMAGE_DIR, "confusion_matrix.png")
fig.savefig(cm_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Confusion matrix image → {cm_img_path}")
plt.show()

# ==============================
# Cost-Benefit Analysis (For Risk DS Framing)
# ==============================
print(f"\n{'='*60}")
print(f"COST-BENEFIT ANALYSIS (Risk Framing)")
print(f"{'='*60}")

# Define costs based on business context
cost_fn = 410   # Cost of False Negative: losing customer (e.g., $410 LTV loss)
cost_fp = 50    # Cost of False Positive: wasting retention offer (e.g., $50 offer cost)

total_fn_loss = fn * cost_fn
total_fp_cost = fp * cost_fp
total_cost    = total_fn_loss + total_fp_cost

# Baseline: no model (random or always "No Churn" prediction)
baseline_cost = len(y_test[y_test == 1]) * cost_fn  # All actual churners missed

cost_avoided = baseline_cost - total_cost
roi = (cost_avoided / total_cost * 100) if total_cost > 0 else 0

print(f"\nAssumptions:")
print(f"  • Cost of False Negative (missing a churner): ${cost_fn}")
print(f"  • Cost of False Positive (false alarm):       ${cost_fp}")
print(f"\nModel Cost:")
print(f"  • Total FN Loss:  {fn:,} missed churners × ${cost_fn} = ${total_fn_loss:,}")
print(f"  • Total FP Cost:  {fp:,} false alarms × ${cost_fp}    = ${total_fp_cost:,}")
print(f"  • Total Cost:     ${total_cost:,}")
print(f"\nBaseline (no model):")
print(f"  • Cost (miss all {len(y_test[y_test == 1]):,} churners): ${baseline_cost:,}")
print(f"\nROI:")
print(f"  • Cost Avoided:   ${cost_avoided:,}")
print(f"  • ROI:            {roi:.1f}%")

# --- Cost-Benefit → CSV ---
cost_benefit_df = pd.DataFrame([
    {"metric": "Cost of FN (missed churn)",     "value": cost_fn},
    {"metric": "Cost of FP (false alarm)",      "value": cost_fp},
    {"metric": "Total FN Loss",                 "value": total_fn_loss},
    {"metric": "Total FP Cost",                 "value": total_fp_cost},
    {"metric": "Total Model Cost",              "value": total_cost},
    {"metric": "Baseline Cost (no model)",      "value": baseline_cost},
    {"metric": "Cost Avoided",                  "value": cost_avoided},
    {"metric": "ROI (%)",                       "value": round(roi, 1)},
])
cost_benefit_path = os.path.join(CSV_DIR, "cost_benefit_analysis.csv")
cost_benefit_df.to_csv(cost_benefit_path, index=False)
print(f"\n[Saved] Cost-benefit analysis  → {cost_benefit_path}")

# ==============================
# Threshold Optimization (Risk Metrics)
# ==============================
print(f"\n{'='*60}")
print(f"THRESHOLD OPTIMIZATION (Precision vs Recall Trade-off)")
print(f"{'='*60}")

precision, recall, thresholds = precision_recall_curve(y_test, probs)

# Find thresholds where precision meets different targets
target_precisions = [0.70, 0.75, 0.80, 0.85, 0.90]
threshold_analysis = []

for target_prec in target_precisions:
    idx = np.where(precision >= target_prec)[0]
    if len(idx) > 0:
        best_idx = idx[np.argmax(recall[idx])]
        opt_threshold = thresholds[best_idx]
        opt_recall = recall[best_idx]
        
        print(f"At {target_prec*100:.0f}% Precision: {opt_recall*100:.1f}% Recall, Threshold={opt_threshold:.3f}")
        
        threshold_analysis.append({
            "target_precision": target_prec,
            "achieved_precision": round(precision[best_idx], 4),
            "achieved_recall": round(opt_recall, 4),
            "threshold": round(opt_threshold, 4)
        })

# --- Threshold analysis → CSV ---
threshold_df = pd.DataFrame(threshold_analysis)
threshold_path = os.path.join(CSV_DIR, "threshold_optimization.csv")
threshold_df.to_csv(threshold_path, index=False)
print(f"\n[Saved] Threshold optimization → {threshold_path}")

print(f"\n{'='*60}")
print(f"All outputs saved to: {CSV_DIR}")
print(f"{'='*60}")