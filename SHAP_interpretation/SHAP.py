import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
from model.evaluation import final_pipeline, X_test, y_test

# ==============================
# Output Paths
# ==============================
CSV_DIR   = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\output_storage\csv_files"
IMAGE_DIR = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\output_storage\images"
os.makedirs(CSV_DIR,   exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ==============================
# Preprocessing & SHAP Setup
# ==============================
X_test_transform = final_pipeline.named_steps['preprocessor'].transform(X_test)
feature_names    = final_pipeline.named_steps['preprocessor'].get_feature_names_out()
X_test_transform = pd.DataFrame(X_test_transform, columns=feature_names)
model            = final_pipeline.named_steps['classifier']

print("Test columns:\n", X_test_transform.columns.tolist())

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_transform)

# ==============================
# Save SHAP Values → CSV
# ==============================
shap_df   = pd.DataFrame(shap_values, columns=feature_names)
shap_path = os.path.join(CSV_DIR, "shap_values.csv")
shap_df.to_csv(shap_path, index=False)
print(f"\n[Saved] SHAP values            → {shap_path}")

# --- Mean Absolute SHAP (feature importance) → CSV ---
mean_shap_df = (
    pd.DataFrame({
        "feature":         feature_names,
        "mean_abs_shap":   pd.DataFrame(shap_values).abs().mean().values
    })
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
mean_shap_path = os.path.join(CSV_DIR, "shap_feature_importance.csv")
mean_shap_df.to_csv(mean_shap_path, index=False)
print(f"[Saved] SHAP feature importance → {mean_shap_path}")

# ==============================
# Plot 1: SHAP Summary Plot → Image
# ==============================
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test_transform, show=False)
plt.tight_layout()

summary_img_path = os.path.join(IMAGE_DIR, "shap_summary_plot.png")
plt.savefig(summary_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] SHAP summary plot      → {summary_img_path}")
plt.show()
plt.close()

# ==============================
# Plot 2: SHAP Bar Plot (Mean |SHAP|) → Image
# ==============================
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test_transform, plot_type='bar', show=False)
plt.tight_layout()

bar_img_path = os.path.join(IMAGE_DIR, "shap_bar_plot.png")
plt.savefig(bar_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] SHAP bar plot          → {bar_img_path}")
plt.show()
plt.close()

# ==============================
# Plot 3: SHAP Dependence Plot → Image
# ==============================
fig, ax = plt.subplots()
shap.dependence_plot(
    "num__EstimatedSalary",
    shap_values,
    X_test_transform,
    ax=ax,
    show=False
)
plt.tight_layout()

dep_img_path = os.path.join(IMAGE_DIR, "shap_dependence_EstimatedSalary.png")
fig.savefig(dep_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Dependence plot        → {dep_img_path}")
plt.show()
plt.close()