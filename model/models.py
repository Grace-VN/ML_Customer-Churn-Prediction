import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from data_processing.train_test_split import X_temp, y_temp, X_test, y_test
from data_processing.normalization_encoding import preprocessor

# ==============================
# Output Paths
# ==============================
CSV_DIR   = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\output_storage\csv_files"
IMAGE_DIR = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\output_storage\images"
os.makedirs(CSV_DIR,   exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ==============================
# Baseline Models
# ==============================
baseline_models = {
    'XGBoost': XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5
    ),
    'LightGBM': LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        random_state=42,
        verbose=0
    )
}

# ==============================
# Cross Validation
# ==============================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("--- Cross Validation OOF (AUC-ROC) Baselines ---\n")

results = []   # collect per-model summary
fold_rows = [] # collect per-fold scores

for name, model in baseline_models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier',   model)
    ])

    scores = cross_val_score(
        pipeline, X_temp, y_temp,
        cv=skf, scoring='roc_auc', n_jobs=-1
    )

    mean_score = np.mean(scores)
    std_score  = np.std(scores)
    print(f"{name}: {mean_score:.4f} (+/- {std_score:.4f})")

    # Summary row
    results.append({
        "model":    name,
        "mean_auc": round(mean_score, 6),
        "std_auc":  round(std_score,  6),
        **{f"fold_{i+1}": round(s, 6) for i, s in enumerate(scores)}
    })

    # Per-fold rows
    for i, s in enumerate(scores):
        fold_rows.append({"model": name, "fold": i + 1, "auc_roc": round(s, 6)})

# ==============================
# Save CSVs
# ==============================

# --- CV Summary (mean ± std + per-fold) → CSV ---
summary_df   = pd.DataFrame(results)
summary_path = os.path.join(CSV_DIR, "cv_baseline_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n[Saved] CV summary             → {summary_path}")

# --- Per-Fold Scores (long format) → CSV ---
fold_df   = pd.DataFrame(fold_rows)
fold_path = os.path.join(CSV_DIR, "cv_baseline_fold_scores.csv")
fold_df.to_csv(fold_path, index=False)
print(f"[Saved] Per-fold scores        → {fold_path}")

# ==============================
# Plot 1: Mean AUC-ROC Bar Chart → Image
# ==============================
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['steelblue', 'seagreen', 'darkorange', 'mediumpurple']
names  = summary_df["model"].tolist()
means  = summary_df["mean_auc"].tolist()
stds   = summary_df["std_auc"].tolist()

bars = ax.bar(names, means, yerr=stds, capsize=5,
              color=colors, alpha=0.85, edgecolor='black', linewidth=0.7)
ax.set_ylim(min(means) - 0.05, 1.0)
ax.set_title('Baseline Models — CV Mean AUC-ROC (±1 STD)', fontsize=14)
ax.set_xlabel('Model')
ax.set_ylabel('AUC-ROC')
ax.grid(axis='y', alpha=0.3)

for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{mean:.4f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
bar_img_path = os.path.join(IMAGE_DIR, "cv_baseline_bar_chart.png")
fig.savefig(bar_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Bar chart              → {bar_img_path}")
plt.show()

# ==============================
# Plot 2: Per-Fold Score Line Plot → Image
# ==============================
fig2, ax2 = plt.subplots(figsize=(9, 5))
fold_nums = list(range(1, skf.n_splits + 1))

for (name, color) in zip(names, colors):
    model_scores = fold_df[fold_df["model"] == name]["auc_roc"].tolist()
    ax2.plot(fold_nums, model_scores, marker='o', label=name, color=color, lw=2)

ax2.set_title('Baseline Models — AUC-ROC per Fold', fontsize=14)
ax2.set_xlabel('Fold')
ax2.set_ylabel('AUC-ROC')
ax2.set_xticks(fold_nums)
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()

fold_img_path = os.path.join(IMAGE_DIR, "cv_baseline_fold_lines.png")
fig2.savefig(fold_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Fold line plot         → {fold_img_path}")
plt.show()