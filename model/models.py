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

METRICS = {
    'auc_roc':   'roc_auc',
    'accuracy':  'accuracy',
    'f1':        'f1',
    'precision': 'precision',
    'recall':    'recall',
}

print("--- Cross Validation OOF Baselines ---\n")

results   = []  # per-model summary (mean ± std for all metrics)
fold_rows = []  # per-model per-fold for all metrics

for name, model in baseline_models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier',   model)
    ])

    metric_scores = {}
    for metric_key, sklearn_scorer in METRICS.items():
        scores = cross_val_score(
            pipeline, X_temp, y_temp,
            cv=skf, scoring=sklearn_scorer, n_jobs=-1
        )
        metric_scores[metric_key] = scores

    # Print summary line
    auc_scores = metric_scores['auc_roc']
    print(
        f"{name:10s} | AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}"
        f" | Acc: {np.mean(metric_scores['accuracy']):.4f}"
        f" | F1: {np.mean(metric_scores['f1']):.4f}"
        f" | Prec: {np.mean(metric_scores['precision']):.4f}"
        f" | Rec: {np.mean(metric_scores['recall']):.4f}"
    )

    # Build summary row
    summary_row = {"model": name}
    for metric_key, scores in metric_scores.items():
        summary_row[f"mean_{metric_key}"] = round(np.mean(scores), 6)
        summary_row[f"std_{metric_key}"]  = round(np.std(scores),  6)
        for i, s in enumerate(scores):
            summary_row[f"{metric_key}_fold_{i+1}"] = round(s, 6)
    results.append(summary_row)

    # Build long-format fold rows
    n_folds = skf.n_splits
    for i in range(n_folds):
        fold_row = {"model": name, "fold": i + 1}
        for metric_key, scores in metric_scores.items():
            fold_row[metric_key] = round(scores[i], 6)
        fold_rows.append(fold_row)

# ==============================
# Save CSVs
# ==============================
summary_df   = pd.DataFrame(results)
summary_path = os.path.join(CSV_DIR, "cv_baseline_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n[Saved] CV summary        → {summary_path}")

fold_df   = pd.DataFrame(fold_rows)
fold_path = os.path.join(CSV_DIR, "cv_baseline_fold_scores.csv")
fold_df.to_csv(fold_path, index=False)
print(f"[Saved] Per-fold scores   → {fold_path}")

# ==============================
# Helpers
# ==============================
MODEL_COLORS = {
    'XGBoost':  'steelblue',
    'KNN':      'seagreen',
    'LightGBM': 'darkorange',
    'CatBoost': 'mediumpurple',
}
names  = summary_df["model"].tolist()
colors = [MODEL_COLORS[n] for n in names]
fold_nums = list(range(1, skf.n_splits + 1))

METRIC_LABELS = {
    'auc_roc':   'AUC-ROC',
    'accuracy':  'Accuracy',
    'f1':        'F1 Score',
    'precision': 'Precision',
    'recall':    'Recall',
}

# ==============================
# Plot 1: Grouped Bar Chart — All Metrics Mean
# ==============================
fig, ax = plt.subplots(figsize=(13, 6))

metric_keys  = list(METRIC_LABELS.keys())
metric_lbls  = list(METRIC_LABELS.values())
n_metrics    = len(metric_keys)
n_models     = len(names)
x            = np.arange(n_metrics)
bar_width    = 0.18
offsets      = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

for idx, (name, color) in enumerate(zip(names, colors)):
    means = [summary_df.loc[summary_df["model"] == name, f"mean_{m}"].values[0] for m in metric_keys]
    stds  = [summary_df.loc[summary_df["model"] == name, f"std_{m}"].values[0]  for m in metric_keys]
    bars  = ax.bar(x + offsets[idx], means, bar_width,
                   yerr=stds, capsize=4,
                   label=name, color=color, alpha=0.85,
                   edgecolor='black', linewidth=0.6)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{mean:.3f}", ha='center', va='bottom', fontsize=7)

all_means = [
    summary_df.loc[summary_df["model"] == n, f"mean_{m}"].values[0]
    for n in names for m in metric_keys
]
ax.set_ylim(min(all_means) - 0.05, 1.05)
ax.set_title('Baseline Models — CV Mean Metrics (±1 STD)', fontsize=14)
ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(metric_lbls)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

grouped_bar_path = os.path.join(IMAGE_DIR, "cv_baseline_grouped_bar.png")
fig.savefig(grouped_bar_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Grouped bar chart → {grouped_bar_path}")
plt.show()

# ==============================
# Plot 2: Per-Fold Line Plots — One Subplot per Metric
# ==============================
fig2, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for ax_idx, (metric_key, metric_lbl) in enumerate(METRIC_LABELS.items()):
    ax = axes[ax_idx]
    for name, color in zip(names, colors):
        model_scores = fold_df[fold_df["model"] == name][metric_key].tolist()
        ax.plot(fold_nums, model_scores, marker='o', label=name, color=color, lw=2)
    ax.set_title(metric_lbl, fontsize=12)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_xticks(fold_nums)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# Hide unused 6th subplot
axes[-1].set_visible(False)

fig2.suptitle('Baseline Models — Score per Fold by Metric', fontsize=14, y=1.01)
plt.tight_layout()

fold_lines_path = os.path.join(IMAGE_DIR, "cv_baseline_fold_lines_all_metrics.png")
fig2.savefig(fold_lines_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Fold line plots   → {fold_lines_path}")
plt.show()

# ==============================
# Plot 3: Per-Model Radar Chart — Mean of All Metrics
# ==============================
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

radar_labels = metric_lbls
num_vars     = len(radar_labels)
angles       = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles      += angles[:1]  # close the polygon

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10),
                            subplot_kw=dict(polar=True))
axes3 = axes3.flatten()

for ax_idx, (name, color) in enumerate(zip(names, colors)):
    ax = axes3[ax_idx]
    means = [summary_df.loc[summary_df["model"] == name, f"mean_{m}"].values[0]
             for m in metric_keys]
    values = means + means[:1]

    ax.plot(angles, values, color=color, lw=2)
    ax.fill(angles, values, color=color, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(name, size=13, pad=14, color=color, fontweight='bold')
    ax.grid(alpha=0.3)

fig3.suptitle('Baseline Models — Metric Radar Charts', fontsize=14, y=1.01)
plt.tight_layout()

radar_path = os.path.join(IMAGE_DIR, "cv_baseline_radar_charts.png")
fig3.savefig(radar_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Radar charts      → {radar_path}")
plt.show()