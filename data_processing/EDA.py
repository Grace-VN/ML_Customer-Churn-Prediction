# ============================================
# Customer Churn Prediction Pipeline
# Local VS Code Version
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# --------------------------------------------
# Settings
# --------------------------------------------
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# ==============================
# Output Paths
# ==============================
CSV_DIR   = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\output_storage\csv_files"
IMAGE_DIR = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\output_storage\images"
os.makedirs(CSV_DIR,   exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ============================================
# 1. Problem Definition
# ============================================
"""
Business Context:
    Customer churn is expensive for companies because acquiring
    new customers costs significantly more than retaining existing ones.
Goal:
    Predict whether a customer is likely to leave the service
    so the company can take preventive action.
"""

# ============================================
# 2. Data Loading
# ============================================
DATA_PATH = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\data_processing\train.csv"
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)
print(f"\nDataset Shape: {df.shape}")
print("\nFirst 5 Rows:")
print(df.head())

# ============================================
# 3. Basic Dataset Information
# ============================================
print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
missing = df.isnull().sum()
print(missing)

print("\nStatistical Summary:")
stats = df.describe().T
print(stats)

# --- Missing Values → CSV ---
missing_df   = missing.reset_index()
missing_df.columns = ["feature", "missing_count"]
missing_df["missing_pct"] = (missing_df["missing_count"] / len(df) * 100).round(4)
missing_path = os.path.join(CSV_DIR, "eda_missing_values.csv")
missing_df.to_csv(missing_path, index=False)
print(f"\n[Saved] Missing values          → {missing_path}")

# --- Statistical Summary → CSV ---
stats_path = os.path.join(CSV_DIR, "eda_statistical_summary.csv")
stats.to_csv(stats_path)
print(f"[Saved] Statistical summary     → {stats_path}")

# ============================================
# 4. Target Variable Analysis
# ============================================
target_col   = 'Exited'
churn_counts = df[target_col].value_counts(normalize=True).round(3) * 100

print("\nTarget Distribution:")
print(f"Retained Customers (0): {churn_counts[0]:.1f}%")
print(f"Churned Customers  (1): {churn_counts[1]:.1f}%")

# --- Target Distribution → CSV ---
target_df = (
    df[target_col]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "class", target_col: "count"})
)
target_df["label"]   = target_df[target_col].map({0: "Retained", 1: "Churned"})
target_df["pct"]     = (target_df["count"] / len(df) * 100).round(2)
target_path          = os.path.join(CSV_DIR, "eda_target_distribution.csv")
target_df.to_csv(target_path, index=False)
print(f"[Saved] Target distribution     → {target_path}")

# --- Churn Distribution Plot → Image ---
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=df[target_col], ax=ax)
# Add count labels on top of bars
for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width() / 2, p.get_height() + 10),  # +10 offset
        ha='center',
        va='bottom',
        fontsize=10
    )
ax.set_title("Customer Churn Distribution")
ax.set_xlabel("Exited")
ax.set_ylabel("Count")
plt.tight_layout()
churn_img_path = os.path.join(
    IMAGE_DIR,
    "eda_churn_distribution.png"
)
fig.savefig(churn_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Churn distribution plot → {churn_img_path}")
plt.show()
plt.close()

# ============================================
# 5. Numerical Feature Analysis
# ============================================
numerical_cols = df.select_dtypes(include=np.number).columns

fig = df[numerical_cols].hist(figsize=(15, 10), bins=30)
plt.suptitle("Numerical Feature Distributions")
plt.tight_layout()

hist_img_path = os.path.join(IMAGE_DIR, "eda_numerical_distributions.png")
plt.savefig(hist_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Numerical distributions → {hist_img_path}")
plt.show()
plt.close()

# ============================================
# 6. Correlation Heatmap
# ============================================
correlation_matrix = df[numerical_cols].corr()

# --- Correlation Matrix → CSV ---
corr_path = os.path.join(CSV_DIR, "eda_correlation_matrix.csv")
correlation_matrix.to_csv(corr_path)
print(f"[Saved] Correlation matrix      → {corr_path}")

# --- Correlation Heatmap → Image ---
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.5,
    ax=ax
)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()

heatmap_img_path = os.path.join(IMAGE_DIR, "eda_correlation_heatmap.png")
fig.savefig(heatmap_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Correlation heatmap     → {heatmap_img_path}")
plt.show()
plt.close()

# ============================================
# 7. Save Processed Copy
# ============================================
output_path = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\data_processing\processed_train.csv"
df.to_csv(output_path, index=False)
print(f"\n[Saved] Processed dataset       → {output_path}")