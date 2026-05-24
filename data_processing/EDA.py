# ============================================
# Customer Churn Prediction Pipeline
# Local VS Code Version
# ============================================

# Core Libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# System Utilities
import os
import warnings

# --------------------------------------------
# Settings
# --------------------------------------------
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

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

# Local dataset path
DATA_PATH = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\data_processing\train.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Dataset overview
print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print(f"\nDataset Shape: {df.shape}")

# Preview data
print("\nFirst 5 Rows:")
print(df.head())

# ============================================
# 3. Basic Dataset Information
# ============================================

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe().T)

# ============================================
# 4. Target Variable Analysis
# ============================================

target_col = 'Exited'

# Class distribution
churn_counts = (
    df[target_col]
    .value_counts(normalize=True)
    .round(3) * 100
)

print("\nTarget Distribution:")
print(f"Retained Customers (0): {churn_counts[0]:.1f}%")
print(f"Churned Customers (1): {churn_counts[1]:.1f}%")

# Visualization
plt.figure(figsize=(6, 4))

sns.countplot(x=df[target_col])

plt.title("Customer Churn Distribution")
plt.xlabel("Exited")
plt.ylabel("Count")

plt.show()

# ============================================
# 5. Numerical Feature Analysis
# ============================================

numerical_cols = df.select_dtypes(include=np.number).columns

# Histograms
df[numerical_cols].hist(
    figsize=(15, 10),
    bins=30
)

plt.suptitle("Numerical Feature Distributions")
plt.tight_layout()
plt.show()

# ============================================
# 6. Correlation Heatmap
# ============================================

plt.figure(figsize=(12, 8))

correlation_matrix = df[numerical_cols].corr()

sns.heatmap(
    correlation_matrix,
    annot=False,
    cmap='coolwarm',
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")

plt.show()

# ============================================
# 7. Optional: Save Processed Copy
# ============================================

output_path = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\data_processing\processed_train.csv"

df.to_csv(output_path, index=False)

print(f"\nProcessed dataset saved to:\n{output_path}")