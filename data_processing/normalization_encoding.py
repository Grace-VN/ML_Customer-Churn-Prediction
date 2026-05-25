import os
import pandas as pd
import matplotlib.pyplot as plt
from data_processing.train_test_split import X_temp
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ==============================
# Output Paths
# ==============================
CSV_DIR   = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\output_storage\csv_files"
IMAGE_DIR = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\output_storage\images"
os.makedirs(CSV_DIR,   exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ==============================
# Feature Groups
# ==============================
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
cat_cols = ['Geography', 'Gender']

# ==============================
# Pipelines
# ==============================
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='passthrough'
)

# ==============================
# Fit & Transform
# ==============================
X_transformed = preprocessor.fit_transform(X_temp)

# ==============================
# Feature Name Resolution
# ==============================
encoded_features = (
    preprocessor.named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names_out(cat_cols)
)
passthrough_cols = [col for col in X_temp.columns if col not in num_cols + cat_cols]
all_features     = num_cols + list(encoded_features) + passthrough_cols

# ==============================
# Console Summary
# ==============================
print("=" * 50)
print("Preprocessing Completed")
print("=" * 50)
print(f"\nOriginal Feature Shape    : {X_temp.shape}")
print(f"Transformed Feature Shape : {X_transformed.shape}")
print(f"\nTotal Engineered Features : {len(all_features)}")
print("\nSample Features:")
print(all_features[:10])

# ==============================
# Save CSVs
# ==============================

# --- Transformed Data → CSV ---
transformed_df   = pd.DataFrame(X_transformed, columns=all_features)
transformed_path = os.path.join(CSV_DIR, "X_transformed.csv")
transformed_df.to_csv(transformed_path, index=False)
print(f"\n[Saved] Transformed data        → {transformed_path}")

# --- Feature Inventory → CSV ---
feature_inventory_df = pd.DataFrame({
    "feature": all_features,
    "type": (
        ["numerical"] * len(num_cols) +
        ["categorical_encoded"] * len(encoded_features) +
        ["passthrough"] * len(passthrough_cols)
    )
})
feature_inventory_path = os.path.join(CSV_DIR, "feature_inventory.csv")
feature_inventory_df.to_csv(feature_inventory_path, index=False)
print(f"[Saved] Feature inventory       → {feature_inventory_path}")

# --- Preprocessing Summary → CSV ---
summary_df = pd.DataFrame([{
    "original_shape":    str(X_temp.shape),
    "transformed_shape": str(X_transformed.shape),
    "n_numerical":       len(num_cols),
    "n_categorical_raw": len(cat_cols),
    "n_encoded":         len(encoded_features),
    "n_passthrough":     len(passthrough_cols),
    "n_total_features":  len(all_features)
}])
summary_path = os.path.join(CSV_DIR, "preprocessing_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"[Saved] Preprocessing summary   → {summary_path}")

# ==============================
# Plot 1: Feature Type Breakdown (Pie) → Image
# ==============================
type_counts = feature_inventory_df["type"].value_counts()

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    type_counts.values,
    labels=type_counts.index,
    autopct='%1.1f%%',
    colors=['steelblue', 'darkorange', 'seagreen'],
    startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=1.5)
)
ax.set_title('Engineered Feature Type Breakdown', fontsize=13)
plt.tight_layout()

pie_img_path = os.path.join(IMAGE_DIR, "preprocessing_feature_breakdown.png")
fig.savefig(pie_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Feature breakdown pie   → {pie_img_path}")
plt.show()
plt.close()

# ==============================
# Plot 2: Scaled Numerical Feature Distributions → Image
# ==============================
num_df = transformed_df[num_cols]

fig2, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(num_df[col], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[i].set_title(col, fontsize=11)
    axes[i].set_xlabel('Scaled Value')
    axes[i].set_ylabel('Count')
    axes[i].grid(alpha=0.3)

plt.suptitle('Scaled Numerical Feature Distributions (Post-StandardScaler)', fontsize=13, y=1.01)
plt.tight_layout()

dist_img_path = os.path.join(IMAGE_DIR, "preprocessing_numerical_distributions.png")
fig2.savefig(dist_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Numerical distributions → {dist_img_path}")
plt.show()
plt.close()