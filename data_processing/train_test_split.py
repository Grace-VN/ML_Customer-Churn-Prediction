# ============================================
# 8. Feature Selection & Data Splitting
# ============================================

from sklearn.model_selection import train_test_split

# Remove non-informative identifier columns
df_clean = df.drop(columns=['id', 'CustomerId', 'Surname'])

# Separate features and target variable
X = df_clean.drop(columns=['Exited'])
y = df_clean['Exited']

# --------------------------------------------
# Train / Test Split
# --------------------------------------------
"""
X_temp:
    Used for model training, cross-validation,
    and hyperparameter optimization.

X_test:
    Completely untouched hold-out dataset
    reserved for final evaluation only.
"""

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================
# Split Summary
# ============================================

print("=" * 50)
print("Data Splitting Completed")
print("=" * 50)

print(f"\nTraining & Validation Set Shape : {X_temp.shape}")
print(f"Final Hold-Out Test Set Shape   : {X_test.shape}")

# Check target balance preservation
train_churn_rate = y_temp.mean() * 100
test_churn_rate = y_test.mean() * 100

print("\nTarget Distribution:")
print(f"Training Set Churn Rate : {train_churn_rate:.2f}%")
print(f"Test Set Churn Rate     : {test_churn_rate:.2f}%")