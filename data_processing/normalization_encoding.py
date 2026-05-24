# ============================================
# 4. Preprocessing Pipeline
# ============================================

"""
Preprocessing Strategy
----------------------
Numerical Features:
    - Standardization using StandardScaler

Categorical Features:
    - One-Hot Encoding using OneHotEncoder

Goal:
Create a reusable preprocessing pipeline that integrates
seamlessly with multiple machine learning algorithms.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --------------------------------------------
# Feature Groups
# --------------------------------------------

num_cols = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'EstimatedSalary'
]

cat_cols = [
    'Geography',
    'Gender'
]

# --------------------------------------------
# Numerical Pipeline
# --------------------------------------------

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# --------------------------------------------
# Categorical Pipeline
# --------------------------------------------

categorical_transformer = Pipeline(steps=[
    (
        'onehot',
        OneHotEncoder(
            handle_unknown='ignore',
            drop='first'
        )
    )
])

# --------------------------------------------
# Combined Preprocessor
# --------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='passthrough'
)

# --------------------------------------------
# Fit & Transform Training Data
# --------------------------------------------

X_transformed = preprocessor.fit_transform(X_temp)

# ============================================
# Preprocessing Summary
# ============================================

print("=" * 50)
print("Preprocessing Completed")
print("=" * 50)

print(f"\nOriginal Feature Shape      : {X_temp.shape}")
print(f"Transformed Feature Shape   : {X_transformed.shape}")

# Display transformed feature names
encoded_features = preprocessor.named_transformers_['cat'] \
    .named_steps['onehot'] \
    .get_feature_names_out(cat_cols)

all_features = (
    num_cols +
    list(encoded_features) +
    [
        col for col in X_temp.columns
        if col not in num_cols + cat_cols
    ]
)

print(f"\nTotal Engineered Features: {len(all_features)}")
print("\nSample Features:")
print(all_features[:10])