from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import numpy as np
from data_processing.train_test_split import X_temp, y_temp, X_test, y_test
from data_processing.setup import preprocessor


# ==============================
# Baseline & Hybrid Models
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

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("--- Cross Validation OOF (AUC-ROC) Baselines ---\n")

for name, model in baseline_models.items():

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    scores = cross_val_score(
        pipeline,
        X_temp,
        y_temp,
        cv=skf,
        scoring='roc_auc',
        n_jobs=-1
    )

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print(f"{name}: {mean_score:.4f} (+/- {std_score:.4f})")