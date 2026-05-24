from sklearn.metrics import classification_report, roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
from model.fine_tuning import best_params
from model.models import baseline_models, preprocessor, X_temp, y_temp, X_test, y_test, skf
from catboost import CatBoostClassifier
from data_processing.train_test_split import X_temp, y_temp, X_test, y_test
from sklearn.pipeline import Pipeline

# ==============================
# Final Model Selection
# ==============================
if 'best_params' in locals():
    print(">>> Using Optimized Hyperparameters from Optuna Tuning")
    final_model = CatBoostClassifier(
        **best_params,
        random_state=42,
        verbose=0,
        eval_metric='AUC'
    )
else:
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
print("\nClassification Report:")
print(classification_report(y_test, preds))

# ==============================
# ROC Curve Plot
# ==============================
fpr, tpr, thresholds = roc_curve(y_test, probs)

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
plt.show()
