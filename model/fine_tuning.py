import os
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from data_processing.train_test_split import X_temp, y_temp
from data_processing.normalization_encoding import preprocessor

# ==============================
# Output Paths
# ==============================
CSV_DIR   = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\output_storage\csv_files"
IMAGE_DIR = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\output_storage\images"
os.makedirs(CSV_DIR,   exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ==============================
# StratifiedKFold
# ==============================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==============================
# Optuna Objective
# ==============================
def objective(trial):
    params = {
        'learning_rate':       trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'iterations':          trial.suggest_int('iterations', 200, 400),        # bug fix: was (200, 300, 400)
        'depth':               trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count':        trial.suggest_int('border_count', 32, 255),
        'random_strength':     trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'random_state': 42,
        'verbose':      0,
        'eval_metric':  'AUC'
    }
    model = CatBoostClassifier(**params)
    pipe  = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    score = cross_val_score(pipe, X_temp, y_temp, cv=skf, scoring='roc_auc', n_jobs=-1).mean()
    return score

# ==============================
# Run Study
# ==============================
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("\nBest Hyperparameters found:")
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"\nBest AUC-ROC: {study.best_value:.4f}")

# ==============================
# Save CSVs
# ==============================

# --- Best Hyperparameters → CSV ---
best_params_df   = pd.DataFrame(list(best_params.items()), columns=["parameter", "value"])
best_params_path = os.path.join(CSV_DIR, "best_hyperparameters.csv")
best_params_df.to_csv(best_params_path, index=False)
print(f"\n[Saved] Best hyperparameters    → {best_params_path}")

# --- All Trials → CSV ---
trials_df   = study.trials_dataframe().round(6)
trials_path = os.path.join(CSV_DIR, "optuna_trials.csv")
trials_df.to_csv(trials_path, index=False)
print(f"[Saved] All trial results       → {trials_path}")

# --- Study Summary → CSV ---
summary_df = pd.DataFrame([{
    "metric":      "best_auc_roc",
    "value":       round(study.best_value, 6),
    "best_trial":  study.best_trial.number,
    "n_trials":    len(study.trials)
}])
summary_path = os.path.join(CSV_DIR, "optuna_study_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"[Saved] Study summary           → {summary_path}")

# ==============================
# Plot 1: Optimization History → Image
# ==============================
trial_numbers = [t.number for t in study.trials if t.value is not None]
trial_values  = [t.value  for t in study.trials if t.value is not None]
best_so_far   = [max(trial_values[:i+1]) for i in range(len(trial_values))]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(trial_numbers, trial_values, color='steelblue', alpha=0.5, s=25, label='Trial AUC-ROC')
ax.plot(trial_numbers, best_so_far, color='darkorange', lw=2, label='Best so far')
ax.set_title('Optuna Optimization History — CatBoost', fontsize=14)
ax.set_xlabel('Trial Number')
ax.set_ylabel('AUC-ROC')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

history_img_path = os.path.join(IMAGE_DIR, "optuna_optimization_history.png")
fig.savefig(history_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Optimization history    → {history_img_path}")
plt.show()
plt.close()

# ==============================
# Plot 2: Hyperparameter Importance → Image
# ==============================
importances = optuna.importance.get_param_importances(study)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(list(importances.keys())[::-1], list(importances.values())[::-1], color='steelblue')
ax2.set_title('Hyperparameter Importance — Optuna', fontsize=14)
ax2.set_xlabel('Importance Score')
ax2.grid(axis='x', alpha=0.3)
plt.tight_layout()

importance_img_path = os.path.join(IMAGE_DIR, "optuna_hyperparameter_importance.png")
fig2.savefig(importance_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Hyperparameter importance → {importance_img_path}")
plt.show()
plt.close()