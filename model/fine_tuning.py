import optuna
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'iterations': trial.suggest_int('iterations', 200, 300, 400),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'random_state': 42,
        'verbose': 0,
        'eval_metric': 'AUC'
    }

    model = CatBoostClassifier(**params)
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    score = cross_val_score(pipe, X_temp, y_temp, cv=skf, scoring='roc_auc', n_jobs=-1).mean()

    return score

# Suppress Optuna's default logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("\nBest Hyperparameters found:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

print(f"\nBest AUC-ROC: {study.best_value:.4f}")