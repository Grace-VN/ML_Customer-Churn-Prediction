import shap
from model.evaluation import final_pipeline, X_test, y_test
import pandas as pd

X_test_transform = final_pipeline.named_steps['preprocessor'].transform(X_test)
feature_names = final_pipeline.named_steps['preprocessor'].get_feature_names_out()
X_test_transform = pd.DataFrame(X_test_transform, columns=feature_names)
model = final_pipeline.named_steps['classifier']

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_transform)
shap.summary_plot(shap_values, X_test_transform)
test_columns = X_test_transform.columns
print(test_columns)
    
shap.dependence_plot("num__EstimatedSalary", shap_values, X_test_transform)