import numpy as np
import pandas as pd
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: install packages from inside Python
# Usually better to install from terminal instead
# !{sys.executable} -m pip install -q pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost optuna shap

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Check files in your local folder
folder_path = r"D:\Job\Portfolio\Machine Learning\Customer Churn Prediction\data_processing"

for dirname, _, filenames in os.walk(folder_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))