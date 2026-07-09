import pickle
import pandas as pd
import numpy as np
import shap

with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('shap_explainer.pkl', 'rb') as f:
    shap_explainer = pickle.load(f)

features = pd.DataFrame([{
    'amount_z_score': 0.0,
    'days_since_last': 30,
    'vendor_age': 1,
    'frequency': 1,
    'round_number_flag': 0,
    'weekend_flag': 1,
    'line_count': 2,
    'ghost_flag': 1,
    'category_z_score': 0.0,
    'po_match_int': 0
}])

prob = xgb_model.predict_proba(features)[0, 1]
print("Probability:", prob)
sv = shap_explainer.shap_values(features)[0]
print("SHAP Values:", sv)
