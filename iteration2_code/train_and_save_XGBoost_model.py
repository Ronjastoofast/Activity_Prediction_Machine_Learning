"""
DIFFERENCE FROM ITERATION 1:
model was saved as xgboost_pIC50_model_tuned3
feature columns were selected using feature_selector.py

simple description:
reads in data, splits into training and test sets, extracts feature columns,
trains model, tests model, saves model, prints r2 and rmse

INPUTS:
descriptors.csv (from generate_molecular_descriptors.py)
 -and you can specify which features from this dataset to use

OUTPUTS:
trained model (.joblib)
printout of r2 and rmse
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import joblib



# Load data

df = pd.read_csv("descriptors.csv") # from generate_molecular_descriptors.py

# IC50 in µM -> pIC50
df = df[df["IC50"] > 0].copy()
ic50_M = df["IC50"] * 1e-6   # µM to M
df["pIC50"] = -np.log10(ic50_M)


# Define Feature columns


#selected using feature_selector
fp_select = ['ecfp4_1083', 'ecfp4_456', 'ecfp4_802', 'ecfp4_1027', 'ecfp4_535', 'ecfp4_1096', 'ecfp4_1160', 'ecfp4_257', 'ecfp4_1459', 'ecfp4_1745', 'ecfp4_270', 'ecfp4_341', 'ecfp4_635', 'ecfp4_1019', 'ecfp4_366', 'ecfp4_437', 'ecfp4_1601', 'ecfp4_1070', 'ecfp4_993', 'ecfp4_701', 'ecfp4_1304', 'ecfp4_1697', 'ecfp4_729', 'ecfp4_1255', 'ecfp4_915', 'ecfp4_506', 'ecfp4_1385', 'ecfp4_1939', 'ecfp4_667', 'ecfp4_896', 'ecfp4_1535', 'ecfp4_1138', 'ecfp4_935', 'ecfp4_1855', 'ecfp4_1099', 'ecfp4_1582', 'ecfp4_715', 'ecfp4_807', 'ecfp4_806', 'ecfp4_1984', 'ecfp4_407', 'ecfp4_1057', 'ecfp4_461', 'ecfp4_1387', 'ecfp4_1831', 'ecfp4_275', 'ecfp4_1754', 'ecfp4_1324', 'ecfp4_1795', 'ecfp4_1800', 'ecfp4_1542', 'ecfp4_389', 'ecfp4_1325', 'ecfp4_1665', 'ecfp4_881', 'ecfp4_1457', 'ecfp4_1529', 'ecfp4_1696', 'ecfp4_1921', 'ecfp4_486', 'ecfp4_680', 'ecfp4_964']
desc_select = desc_cols = ['Kappa3', 'SlogP_VSA1', 'SlogP_VSA3', 'PEOE_VSA2', 'VSA_EState2', 'Kappa1', 'PEOE_VSA1', 'NumAromaticAtoms', 'MolWt', 'NOCount', 'PEOE_VSA3', 'VSA_EState1', 'MolLogP', 'BalabanJ', 'HallKierAlpha', 'HeavyAtomCount', 'EState_VSA1', 'TPSA']
maccs_select = ['maccs_152', 'maccs_25', 'maccs_144', 'maccs_112', 'maccs_89', 'maccs_94', 'maccs_80', 'maccs_129', 'maccs_77', 'maccs_69', 'maccs_149', 'maccs_91', 'maccs_106', 'maccs_110', 'maccs_146', 'maccs_150', 'maccs_59', 'maccs_145', 'maccs_140', 'maccs_37']

select = fp_select+desc_select+maccs_select


feature_cols = select
X = df[feature_cols].values
y = df["pIC50"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


# Base model

base_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_jobs=-1,
    tree_method="hist",
)


# Hyperparameter search space

param_dist = {
    "n_estimators":   [500, 1000, 1500, 2000],
    "learning_rate":  [0.01, 0.02, 0.03, 0.05],
    "max_depth":      [4, 6, 8, 10],
    "subsample":      [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda":     [0.1, 1.0, 10.0],
    "reg_alpha":      [0.0, 0.1, 1.0],
}

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=30,                # number of random configs to try
    scoring="neg_root_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=0,
)

search.fit(X_train, y_train)

print("\nBest params:")
print(search.best_params_)
print("Best CV score (neg RMSE):", search.best_score_)


# Evaluate best model on held-out test set

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)

print("\nTest set performance with tuned XGBoost:")
print("R2:", r2)
print("RMSE:", rmse)


# Save tuned model

#get metadata on number of ecfp columns, needed for later analysis
ecfp_ids = [
    int(c.split("_")[-1])
    for c in feature_cols
    if c.startswith("ecfp4_")
]

ecfp_nbits = max(ecfp_ids) + 1

joblib.dump(
    {"model": best_model,
     "feature_cols": feature_cols, #useful for feature engineering
     "ecfp_radius": 2,
     "ecfp_nbits": ecfp_nbits,
     },
    "xgboost_pIC50_model_tuned2.joblib"
    )
print("Saved tuned model to xgboost_pIC50_model_tuned2.joblib")
