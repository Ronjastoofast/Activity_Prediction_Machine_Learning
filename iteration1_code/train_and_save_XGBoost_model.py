"""
simple description:
reads in data, splits into training and test sets, extracts feature columns,
trains model, tests model, saves model, prints r2 and rmse

INPUTS:
descriptors.csv (from generate_molecular_descriptors.py)
 -and you can specify which features from this dataset to use

OUTPUTS:
trained model (.joblib)
printout of rs and rmse
"""


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import joblib



# Load data

df = pd.read_csv("descriptors.csv")

# IC50 in µM -> pIC50
df = df[df["IC50"] > 0].copy()
ic50_M = df["IC50"] * 1e-6   # µM to M
df["pIC50"] = -np.log10(ic50_M)


# Features: ECFP4 bits + maccs_columns + some descriptors (only if they exist in the descriptors.csv)
maccs_cols = [c for c in df.columns if c.startswith("maccs")]
fp_cols = [c for c in df.columns if c.startswith("ecfp4_")]
desc_cols = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA",
    "NumRotatableBonds", "RingCount", "FractionCSP3",
    # plus any extra descriptors you’re actually using at TRAIN TIME:
    "HeavyAtomCount", "NumHeteroAtoms", "NumAtoms",
    "NumAromaticAtoms", "NumAromaticRings",
    "NHOHCount", "NOCount",
    "BalabanJ", "BertzCT", "HallKierAlpha",
    "Kappa1", "Kappa2", "Kappa3",
    "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6",
    "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6",
    "EState_VSA1", "EState_VSA2", "VSA_EState1", "VSA_EState2",
]

# keep only those that actually exist in df
desc_cols = [c for c in desc_cols if c in df.columns]
feature_cols = fp_cols + desc_cols + maccs_cols
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

joblib.dump(
    {"model": best_model, "feature_cols": feature_cols},
    "xgboost_pIC50_model_tuned.joblib"
)
print("Saved tuned model to xgboost_pIC50_model_tuned.joblib")





