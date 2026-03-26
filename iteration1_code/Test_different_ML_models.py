"""
Simple description:
reads a descriptor matrix with IC50 values,
converts them to pIC50,
trains four tree-based regression models on ECFP4 fingerprints (plus optional descriptors),
and prints their test-set R² and RMSE for comparison.
NOTE: doesnt save models! just shows r2 and rmse

INPUT
descriptors.csv file (FROM denerate_model_descriptors.py)
with columns: IC50 and at least one column starting with ecfp4_
-filters off IC50 0 values
-converts IC50 to pIC50
-does an 80/20 train/test split
trais randomforest, xgboost, lightgbm and catboost
(all are tree based regressors)

OUTPUT
printout of the models with their r2 and rmse values

FUTURE WORK: Progress bar?
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

#Fit model, evaluate on test set, print R2 and RMSE
def eval_model(model, X_train, y_train, X_test, y_test, name: str):

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"{name:10s} | R2 = {r2:6.3f} | RMSE = {rmse:6.3f}")


def main():
    df = pd.read_csv("descriptors.csv")

    if "IC50" not in df.columns:
        raise ValueError("Column 'IC50' not found in descriptors.csv")

    # Remove any invalid IC50
    df = df[df["IC50"] > 0].copy()
    if df.empty:
        raise ValueError("No valid IC50 values (>0) found after filtering.")


    # Convert IC50 to pIC50
    # pIC50 = -log10(IC50 in M)

    ic50_M = df["IC50"] * 1e-6   # change if concentration is not micromolar
    df["pIC50"] = -np.log10(ic50_M)



    # here you can change the specific features used!
    fp_cols = [c for c in df.columns if c.startswith("ecfp4_")]
    desc_cols = [c for c in ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds", "RingCount", "FractionCSP3", "HeavyAtomCount", "NumHeteroAtoms", "NumAtoms", "NumAromaticRings", "NHOHCount", "NOCount", "BalabanJ", "BertzCT", "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3" "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6","EState_VSA1","EState_VSA2", "VSA_EState1", "VSA_EState2"       ]

                 if c in df.columns]

    if not fp_cols:
        raise ValueError("No ECFP4 columns found (no columns starting with 'ecfp4_').")

    maccs_cols = [c for c in df.columns if c.startswith("maccs")]
    if not maccs_cols:
        raise ValueError("No maccs columns found (no columns starting with 'maccs_').")

    #define which feature column sets to use
    feature_cols = desc_cols + maccs_cols + fp_cols

    X = df[feature_cols].values
    #specify the target column
    y = df["pIC50"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )


    # Define models

    models = []

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=0,
        n_jobs=-1
    )
    models.append(("RandomForest", rf))

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",
    )
    models.append(("XGBoost", xgb_model))

    # LightGBM
    lgbm = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=512,
        max_depth=12,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        sparse_threshold=1.0,

    )
    models.append(("LightGBM", lgbm))

    # CatBoost
    cat = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        one_hot_max_size=65535,  # force one-hot encoding (reported to work well for ECFP)
        verbose=False,
    )
    models.append(("CatBoost", cat))


    # Train & evaluate all models

    print(f"Using {len(feature_cols)} features "
          f"({len(fp_cols)} ECFP4 bits + {len(desc_cols)} physchem descriptors)")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")
    print("Model       |    R2   |  RMSE")
    print("--------------------------------")

    for name, model in models:
        eval_model(model, X_train, y_train, X_test, y_test, name)


if __name__ == "__main__":
    main()
