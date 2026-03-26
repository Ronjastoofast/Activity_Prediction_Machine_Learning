"""
simple description:
used to select features to use in training the model
(used to cut down the number of features)
-removes features with little variation
-removes highly correlated features
-then chooses a smaller subset of features from this by repeatedly training models with or without features


INPUTS:
descriptors.csv (was generated in iteration 1)

OUTPUTS:
prints a copypastable list of good features
saves a csv of these features

"""

import pandas as pd
import numpy as np
import joblib

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
import xgboost as xgb


#INPUTS
INPUT_CSV = "descriptors.csv"

VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95
TOP_N_IMPORTANCE = 200
RFE_FEATURES = 100



# LOAD DATA

df = pd.read_csv(INPUT_CSV)

df = df[df["IC50"] > 0].copy()

# target
ic50_M = df["IC50"] * 1e-6
df["pIC50"] = -np.log10(ic50_M)

y = df["pIC50"].values



#build feature lists

fp_cols = [c for c in df.columns if c.startswith("ecfp4_")]
maccs_cols = [c for c in df.columns if c.startswith("maccs")]

desc_cols = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA",
    "NumRotatableBonds", "RingCount", "FractionCSP3",
    "HeavyAtomCount", "NumHeteroAtoms", "NumAtoms",
    "NumAromaticAtoms", "NumAromaticRings",
    "NHOHCount", "NOCount",
    "BalabanJ", "BertzCT", "HallKierAlpha",
    "Kappa1", "Kappa2", "Kappa3",
    "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6",
    "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6",
    "EState_VSA1", "EState_VSA2", "VSA_EState1", "VSA_EState2",
]

desc_cols = [c for c in desc_cols if c in df.columns]

feature_cols = fp_cols + desc_cols + maccs_cols

X = df[feature_cols].values

print(f"[INFO] Starting with {len(feature_cols)} features")



#low variance filter

selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
X_var = selector.fit_transform(X)

features_var = [
    f for f, keep in zip(feature_cols, selector.get_support()) if keep
]

print(f"[STEP 1] After variance filter: {len(features_var)} features")



#correlation filter

df_var = pd.DataFrame(X_var, columns=features_var)

corr_matrix = df_var.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = [col for col in upper.columns if any(upper[col] > CORRELATION_THRESHOLD)]

features_corr = [f for f in features_var if f not in to_drop]

X_corr = df_var[features_corr].values

print(f"[STEP 2] After correlation filter: {len(features_corr)} features")



#model importance filter

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    n_jobs=-1,
)

model.fit(X_corr, y)

importances = model.feature_importances_

fi = pd.DataFrame({
    "feature": features_corr,
    "importance": importances
}).sort_values("importance", ascending=False)

features_imp = fi.head(TOP_N_IMPORTANCE)["feature"].tolist()

X_imp = df_var[features_imp].values

print(f"[STEP 3] After importance filter: {len(features_imp)} features")


#final refinement
rfe_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    n_jobs=-1
)

rfe = RFE(rfe_model, n_features_to_select=RFE_FEATURES, step=50)
rfe.fit(X_imp, y)

features_final = [
    f for f, keep in zip(features_imp, rfe.support_) if keep
]

print(f"[STEP 4] Final selected features: {len(features_final)}")


#print as copypasteable lists (for use in train_and_save_XGBoost_model.py)
selected_fp = [f for f in features_final if f.startswith("ecfp4_")]
selected_maccs = [f for f in features_final if f.startswith("maccs")]
selected_desc = [f for f in features_final if f in desc_cols]

print("\n# ================= COPY INTO TRAINING SCRIPT =================")
print(f"fp_cols = {selected_fp}")
print(f"desc_cols = {selected_desc}")
print(f"maccs_cols = {selected_maccs}")


# also save CSV
pd.DataFrame({"feature": features_final}).to_csv("selected_features.csv", index=False)
print("\n[INFO] Saved selected_features.csv")