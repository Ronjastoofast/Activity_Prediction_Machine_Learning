"""
simple description:
plots importance of features in pIC50 prediction by XGBoost model
    - from all columns
    - and excluding the morgan fingerprint (ecfp4_X) features

INPUTS:
training dataset
model
number of top features to plot

OUPUTS:
shows images of plots of the top features and their importance
(plots need to be manually saved)
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt

#INPUTS: csv used for training/testing, saved model, N for N most important features
DESCRIPTORS_CSV = "descriptors.csv" # from generate_molecular_descriptors.py
MODEL_PATH = "xgboost_pIC50_model_tuned2.joblib"   # from train_and_save_XGBoost_model.py
TOP_N = 20

#Load data and model
df = pd.read_csv(DESCRIPTORS_CSV)
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_cols = bundle["feature_cols"]

#Filter invalid IC50 if needed (must match training)
df = df[df["IC50"] > 0].copy()

#Build feature list exactly as in training
fp_cols = [c for c in df.columns if c.startswith("ecfp4_")]
maccs_cols = [c for c in df.columns if c.startswith("maccs")]
#hardcoded desc cols, copied from train_and_save_XGBoost_model.py
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

#keep only descriptors that actually exist in the CSV
desc_cols = [c for c in desc_cols if c in df.columns]

feature_cols = bundle["feature_cols"]
X = df[feature_cols].values


#Get feature importances from XGBoost

importances = model.feature_importances_
if len(importances) != len(feature_cols):
    raise ValueError(
        f"Model has {len(importances)} importances but we built {len(feature_cols)} features. "
        "Feature ordering or preprocessing mismatch."
    )

fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
})

# Normalize to sum to 1 (optional but nice)
fi["importance"] = fi["importance"] / fi["importance"].sum()


# Plot: top N overall features

fi_top = fi.sort_values("importance", ascending=False).head(TOP_N)
fi_top = fi_top.iloc[::-1]  # reverse for nicer horizontal bar order

plt.figure(figsize=(8, 6))
plt.barh(fi_top["feature"], fi_top["importance"])
plt.xlabel("Relative importance")
plt.title(f"Top {TOP_N} feature importances (XGBoost)")
plt.tight_layout()
plt.show()


# Plot: top N RDKit descriptors only (desc_cols)

fi_desc = fi[fi["feature"].isin(desc_cols)].copy()
fi_desc = fi_desc.sort_values("importance", ascending=False).head(TOP_N)
fi_desc = fi_desc.iloc[::-1]

plt.figure(figsize=(8, 6))
plt.barh(fi_desc["feature"], fi_desc["importance"])
plt.xlabel("Relative importance")
plt.title(f"Top {TOP_N} RDKit descriptor importances (XGBoost)")
plt.tight_layout()
plt.show()

print("\nTop descriptor importances:")
print(fi_desc.sort_values("importance", ascending=False))

# EXPORT: copy-pasteable feature list for retraining
TOP_FEATURES_FOR_MODEL = fi.sort_values("importance", ascending=False)["feature"].head(TOP_N).tolist()

print("list of top features for use in training new models")
print(f"selected_features = {TOP_FEATURES_FOR_MODEL}")