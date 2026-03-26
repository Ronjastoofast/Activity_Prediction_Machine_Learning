"""
simple description:
takes in raw data csv
needs column names: concentration, value, smiles
concentration must be >10

INPUTS:
EOS_data.csv

OUTPUTS
calculates IC50 and saves it in
ic50_results.csv
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
#INPUT raw data csv name below
df = pd.read_csv("EOS_data.csv")

def hill_4p(x, bottom, top, logIC50, hill):
    return bottom + (top - bottom) / (1.0 + 10.0**((logIC50 - np.log10(x)) * hill))

def fit_ic50_for_group(group: pd.DataFrame) -> pd.Series:
    # x = concentration, y = response/value
    x = group["concentration"].values.astype(float)
    y = group["value"].values.astype(float)

    # Drop NaNs and non-positive concentrations (log10 problem)
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (x > 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        # not enough points to fit a curve
        return pd.Series({
            "IC50": np.nan,
            "logIC50": np.nan,
            "bottom": np.nan,
            "top": np.nan,
            "hill": np.nan,
            "fit_success": False
        })

    # Sort by concentration (helps stability)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Initial guesses
    bottom0 = float(y.min())
    top0    = float(y.max())
    logIC50_0 = float(np.log10(np.median(x)))
    hill0   = 1.0

    p0 = [bottom0, top0, logIC50_0, hill0]

    # Bounds: keep things sane
    # bottom, top in [min(y)-10, max(y)+10], logIC50 anywhere, hill >= 0.1
    bounds = (
        [bottom0 - 10, bottom0 - 10, -10, 0.1],  # lower
        [top0 + 10,    top0 + 10,    10,  5.0],  # upper
    )

    try:
        popt, pcov = curve_fit(
            hill_4p, x, y,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        bottom, top, logIC50, hill = popt
        IC50 = 10**logIC50

        # Optional: simple R²
        y_pred = hill_4p(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

        return pd.Series({
            "IC50": IC50,
            "logIC50": logIC50,
            "bottom": bottom,
            "top": top,
            "hill": hill,
            "r2": r2,
            "fit_success": True
        })

    except Exception:
        # Fit failed for this compound
        return pd.Series({
            "IC50": np.nan,
            "logIC50": np.nan,
            "bottom": np.nan,
            "top": np.nan,
            "hill": np.nan,
            "r2": np.nan,
            "fit_success": False
        })


ic50_df = (
    df.groupby("smiles", as_index=False)
      .apply(fit_ic50_for_group)
      .reset_index(drop=True)
)
ic50_df.to_csv("ic50_results.csv", index=False)

ic50_good = ic50_df[ic50_df["fit_success"] & (ic50_df["r2"] > 0.8)]

