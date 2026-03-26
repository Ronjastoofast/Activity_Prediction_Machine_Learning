"""
plots an IC50 curve (with the experimental datapoints) for a given SMILES code

INPUTS: under INPUTS
raw data csv
SMILES code for compound of interest (for example, from the output .csv of find_molecules_by_ecfp4_bit.py)

OUTPUTS:
shows an image of the IC50 curve (needs manual saving!)
prints the IC50
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def hill_4p(x, bottom, top, logIC50, hill):
    return bottom + (top - bottom) / (1.0 + 10.0**((logIC50 - np.log10(x)) * hill))

def plot_ic50_curve(df: pd.DataFrame, smiles: str):
    # Filter for the compound
    group = df[df["smiles"] == smiles].copy()

    if group.empty:
        print(f"No data found for SMILES: {smiles}")
        return

    # Extract x and y
    x = group["concentration"].values.astype(float)
    y = group["value"].values.astype(float)

    # Clean data
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (x > 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        print("Not enough points to fit curve.")
        return

    # Sort for plotting
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Initial guesses
    bottom0 = float(y.min())
    top0 = float(y.max())
    logIC50_0 = float(np.log10(np.median(x)))
    hill0 = 1.0

    p0 = [bottom0, top0, logIC50_0, hill0]

    bounds = (
        [bottom0 - 10, bottom0 - 10, -10, 0.1],
        [top0 + 10, top0 + 10, 10, 5.0],
    )

    try:
        popt, _ = curve_fit(
            hill_4p, x, y,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )

        bottom, top, logIC50, hill = popt
        IC50 = 10**logIC50

        # Generate smooth curve
        x_fit = np.logspace(np.log10(min(x)), np.log10(max(x)), 200)
        y_fit = hill_4p(x_fit, *popt)

        # Plot
        plt.figure(figsize=(6, 5))

        # Raw data points
        plt.scatter(x, y, color="black", label="Data")

        # Fitted curve
        plt.plot(x_fit, y_fit, color="red", label="4PL fit")

        # IC50 line
        plt.axvline(IC50, linestyle="--", color="blue", label=f"IC50 = {IC50:.3g}")

        plt.xscale("log")
        plt.xlabel("Concentration")
        plt.ylabel("Response")
        plt.title("Dose–response curve")
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"IC50 = {IC50:.4g}")

    except Exception as e:
        print("Fit failed:", e)

#INPUTS
df = pd.read_csv("EOS_data.csv")

plot_ic50_curve(df, "CCc1nc2c(C)cc(N3CCN(CC(=O)N4CC(O)C4)CC3)cn2c1N(C)c1nc(-c2ccc(F)cc2)c(C#N)s1")  # replace with your SMILES