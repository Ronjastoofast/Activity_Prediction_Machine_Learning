"""
simple description:
generates molecular descriptors from SMILES and saves as descriptors.py
    -RDKit molecular descriptors
    -Morgan Fingerprints (ECFP4_X)
    -MACCS Keys (a set of 167 predefined substructure features commonly used in chemoinformatics)



INPUTS
ic50_results.csv (via INPUT_CSV)
with columns: IC50 and smiles
(and optionally can select which descriptors to use, from RDKit)

OUTPUTS descriptors.csv
with SMILES, IC50, and all of the descriptors generated


"""


import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Chem import rdMolDescriptors

#INPUTS

INPUT_CSV = "ic50_results.csv"     # input file with at least a 'smiles' column
SMILES_COLUMN = "smiles"           # name of the SMILES column
OUTPUT_CSV = "descriptors.csv"     # where to save the output

# Morgan fingerprint settings
ECFP_RADIUS = 2#3                    # ECFP4 = radius 2. using 3 made the model worse....
ECFP_NBITS = 2048

#SPECIFY RDKit Descriptors
#Standard set
descriptor_funcs = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "TPSA": Descriptors.TPSA,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "FractionCSP3": Descriptors.FractionCSP3,

#extra descriptors
#Composition / size
    "HeavyAtomCount": rdMolDescriptors.CalcNumHeavyAtoms,
    "NumHeteroAtoms": rdMolDescriptors.CalcNumHeteroatoms,
    "NumAtoms": lambda m: m.GetNumAtoms(),
    "NumAromaticAtoms": lambda m: sum(1 for a in m.GetAtoms() if a.GetIsAromatic()),
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,

#Hydrogen-bond / heteroatom counts
    "NHOHCount": rdMolDescriptors.CalcNumHeteroatoms,  # common QSAR proxy
    "NOCount": lambda m: sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 7),

#Topological descriptors
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
    "HallKierAlpha": Descriptors.HallKierAlpha,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "Kappa3": Descriptors.Kappa3,

#PEOE charge × surface descriptors
    "PEOE_VSA1": Descriptors.PEOE_VSA1,
    "PEOE_VSA2": Descriptors.PEOE_VSA2,
    "PEOE_VSA3": Descriptors.PEOE_VSA3,
    "PEOE_VSA4": Descriptors.PEOE_VSA4,
    "PEOE_VSA5": Descriptors.PEOE_VSA5,
    "PEOE_VSA6": Descriptors.PEOE_VSA6,

#SlogP × surface descriptors
    "SlogP_VSA1": Descriptors.SlogP_VSA1,
    "SlogP_VSA2": Descriptors.SlogP_VSA2,
    "SlogP_VSA3": Descriptors.SlogP_VSA3,
    "SlogP_VSA4": Descriptors.SlogP_VSA4,
    "SlogP_VSA5": Descriptors.SlogP_VSA5,
    "SlogP_VSA6": Descriptors.SlogP_VSA6,

#EState-based surface descriptors
    "EState_VSA1": Descriptors.EState_VSA1,
    "EState_VSA2": Descriptors.EState_VSA2,
    "VSA_EState1": Descriptors.VSA_EState1,
    "VSA_EState2": Descriptors.VSA_EState2,
}


def smiles_to_descriptors(smi: str) -> dict | None:
    #Convert a single SMILES to descriptor dict (or None if invalid)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"[WARN] Invalid SMILES, skipping: {smi}")
        return None

    row = {SMILES_COLUMN: smi}

    # Physchem descriptors
    for name, func in descriptor_funcs.items():
        try:
            row[name] = func(mol)
        except Exception:
            row[name] = None

    # Morgan fingerprint (ECFP4)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=ECFP_RADIUS, nBits=ECFP_NBITS)
    ecfp_bits = ecfp.ToBitString()  # string of '0'/'1'
    for i, bit in enumerate(ecfp_bits):
        row[f"ecfp4_{i}"] = int(bit)

    # MACCS keys (167 bits)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_bits = maccs.ToBitString()
    for i, bit in enumerate(maccs_bits):
        row[f"maccs_{i}"] = int(bit)

    return row


def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    if SMILES_COLUMN not in df.columns:
        raise ValueError(f"Column '{SMILES_COLUMN}' not found in {INPUT_CSV}")

    # One row per unique SMILES
    smiles_list = df[SMILES_COLUMN].dropna().unique()
    print(f"[INFO] Found {len(smiles_list)} unique SMILES")

    rows = []
    for i, smi in enumerate(smiles_list, start=1):
        d = smiles_to_descriptors(smi)
        if d is not None:
            rows.append(d)

        if i % 100 == 0:
            print(f"[INFO] Processed {i} molecules...")

    desc_df = pd.DataFrame(rows)

    # If the input CSV already contains an IC50 (or pIC50) column, merge it in
    for activity_col in ["IC50", "pIC50", "activity"]:
        if activity_col in df.columns:
            print(f"[INFO] Merging activity column '{activity_col}' into descriptor table")
            activity = df[[SMILES_COLUMN, activity_col]].drop_duplicates(SMILES_COLUMN)
            desc_df = desc_df.merge(activity, on=SMILES_COLUMN, how="left")
            break  # only merge the first one we find

    desc_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved descriptors to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

