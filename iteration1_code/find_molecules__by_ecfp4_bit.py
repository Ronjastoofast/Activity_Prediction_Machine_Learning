"""
simple description
you specify a fingerprint 'bit' (for example, ecfp4_70)
it will find the compounds with that bit,
    save a .csv of the training data for those compounds
    generate an image of those compounds with the 'bit' highlighted

INPUTS:
ECFP4 bit ID (X for ECFP4_X, for example 70 for ECFP4_70)
training dataset (descriptors.csv from generate_molecular_descriptors.py)

OUTPUTS
image of molecules containing the bit (needs to be manually saved)
compounds_with_ecfp_X.csv, which has the row in descriptors.csv for compounds containing ECFP4_X

define the inputs in if __name__ == "__main__":
"""

from typing import List, Sequence, Tuple, Optional, Dict

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw




def _get_ecfp4_bit_substructures(
    mol: Chem.Mol,
    bit_id: int,
    radius: int = 2,
    nBits: int = 2048,
) -> List[Chem.Mol]:

    #For a single RDKit Mol and ECFP4 bit, return submols corresponding to that bit.

    bitInfo = {}
    _ = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=nBits, bitInfo=bitInfo
    )

    if bit_id not in bitInfo:
        return []

    submols: List[Chem.Mol] = []
    for center_atom_idx, rad_used in bitInfo[bit_id]:
        env_bond_indices = Chem.FindAtomEnvironmentOfRadiusN(
            mol, rad_used, center_atom_idx
        )

        if env_bond_indices:
            submol = Chem.PathToSubmol(mol, env_bond_indices)
            submols.append(submol)
        else:
            # radius 0 environment: just a single atom
            rw = Chem.RWMol()
            at = mol.GetAtomWithIdx(center_atom_idx)
            rw.AddAtom(Chem.Atom(at.GetAtomicNum()))
            submols.append(rw.GetMol())

    return submols


#find molecules with ecfp4 bit

def find_smiles_with_bit(
    df: pd.DataFrame,
    bit_id: int,
    smiles_col: str = "smiles",
    bit_prefix: str = "ecfp4_",
) -> List[str]:

    bit_col = f"{bit_prefix}{bit_id}"
    if bit_col not in df.columns:
        raise ValueError(f"Column '{bit_col}' not found in DataFrame.")

    subdf = df[df[bit_col] == 1]
    return subdf[smiles_col].dropna().unique().tolist()


# highlight bit substructure on dataset molecules

def highlight_bit_on_dataset(
    df: pd.DataFrame,
    bit_id: int,
    smiles_col: str = "smiles",
    bit_prefix: str = "ecfp4_",
    radius: int = 2,
    nBits: int = 2048,
    max_mols: int = 16,
    molsPerRow: int = 4,
    subImgSize: Tuple[int, int] = (300, 300),
):

    smiles_list = find_smiles_with_bit(df, bit_id, smiles_col=smiles_col, bit_prefix=bit_prefix)
    if not smiles_list:
        raise ValueError(f"No molecules have bit {bit_id} set.")

    smiles_list = smiles_list[:max_mols]

    mols: List[Chem.Mol] = []
    highlight_atom_lists: List[List[int]] = []
    legends: List[str] = []



    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        bitInfo: Dict[int, List[Tuple[int, int]]] = {}
        _ = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=nBits, bitInfo=bitInfo
        )
        if bit_id not in bitInfo:
            # should not happen if df bit column is consistent, but be safe
            continue

        atom_indices_for_bit = set()
        for center_atom_idx, rad_used in bitInfo[bit_id]:
            env_bond_indices = Chem.FindAtomEnvironmentOfRadiusN(
                mol, rad_used, center_atom_idx
            )
            if env_bond_indices:
                for bidx in env_bond_indices:
                    bond = mol.GetBondWithIdx(bidx)
                    atom_indices_for_bit.add(bond.GetBeginAtomIdx())
                    atom_indices_for_bit.add(bond.GetEndAtomIdx())
            else:
                atom_indices_for_bit.add(center_atom_idx)

        mols.append(mol)
        highlight_atom_lists.append(sorted(atom_indices_for_bit))

        match = df.loc[df[smiles_col] == smi, "IC50"]

        if len(match) > 0 and pd.notna(match.values[0]):
            legends.append(f"{smi}\nIC50={match.values[0]:.2f}")
        else:
            legends.append(smi)

    if not mols:
        raise ValueError(f"No valid molecules to draw for bit {bit_id}.")

    img = Draw.MolsToGridImage(
        mols,
        highlightAtomLists=highlight_atom_lists,
        legends=legends,
        molsPerRow=molsPerRow,
        subImgSize=subImgSize,
    )

    # save a csv of all compounds with that ecfp4 bit
    # Extract matching rows
    bit_col = f"{bit_prefix}{bit_id}"
    subdf = df[df[bit_col] == 1].copy()

    # Save CSV
    output_name = f"compounds_with_ecfp_{bit_id}.csv"
    subdf.to_csv(output_name, index=False)

    print(f"[INFO] Saved {len(subdf)} compounds to {output_name}")

    return img


#draw isolated substructures using df

def draw_bit_substructures_from_dataset(
    df: pd.DataFrame,
    bit_id: int,
    smiles_col: str = "smiles",
    bit_prefix: str = "ecfp4_",
    radius: int = 2,
    nBits: int = 2048,
    max_examples: int = 4,
    molsPerRow: int = 4,
    subImgSize: Tuple[int, int] = (250, 250),
):

    smiles_list = find_smiles_with_bit(df, bit_id, smiles_col=smiles_col, bit_prefix=bit_prefix)
    if not smiles_list:
        raise ValueError(f"No molecules have bit {bit_id} set.")

    smiles_list = smiles_list[:max_examples]

    all_submols: List[Chem.Mol] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        submols = _get_ecfp4_bit_substructures(mol, bit_id, radius=radius, nBits=nBits)
        all_submols.extend(submols)

    if not all_submols:
        raise ValueError(f"No substructures could be extracted for bit {bit_id}.")

    img = Draw.MolsToGridImage(
        all_submols,
        molsPerRow=molsPerRow,
        subImgSize=subImgSize,
    )

    #save a csv of all compounds with that ecfp4 bit
    # Extract matching rows
    bit_col = f"{bit_prefix}{bit_id}"
    subdf = df[df[bit_col] == 1].copy()

    # Save CSV
    output_name = f"compounds_with_ecfp_{bit_id}.csv"
    subdf.to_csv(output_name, index=False)

    print(f"[INFO] Saved {len(subdf)} compounds to {output_name}")

    return img


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("descriptors.csv")

    bit_id = 70  # change this depending on the bit of interest (look at plot_feature_importance)

    img = highlight_bit_on_dataset(df, bit_id)
    img.show()