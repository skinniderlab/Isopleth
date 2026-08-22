"""
Utility functions for parsing and standardizing chemical structures from SMILES
"""

from rdkit import Chem
from tqdm import tqdm

def clean_mol(smiles, stereochem=False):
    """
    Convert a SMILES string to a standardized RDKit molecule.

    Parameters
    ----------
    smiles : str
        SMILES representation of a candidate or reference structure.
    stereochem : bool, default=False
        If False, remove stereochemistry from molecule. 

    Returns
    -------
    rdkit.Chem.Mol
        Sanitized molecule with hydrogens removed and aromaticity set.

    Raises
    ------
    ValueError
        If RDKit cannot parse the input SMILES string.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("invalid SMILES: " + str(smiles))
    if not stereochem:
        Chem.RemoveStereochemistry(mol)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    return mol

def clean_mols(all_smiles, stereochem=False):
    """
    Convert a list of SMILES strings to standardized RDKit molecules.
   
    Parameters
    ----------
    all_smiles : iterable of str
        SMILES strings to parse.
    stereochem : bool, default=False
        Passed to clean_mol()

    Returns
    -------
    list
        RDKit molecule objects, with None in positions where parsing failed.
    """
    mols = []
    for smiles in tqdm(all_smiles):
        try:
            mol = clean_mol(smiles, stereochem)
            mols.append(mol)
        except ValueError:
            mols.append(None)
    return mols
