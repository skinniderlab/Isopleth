from rdkit import Chem
from tqdm import tqdm

def clean_mol(smiles, stereochem=False):
    """
    Construct a molecule from a SMILES string, removing stereochemistry and
    explicit hydrogens, and setting aromaticity.
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
    Construct a list of molecules from a list of SMILES strings, replacing
    invalid molecules with None in the list.
    """
    mols = []
    for smiles in tqdm(all_smiles):
        try:
            mol = clean_mol(smiles, stereochem)
            mols.append(mol)
        except ValueError:
            mols.append(None)
    return mols
