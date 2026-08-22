"""
Fingerprint generation for chemical structures.
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as fpg

def get_fingerprint(mol, fp_algo, fp_type, length, radius):
    """
    Generate a fixed-length chemical fingerprint from an RDKit molecule object.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    fp_algo : {'ecfp', 'fcfp'}
        Fingerprint algorithm.
    fp_type : {'bits', 'counts'}
        Fingerprint representation.
    length : int
        Length of the fingerprint.
    radius : int
        Radius of the fingerprint.

    Returns
    -------
    numpy.ndarray
        One-dimensional `float32` array of length `length`.
    """
    fp_arr = np.zeros((length,), dtype=np.float32)

    try:
        if fp_algo == 'ecfp':
            # Morgan fingerprint (ECFP)
            generator = fpg.GetMorganGenerator(radius=int(radius),
                                               fpSize=int(length))
        elif fp_algo == 'fcfp':
            # Morgan fingerprint with feature invariants (FCFP)
            inv_gen = fpg.GetMorganFeatureAtomInvGen()
            generator = fpg.GetMorganGenerator(radius=int(radius),
                                               fpSize=int(length),
                                               atomInvariantsGenerator=inv_gen)
        else:
            raise ValueError(f"Unknown fingerprint algorithm: {fp_algo}")

        if fp_type == 'bits':
            fp = generator.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, fp_arr)
        elif fp_type == 'counts':
            fp = generator.GetCountFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, fp_arr)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")

    except Exception as e:
        smiles = Chem.MolToSmiles(mol)
        print(f"Warning: failed to generate fingerprint for mol with {smiles}: {e}")
        return fp_arr

    return fp_arr
