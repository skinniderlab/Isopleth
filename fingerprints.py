import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as fpg

def get_fingerprint(mol, fp_algo, fp_type, length, radius):
    """
    Generate a chemical fingerprint from a RDKit molecule object.

    Parameters:
    - mol: RDKit molecule object
    - fp_algo: type of fingerprint to generate; one of 'ecfp', 'fcfp'
    - fp_type: one of 'bits' (binary fingerprint) or 'counts' (count
      fingerprint)
    - length: length of the fingerprint (number of bit/counts)
    - radius: radius of the fingerprint
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
