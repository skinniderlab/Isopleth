"""
Evaluate isopleth model predictions on held-out labelling patterns.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.DataStructs import FingerprintSimilarity
from tqdm import tqdm

# add script directory to path so sibling modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions import clean_mol, clean_mols

# suppress rdkit errors
rdBase.DisableLog('rdApp.error')

# CLI
parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, required=True)
parser.add_argument('--pred_file', type=str, required=True)
parser.add_argument('--eval_file', type=str, required=True)
args = parser.parse_args()
print(args)

# create output directory
output_dir = os.path.dirname(args.eval_file)
if output_dir and not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# read ground truth and predictions
test_meta = pd.read_csv(args.test_file)
print(f'loaded {len(test_meta)} test samples')
preds = pd.read_csv(args.pred_file)
print(f'loaded {len(preds)} predictions')

unique_peak_ids = preds['peak_id'].unique()
print(f'evaluating {len(unique_peak_ids)} unique peaks')

results = []
for peak_id in tqdm(unique_peak_ids, desc='evaluating'):
    # get ground truth
    true_rows = test_meta[test_meta['peak_id'] == peak_id]
    assert len(true_rows) == 1, \
        f'expected 1 ground truth row for peak_id {peak_id}, ' \
        f'got {len(true_rows)}'
    true_row = true_rows.iloc[0]
    true_smiles = true_row['canonical_smiles']
    true_inchikey = true_row['InChIKey']
    true_inchi14 = true_inchikey.split('-')[0]
    
    true_mol = clean_mol(true_smiles)
    assert true_mol, \
        f'could not parse ground truth SMILES for peak_id {peak_id}'
    true_fp = Chem.RDKFingerprint(true_mol)
    
    # get candidates sorted by cosine similarity
    peak_preds = preds[preds['peak_id'] == peak_id].sort_values(
        'cosine', ascending=False).reset_index(drop=True)
    
    # compute Tanimoto coefficients and InChIKeys for all candidates
    cand_mols = clean_mols(peak_preds['candidate_smiles'].values)
    tcs = []
    cand_inchi14s = []
    for mol in cand_mols:
        if mol is not None:
            cand_fp = Chem.RDKFingerprint(mol)
            tc = FingerprintSimilarity(true_fp, cand_fp)
            cand_inchikey = Chem.inchi.MolToInchiKey(mol)
            cand_inchi14 = cand_inchikey.split('-')[0]
        else:
            tc = -1
            cand_inchi14 = None
        tcs.append(tc)
        cand_inchi14s.append(cand_inchi14)
    
    peak_preds = peak_preds.assign(tc=tcs, cand_inchi14=cand_inchi14s)
    
    # rank of correct structure
    inchi_matches = peak_preds['cand_inchi14'] == true_inchi14
    rank_inchi = inchi_matches.idxmax() + 1 if inchi_matches.any() \
        else np.inf
    
    # top-1 Tanimoto coefficient
    top1_tc = peak_preds.iloc[0]['tc'] if len(peak_preds) > 0 else np.nan
    
    results.append({
        'peak_id': peak_id,
        'rank_inchi': rank_inchi,
        'top1_tc': top1_tc,
        'n_candidates': len(peak_preds),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(args.eval_file, index=False)
print(f'wrote evaluation results for {len(results_df)} peaks to '
      f'{args.eval_file}')
