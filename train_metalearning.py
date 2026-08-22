"""
Train the meta-learning model and score cnadidate structures

The meta-learning model is a random forest classifier that scores
DeepMet-generated candidate structures for each peak, using a combination of 
features derived from DeepMet itself, predicted MS/MS spectra and retention 
times for each candidate, and Isopleth cosine similarities between candidate
structures and the peak's isotopic labelling pattern. All of these features 
are computed in a leakage-free manner for each candidate structure, i.e., 
using models (DeepMet, FraGNNet, retention time prediction, or Isopleth) that 
had not seen the structure in question within their training sets. 
The feature table itself is provided in data/, and preprocessing code used to
assemble the feature table is provided in preprocessing/. 

Input files
-----------
train_feat_file
    .csv matrix of training candidates. Requires a `peak_id` column; a
    `match` column giving the binary ground-truth label; the DeepMet scores 
    (`structure_confidence`, `structure_size`, `structure_rank`, `ppm_error`);
    and the MS/MS or RT derived features `delta_RT`, `dot_product`, and 
    `n_overlap`.
test_feat_file
    .csv matrix of candidates to score. Requires the same feature columns as
    `train_feat_file`.
train_cosine_file
    .csv file with Isopleth predictions for training candidates. Requires
    `peak_id`, `candidate_smiles`, and `cosine`. The cosine values are merged
    into `train_feat_file` by `peak_id` and `candidate_smiles`.
test_cosine_file
    .csv file with Isopleth predictions for test candidates. Requires
    `peak_id`, `candidate_smiles`, and `cosine`. The cosine values are merged
    into `test_feat_file` by `peak_id` and `candidate_smiles`.

Output files
------------
pred_file
    Candidate rankings with `peak_id`, `candidate_smiles`,
    `candidate_inchikey`, `cosine`, `ml_prob`, and `ml_rank`.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# add script directory to path so sibling modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metalearning_datasets import MetaLearningDataset

# CLI
parser = argparse.ArgumentParser()
parser.add_argument('--train_feat_file', type=str, required=True)
parser.add_argument('--test_feat_file', type=str, required=True)
parser.add_argument('--train_cosine_file', type=str, required=True)
parser.add_argument('--test_cosine_file', type=str, required=True)
parser.add_argument('--pred_file', type=str, required=True)
parser.add_argument('--coef_file', type=str, required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_estimators', type=int, default=100)
args = parser.parse_args()
print(args)

# create output directory 
for output_file in [args.pred_file, args.coef_file]:
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

# create dataset
dataset = MetaLearningDataset(
    train_feat_file=args.train_feat_file,
    test_feat_file=args.test_feat_file,
    train_cosine_file=args.train_cosine_file,
    test_cosine_file=args.test_cosine_file,
)
X_train, y_train = dataset.get_train()
X_test = dataset.get_test()

# RF pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median',
                              keep_empty_features=True)),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=args.n_estimators,
                                          random_state=args.seed)),
])
pipeline.fit(X_train, y_train)

# predict and rank candidates within each peak
test_feat = dataset.test_feat
test_feat['ml_prob'] = pipeline.predict_proba(X_test)[:, 1]
test_feat['ml_rank'] = (
    test_feat.groupby('peak_id')['ml_prob']
    .rank(ascending=False, method='min')
    .astype(int))

pred_cols = ['peak_id', 'candidate_smiles', 'candidate_inchikey',
             'cosine', 'ml_prob', 'ml_rank', 'match']
test_feat[[c for c in pred_cols if c in test_feat.columns]].sort_values(
    ['peak_id', 'ml_rank']).to_csv(args.pred_file, index=False)
