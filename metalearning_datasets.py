"""Utility functions to prepare the datasets used in Isopleth meta-learning."""

import numpy as np
import pandas as pd

# list of meta-learning features
meta_learning_features = [
    'structure_confidence',
    'structure_size',
    'structure_rank',
    'ppm_error',
    'delta_RT',
    'dot_product',
    'n_overlap',
    'cosine'
]

def _standardize_candidate_columns(df):
    """
    Standardize candidate structure column names.
    """
    rename = {}
    if 'candidate_smiles' not in df.columns and 'SMILES' in df.columns:
        rename['SMILES'] = 'candidate_smiles'
    if 'candidate_inchikey' not in df.columns and 'InChIKey' in df.columns:
        rename['InChIKey'] = 'candidate_inchikey'
    return df.rename(columns=rename)

def _read_cosines(cosine_file):
    """
    Load Isopleth cosine predictions.
    """
    cosines = pd.read_csv(cosine_file)
    missing = [c for c in ['peak_id', 'candidate_smiles', 'cosine']
               if c not in cosines.columns]
    if missing:
        raise ValueError(f'missing columns in cosine file {cosine_file}: '
                         f'{missing}')
    
    cosines = cosines[['peak_id', 'candidate_smiles', 'cosine']]
    cosines['peak_id'] = cosines['peak_id'].astype(str)
    cosines = cosines.drop_duplicates()
    
    key_cols = ['peak_id', 'candidate_smiles']
    dup = cosines.duplicated(key_cols, keep=False)
    if dup.any():
        examples = cosines.loc[dup, key_cols].head(5).to_dict('records')
        raise ValueError(f'cosine file {cosine_file} has '
                         f'duplicate candidate rows, e.g. {examples}')
    return cosines

def _merge_cosines(feat, cosine_file, name):
    """
    Merge Isopleth cosine scores into a feature table.
    """
    if 'candidate_smiles' not in feat.columns:
        raise ValueError(f'{name} feature table needs candidate_smiles col '
                         'to merge cosine values')
    
    feat = feat.drop(columns=['cosine'], errors='ignore')
    before = len(feat)
    cosines = _read_cosines(cosine_file)
    feat = feat.merge(cosines, on=['peak_id', 'candidate_smiles'],
                      how='left')
    if len(feat) != before:
        raise ValueError(f'merging {name} cosines changed row count from '
                         f'{before} to {len(feat)}')
    
    n_missing = feat['cosine'].isna().sum()
    if n_missing:
        examples = (
            feat.loc[feat['cosine'].isna(),
                     ['peak_id', 'candidate_smiles']]
            .head(5)
            .to_dict('records'))
        raise ValueError(f'{n_missing} {name} rows are missing Isopleth '
                         f'cosines after merging {cosine_file}; examples: '
                         f'{examples}')
    return feat

class MetaLearningDataset():
    """
    Dataset of candidate-level features for meta-learning.
    """

    def __init__(self, train_feat_file, test_feat_file, train_cosine_file,
                 test_cosine_file):
        """
        Load training and test feature tables.
        """
        self.train_feat_file = train_feat_file
        self.test_feat_file = test_feat_file
        self.train_cosine_file = train_cosine_file
        self.test_cosine_file = test_cosine_file
        
        # load precomputed candidate feature tables
        train_feat = _standardize_candidate_columns(
            pd.read_csv(self.train_feat_file))
        test_feat = _standardize_candidate_columns(
            pd.read_csv(self.test_feat_file))
        
        train_feat['peak_id'] = train_feat['peak_id'].astype(str)
        test_feat['peak_id'] = test_feat['peak_id'].astype(str)

        # drop peaks for which there is no ground-truth structure
        nan_peaks = (
            train_feat.groupby('peak_id')['match']
            .apply(lambda x: x.isna().all()))
        nan_peaks = nan_peaks[nan_peaks].index
        train_feat = train_feat[~train_feat['peak_id'].isin(nan_peaks)]

        train_feat = _merge_cosines(train_feat, self.train_cosine_file,
                                    'train')
        test_feat = _merge_cosines(test_feat, self.test_cosine_file,
                                   'test')
        
        missing = [f for f in ['peak_id', 'match'] + meta_learning_features
                   if f not in train_feat.columns]
        if missing:
            raise ValueError(f'missing columns in train_feat_file: {missing}')
        
        missing = [f for f in ['peak_id'] + meta_learning_features
                   if f not in test_feat.columns]
        if missing:
            raise ValueError(f'missing columns in test_feat_file: {missing}')

        # replace -inf with 0 in n_overlap (occurs when exp MS2 is empty)
        train_feat['n_overlap'] = train_feat['n_overlap'].replace(-np.inf, 0)
        test_feat['n_overlap'] = test_feat['n_overlap'].replace(-np.inf, 0)

        self.train_feat = train_feat
        self.test_feat = test_feat
        print(f'loaded {len(self.train_feat)} train and {len(self.test_feat)} '
              f'test candidate rows for '
              f'{self.train_feat["peak_id"].nunique()} and '
              f'{self.test_feat["peak_id"].nunique()} peaks')

    def get_train(self):
        """
        Return training features and binary labels.
        
        Returns
        -------
        tuple of numpy.ndarray
            `X_train` with shape `(n_train_candidates, n_features)` and
            `y_train` with shape `(n_train_candidates,)`.
        """
        X_train = self.train_feat[meta_learning_features].values
        y_train = self.train_feat['match'].values.astype(int)
        return X_train, y_train
    
    def get_test(self):
        """
        Return candidate features to score.
        
        Returns
        -------
        numpy.ndarray
            `X_test` with shape `(n_test_candidates, n_features)`.
        """
        return self.test_feat[meta_learning_features].values
