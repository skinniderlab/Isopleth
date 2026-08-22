"""
Dataset class and functions used to train Isopleth

This file converts paired training data (isotopic labelling patterns for peaks
corresponding to known metabolites and chemical fingerprints for those
metabolites' structures) into tensors for input to the contrastive learning
model.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fingerprints import get_fingerprint
from functions import clean_mols

def collate_fn(batch):
    """
    Custom collate function for pattern-fingerprint pairs.

    Parameters
    ----------
    batch : list of tuple
        Items returned by `IsotopicPatternDataset.__getitem__`. Each item is
        `(pattern, true_fp)`, where `pattern` has shape `(num_labels,)` and
        `true_fp` has shape `(fp_length,)`.

    Returns
    -------
    tuple of torch.Tensor
        `patterns` with shape `(batch_size, num_labels)` and `true_fps`
        with shape `(batch_size, fp_length)`.
    """
    patterns, true_fps = zip(*batch)
    return torch.stack(patterns), torch.stack(true_fps)

class IsotopicPatternDataset(Dataset):
    """
    Dataset of isotopic labelling patterns paired with ground-truth structures.

    Each item represents one peak and contains an observed labelling
    pattern paired together with the fingerprint of that peak's ground-truth
    chemical structure.

    Input files
    -----------
    label_file : str
        matrix in csv format with peak_ids as the row index and isotopologue-level 
        labelling pattern features (i.e., specific numbers of carbon atoms
        from each tracer administered to mice; for example, 13C2-Tau) as 
        columns.
    struct_file : str
        structure metadata table in csv format with one row per peak. Rows 
        must be in the same order as `label_file` and must include 
        `canonical_smiles` and `InChIKey` columns.

    Items
    -----
    pattern : torch.Tensor
        Labelling-pattern vector with shape `(num_labels,)`.
    true_fp : torch.Tensor
        Ground-truth structure fingerprint with shape `(fp_length,)`.
    """
    
    def __init__(self, config, label_file, struct_file, device,
                 val_split=0.1, seed=0):
        """
        Load paired training data and create train/validation splits.

        Parameters
        ----------
        config : Config
            Configuration containing the fingerprint settings.
        label_file : str
            Path to the labelling-pattern .csv matrix.
        struct_file : str
            Path to the ground-truth structure metadata .csv
        device : torch.device
            Device on which returned tensors are stored.
        val_split : float, default=0.1
            Validation split fraction.
        seed : int, default=0
            Random seed used for the train/validation split and validation batch
            sampling.
        """
        self.config = config
        self.label_file = label_file
        self.struct_file = struct_file
        self.device = device
        self.val_split = val_split
        self.seed = seed
        
        # load labelling patterns
        label_df = pd.read_csv(self.label_file, index_col=0)
        self.label_df = label_df
        print('loaded {}x{} labelling matrix'.format(label_df.shape[0],
                                                     label_df.shape[1]))
        
        # load ground-truth structures for the labelling patterns
        struct_df = pd.read_csv(self.struct_file)
        struct_df.index = label_df.index
        # remove any rows with invalid SMILES and sync label_df
        valid_idx = struct_df.dropna(subset=['canonical_smiles']).index
        n_dropped = len(struct_df) - len(valid_idx)
        if n_dropped > 0:
            print(f'WARNING: dropping {n_dropped} peaks with missing '
                  f'ground-truth SMILES')
        self.struct_df = struct_df.loc[valid_idx]
        self.label_df = self.label_df.loc[valid_idx]
        print('loaded {} ground-truth structures ({} unique)'.format(
            self.struct_df.shape[0], self.struct_df['InChIKey'].nunique()))
        
        # compute fingerprints for ground-truth structures
        print('computing fingerprints for {} ground-truth structures...'.format(
            self.struct_df.shape[0]))
        struct_fps = self.get_fingerprint_matrix(
            self.struct_df['canonical_smiles'].values)
        struct_fps.index = self.struct_df.index
        self.struct_fps = struct_fps
        
        # build list of (pattern, fingerprint) items
        all_items = []
        for i in range(len(self.label_df)):
            pattern = self.label_df.iloc[i].values.astype(np.float32)
            true_fp = self.struct_fps.iloc[i].values.astype(np.float32)
            all_items.append((
                torch.from_numpy(pattern).to(device),
                torch.from_numpy(true_fp).to(device),
            ))
        
        # split into train and validation sets
        rng = np.random.default_rng(self.seed)
        n_total = len(all_items)
        n_val = int(n_total * self.val_split)
        indices = rng.permutation(n_total)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        self.train_items = [all_items[i] for i in train_indices]
        self.val_items = [all_items[i] for i in val_indices]
        print(f'split into {len(self.train_items)} train and '
              f'{len(self.val_items)} validation items')
        
        # seed RNG for validation set
        self._val_rng = torch.Generator()
        self._val_rng.manual_seed(self.seed)
    
    def get_fingerprint_matrix(self, smiles_list):
        """
        Compute fingerprints for a list of SMILES strings.

        Parameters
        ----------
        smiles_list : iterable of str
            SMILES strings for the ground-truth structures.

        Returns
        -------
        pandas.DataFrame
            Fingerprint matrix with one row per SMILES string and `fp_length`
            columns.
        """
        mols = clean_mols(smiles_list)
        fps = np.vstack([
            get_fingerprint(mol=mol,
                            fp_algo=self.config.get_param('fp_algo'),
                            fp_type=self.config.get_param('fp_type'),
                            length=self.config.get_param('fp_length'),
                            radius=self.config.get_param('fp_radius'))
            for mol in mols
        ])
        fp_cols = [f"bit_{i}" for i in range(self.config.get_param('fp_length'))]
        return pd.DataFrame(fps, columns=fp_cols)
    
    def num_labels(self):
        """
        Return the number of isotopologue-level labelling pattern features 
        (i.e., specific numbers of carbon atoms from each tracer administered 
        to mice; for example, 13C2-Tau)

        Returns
        -------
        int
            Number of columns in the labelling-pattern matrix after filtering.
        """
        return self.label_df.shape[1]
    
    def __len__(self):
        """
        Return the number of training examples.

        Returns
        -------
        int
            Number of items in the training set.
        """
        return len(self.train_items)
    
    def __getitem__(self, idx):
        """
        Return one training example.

        Parameters
        ----------
        idx : int
            Position within the training set.

        Returns
        -------
        tuple of torch.Tensor
            `(pattern, true_fp)`, where `pattern` has shape
            `(num_labels,)` and `true_fp` has shape `(fp_length,)`.
        """
        return self.train_items[idx]
    
    def get_validation(self, batch_size):
        """
        Sample a validation batch.

        Parameters
        ----------
        batch_size : int
            Maximum number of validation examples to sample. 

        Returns
        -------
        tuple of torch.Tensor
            `patterns` with shape `(n_validation, num_labels)` and
            `true_fps` with shape `(n_validation, fp_length)`, where
            `n_validation` is at most `batch_size`.
        """
        indices = torch.randperm(len(self.val_items),
                                 generator=self._val_rng)[:batch_size]
        items = [self.val_items[i] for i in indices]
        return collate_fn(items)
