import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fingerprints import get_fingerprint
from functions import clean_mols

def collate_fn(batch):
    """
    Custom collate function for pattern-fingerprint pairs.

    Returns:
        patterns: (batch_size, num_labels) tensor
        true_fps: (batch_size, fp_length) tensor
    """
    patterns, true_fps = zip(*batch)
    return torch.stack(patterns), torch.stack(true_fps)

class IsotopicPatternDataset(Dataset):
    """
    A dataset containing isotopic labelling patterns for a set of peaks
    and ground-truth structures for each peak.

    Each item represents a peak, with:
    - an observed labelling pattern
    - the fingerprint of the ground-truth structure
    """
    
    def __init__(self, config, label_file, struct_file, device,
                 val_split=0.1, seed=0):
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
        return self.label_df.shape[1]
    
    def __len__(self):
        return len(self.train_items)
    
    def __getitem__(self, idx):
        return self.train_items[idx]
    
    def get_validation(self, batch_size):
        indices = torch.randperm(len(self.val_items),
                                 generator=self._val_rng)[:batch_size]
        items = [self.val_items[i] for i in indices]
        return collate_fn(items)
