"""
Neural network architecture for the Isopleth contrastive model

The model contains two encoders: one for molecular fingerprints and one for
isotopic labelling patterns. Both encoders project their inputs into the same
embedding space, where cosine similarity is used to score candidate structures
against observed labelling patterns
"""

import torch.nn as nn
import torch.nn.functional as F

class Isopleth(nn.Module):
    """
    Contrastive dual encoder for chemical fingerprints and isotopic labelling 
    patterns

    The fingerprint encoder maps fixed-length molecular fingerprints, such as
    ECFP bit vectors, into the shared embedding space. The pattern encoder maps
    binary isotopologue-level labelling patterns into the same space.

    Parameters
    ----------
    fp_in_dim : int
        Dimensionality of the input chemical fingerprint (e.g. length of a
        ECFP bit vector).
    fp_hidden_dim : int
        Size of the hidden layer in the fingerprint encoder.
    patt_in_dim : int
        Dimensionality of the input isotopic labelling pattern vector.
    patt_hidden_dim1 : int
        Size of the first hidden layer in the pattern encoder.
    patt_hidden_dim2 : int
        Size of the second hidden layer in the pattern encoder.
    emb_dim : int
        Dimension of the shared embedding space.
    fp_dropout : float, default = 0
        Dropout probability in the fingerprint encoder.
    patt_dropout : float, default = 0
        Dropout probability in the pattern encoder.
    """
    
    def __init__(self, fp_in_dim, fp_hidden_dim,
                 patt_in_dim, patt_hidden_dim1, patt_hidden_dim2,
                 emb_dim, fp_dropout=0, patt_dropout=0):
        super().__init__()
        
        self.fingerprint_encoder = nn.Sequential(
            nn.Linear(fp_in_dim, fp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fp_dropout),
            nn.Linear(fp_hidden_dim, emb_dim),
        )
        
        self.pattern_backbone = nn.Sequential(
            nn.Linear(patt_in_dim, patt_hidden_dim1),
            nn.ReLU(),
            nn.Dropout(patt_dropout),
            nn.Linear(patt_hidden_dim1, patt_hidden_dim2),
            nn.ReLU(),
            nn.Dropout(patt_dropout),
        )
        self.pattern_proj = nn.Sequential(
            nn.Linear(patt_hidden_dim2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
    
    def encode_fingerprints(self, x):
        """
        Encode chemical fingerprints.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(batch_size, fp_in_dim)` containing molecular
            fingerprints.

        Returns
        -------
        torch.Tensor
            L2-normalized fingerprint embeddings with shape
            `(batch_size, emb_dim)`.
        """
        return F.normalize(self.fingerprint_encoder(x), p=2, dim=-1, eps=1e-8)
    
    def encode_patterns(self, pattern):
        """
        Encode isotopic labelling patterns.

        Parameters
        ----------
        pattern : torch.Tensor
            Tensor of shape `(batch_size, patt_in_dim)` containing binary
            isotopic labelling patterns.

        Returns
        -------
        torch.Tensor
            L2-normalized pattern embeddings with shape
            `(batch_size, emb_dim)`.
        """
        h = self.pattern_backbone(pattern)
        h = self.pattern_proj(h)
        return F.normalize(h, p=2, dim=-1, eps=1e-8)
    
    def forward(self, fingerprints, patterns):
        """
        Encode both fingerprints and patterns, and return embeddings of each.

        Parameters
        ----------
        fingerprints : torch.Tensor
            Tensor of shape `(batch_size, fp_in_dim)`.
        patterns : torch.Tensor
            Tensor of shape `(batch_size, patt_in_dim)`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Fingerprint and pattern embeddings, each with shape
            `(batch_size, emb_dim)`.
        """
        return self.encode_fingerprints(fingerprints), \
            self.encode_patterns(patterns)
