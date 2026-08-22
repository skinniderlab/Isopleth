"""Loss functions for training the Isopleth contrastive model."""

import math
import torch
import torch.nn.functional as F
from torch import nn

class CLIPLoss(nn.Module):
    """
    CLIP-style symmetric contrastive loss: co-embed structures and labelling
    patterns from the same peak, while separating structure-labelling pattern
    pairs from different peaks.

    For a batch of matched fingerprint and labelling-pattern embeddings,
    diagonal entries of the similarity matrix are treated as positive pairs and
    off-diagonal entries are treated as negatives. The loss is computed in both
    directions (symmetric), from fingerprints to patterns and from patterns to fingerprints.

    Parameters
    ----------
    temperature : float, default=0.1
        Temperature parameter
    label_smoothing : float, default=0.0
        Label smoothing
    learnable_temperature : bool, default=False
        If True, add temperature as a trainable parameter.  (Not used here:
        loss parameters are not passed to the optimizer.)
    """
    
    def __init__(self, temperature=0.1, label_smoothing=0.0,
                 learnable_temperature=False):
        super().__init__()
        self.label_smoothing = label_smoothing
        if learnable_temperature:
            # log scale for numerical stability
            self.log_temperature = nn.Parameter(torch.tensor(math.log(
                temperature)))
        else:
            self.register_buffer('log_temperature', torch.tensor(math.log(
                temperature)))
    
    @property
    def temperature(self):
        """Return the temperature value used to scale logits."""
        return self.log_temperature.exp()
    
    def forward(self, emb_fingerprints, emb_patterns):
        """
        Compute the symmetric contrastive loss for a minibatch.

        Parameters
        ----------
        emb_fingerprints : torch.Tensor
            L2-normalized fingerprint embeddings with shape
            `(batch_size, emb_dim)`.
        emb_patterns : torch.Tensor
            L2-normalized labelling-pattern embeddings with shape
            `(batch_size, emb_dim)`.

        Returns
        -------
        torch.Tensor
            Scalar contrastive loss.
        """
        batch_size = emb_fingerprints.shape[0]
        assert batch_size >= 2
        assert emb_fingerprints.shape[0] == emb_patterns.shape[0]
        
        # compute similarity matrix of dimensions (batch_size, batch_size)
        # L2 norm in encoders produces cosine similarities
        logits = emb_fingerprints @ emb_patterns.T / self.temperature
        
        # get labels (diagonal entries are positive pairs)
        labels = torch.arange(batch_size, device=emb_fingerprints.device)
        
        # symmetric loss
        loss_a_to_b = F.cross_entropy(logits, labels,
                                      label_smoothing=self.label_smoothing)
        loss_b_to_a = F.cross_entropy(logits.T, labels,
                                      label_smoothing=self.label_smoothing)
        return (loss_a_to_b + loss_b_to_a) / 2
