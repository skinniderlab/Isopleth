import math
import torch
import torch.nn.functional as F
from torch import nn

class CLIPLoss(nn.Module):
    """
    CLIP-style symmetric contrastive loss: co-embed structures and labelling
    patterns from the same peak, while separating structure-labelling pattern
    pairs from different peaks.
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
        return self.log_temperature.exp()
    
    def forward(self, emb_fingerprints, emb_patterns):
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
