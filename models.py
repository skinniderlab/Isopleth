import torch.nn as nn
import torch.nn.functional as F

class Isopleth(nn.Module):
    """
    Co-embed chemical fingerprints and isotopic labelling patterns
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
        return F.normalize(self.fingerprint_encoder(x), p=2, dim=-1, eps=1e-8)
    
    def encode_patterns(self, pattern):
        h = self.pattern_backbone(pattern)
        h = self.pattern_proj(h)
        return F.normalize(h, p=2, dim=-1, eps=1e-8)
    
    def forward(self, fingerprints, patterns):
        """Encode both fingerprints and patterns, and return embeddings
        of each."""
        return self.encode_fingerprints(fingerprints), \
            self.encode_patterns(patterns)
