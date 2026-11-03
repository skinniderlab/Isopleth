""" This code uses the pretrained Isopleth model and makes predictions for an input SMILES and labelling pattern """

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fingerprints import get_fingerprint
from functions import clean_mol
from models import Isopleth

# Get the structure for a reference input matrix 
reference_matrix = Path(__file__).parent / "data/workflow_inputs/train.mat.gz"


def load_model(model_file, device=None,):
    """ Load pretrained isopleth model """
    
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    model = Isopleth(
        fp_in_dim=2048,
        fp_hidden_dim=1024,
        patt_in_dim=840,
        patt_hidden_dim1=512,
        patt_hidden_dim2=512,
        emb_dim=1024,
    )

    state_dict = torch.load(model_file, map_location=device,weights_only=True,)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def clean_labelling_inputs(label):
    """ Function cleans the input of the isotope label, e.g. Serine.13C1, Serine_13C1, serine 13C1 are all valid inputs """
    label = str(label).replace(".", " ").replace("_", " ")
    return " ".join(label.split()).casefold()


def create_labelling_pattern(labelling):
    """ Function takes original labelling matrix as reference and sets everything to zero by default, then fills in with labelling input """
    reference_labels = pd.read_csv(reference_matrix, index_col=0, nrows=0,).columns.tolist()
    label_lookup = {clean_labelling_inputs(label): index for index, label in enumerate(reference_labels)}

    if isinstance(labelling, str):
        raise TypeError(
            "labelling must be a list of labels, for example "
            "['Serine 13C1', 'Taurine 13C2']"
        )

    pattern = np.zeros(len(reference_labels), dtype=np.float32)
    invalid_labels = []
    for label in labelling:
        index = label_lookup.get(clean_labelling_inputs(label))
        if index is None:
            invalid_labels.append(str(label))
        else:
            pattern[index] = 1

    if invalid_labels:
        raise ValueError("Unknown labelling value(s): " + ", ".join(invalid_labels))
    return pattern


def score_predictions(model, smiles, labelling):
    """
    Input smiles either as a single string or a list of multiple SMILES 
    Function takes labelling pattern and smiles and returns cosine similarity and rank
    """
    candidate_smiles = [smiles] if isinstance(smiles, str) else list(smiles)
    if not candidate_smiles:
        raise ValueError("smiles must contain at least one candidate structure")

    device = next(model.parameters()).device
    pattern = create_labelling_pattern(labelling) # Turn labelling input into binary vector
    fingerprints = np.stack([ # Turn SMILES input into fingerprint 
        get_fingerprint(
            mol=clean_mol(candidate),
            fp_algo="ecfp",
            fp_type="bits",
            length=2048,
            radius=3,
        )
        for candidate in candidate_smiles
    ])

    # Do predictions and calculate cosine, rank SMILES if needed
    with torch.no_grad():
        pattern_tensor = torch.from_numpy(pattern).unsqueeze(0).to(device)
        fingerprint_tensor = torch.from_numpy(fingerprints).to(device)
        pattern_embedding = model.encode_patterns(pattern_tensor)
        candidate_embeddings = model.encode_fingerprints(fingerprint_tensor)
        cosine = (candidate_embeddings @ pattern_embedding.T).squeeze(1).cpu().numpy()

    predictions = pd.DataFrame({"candidate_smiles": candidate_smiles, "cosine": cosine,})
    
    predictions["rank"] = (
        predictions["cosine"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return predictions.sort_values("rank").reset_index(drop=True)


