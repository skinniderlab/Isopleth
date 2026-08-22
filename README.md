# Isopleth

Isopleth is a contrastive learning model that maps labelling patterns and chemical structures into a shared embedding space, placing labelling patterns for each metabolite near their corresponding structures while pushing unpaired patterns and structures apart. After model training, candidate structures for unidentified peaks are ranked by their cosine similarity to the corresponding labelling patterns for those peaks within this learned embedding space.

This repository also includes a meta-learning model that combines the Isopleth cosine similarity score with features derived from DeepMet, predicted retention times, and predicted MS/MS spectra for each candidate.

Predictions for all 6,651 peaks (the subset of all 8,277 peaks with at least one DeepMet-generated candidate structure; predictions made in cross-validation) are provided in `data/isopleth_predictions.csv.gz`. Predictions made by the meta-learning model for the same set of 6,651 peaks in cross-validation are provided in `data/metalearning_predictions.csv.gz`.

The experiments described in the manuscript were carried out in the conda environment specified in `environment.yaml`.

## Scripts

`train_isopleth.py` trains a contrastive learning model and makes predictions on a set of withheld peaks.

`evaluate.py` quantifies the performance of the contrastive learning model in the held-out test set using various metrics (e.g. top-k accuracy, top-1 Tanimoto coefficient to the ground-truth structure).

`train_metalearning.py` trains a random forest model using additional features beyond the isotope tracing data (derived from DeepMet, predicted retention times, predicted MS/MS spectra, and Isopleth) and scores candidate structures for a set of withheld peaks.