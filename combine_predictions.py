""" Gets predictions from all CV-folds and the final model and writes prediction outputs for the Isopleth model """

from pathlib import Path

import pandas as pd


def read_predictions(path):
    frame = pd.read_csv(path)
    required = {"peak_id", "candidate_smiles", "cosine", "rank"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["peak_id"] = frame["peak_id"].astype(str)
    return frame[["peak_id", "candidate_smiles", "cosine", "rank"]]


fold_predictions = [read_predictions(path) for path in snakemake.input.folds]
final_predictions = read_predictions(snakemake.input.final)
groups = fold_predictions + [final_predictions]

peak_groups = [set(frame["peak_id"]) for frame in groups]
for i in range(len(peak_groups)):
    for j in range(i + 1, len(peak_groups)):
        if peak_groups[i].intersection(peak_groups[j]):
            raise ValueError("prediction groups contain overlapping peaks")

predictions = pd.concat(groups, ignore_index=True)
if predictions[["peak_id", "candidate_smiles"]].duplicated().any():
    raise ValueError("combined predictions contain duplicate candidates")
if not predictions["cosine"].map(pd.notna).all():
    raise ValueError("combined predictions contain missing cosine scores")

meta_predictions = pd.concat(
    [read_predictions(path) for path in snakemake.input.meta_folds],
    ignore_index=True)
if meta_predictions[["peak_id", "candidate_smiles"]].duplicated().any():
    raise ValueError("meta-learning cosine predictions contain duplicates")
if meta_predictions["cosine"].isna().any():
    raise ValueError("meta-learning cosine predictions contain missing scores")
expected_meta = pd.read_csv(
    snakemake.input.metalearning_train,
    usecols=["peak_id", "candidate_smiles"]).dropna(
        subset=["candidate_smiles"])
expected_meta["peak_id"] = expected_meta["peak_id"].astype(str)
meta_keys = ["peak_id", "candidate_smiles"]
if expected_meta.duplicated(meta_keys).any():
    raise ValueError("meta-learning training table contains duplicate candidates")
if len(meta_predictions) != len(expected_meta):
    raise ValueError(
        f"scored {len(meta_predictions)} meta-learning candidates; "
        f"expected {len(expected_meta)}")
if not meta_predictions[meta_keys].merge(
        expected_meta, on=meta_keys, how="outer", indicator=True
        )["_merge"].eq("both").all():
    raise ValueError(
        "meta-learning cosine keys do not match the training feature table")
Path(snakemake.output.metalearning_train_cosines).parent.mkdir(
    parents=True, exist_ok=True)
meta_predictions.to_csv(
    snakemake.output.metalearning_train_cosines, index=False)

expected = pd.read_csv(
    snakemake.input.candidates,
    usecols=["peak_id", "candidate_smiles"])
expected["peak_id"] = expected["peak_id"].astype(str)
if expected.duplicated(["peak_id", "candidate_smiles"]).any():
    raise ValueError("input candidate table contains duplicate candidates")
predictions = expected.merge(
    predictions, on=["peak_id", "candidate_smiles"], how="left",
    validate="one_to_one")
if predictions[["cosine", "rank"]].isna().any().any():
    raise ValueError(
        "not every candidate in the final table received a prediction")

numeric_peak_ids = pd.to_numeric(predictions["peak_id"], errors="coerce")
if numeric_peak_ids.notna().all():
    predictions["peak_id"] = numeric_peak_ids.astype(int)
predictions = predictions.sort_values(["peak_id", "rank"])
Path(snakemake.output.predictions).parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(snakemake.output.predictions, index=False)

split = pd.read_csv(snakemake.input.split)
standards = split.loc[split["Source"] == "Standard"].copy()
standard_ids = set(standards["Peak #"].astype(str))

cv_predictions = predictions.copy()
cv_predictions["peak_id"] = cv_predictions["peak_id"].astype(str)
cv_predictions = cv_predictions.loc[
    cv_predictions["peak_id"].isin(standard_ids)]
if cv_predictions["peak_id"].nunique() != 231:
    raise ValueError("cross-validation predictions do not cover 231 standards")
cv_predictions.to_csv(snakemake.output.cv_predictions, index=False)

cv_truth = standards.rename(columns={
    "Peak #": "peak_id",
    "SMILES": "canonical_smiles",
})[["peak_id", "canonical_smiles", "InChIKey"]]
cv_truth.to_csv(snakemake.output.cv_truth, index=False)

print(
    f"wrote {len(predictions)} predictions for "
    f"{predictions['peak_id'].nunique()} peaks")
