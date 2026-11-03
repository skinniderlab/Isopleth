""" Gets predictions from all CV-folds and the final model and writes prediction outputs for the Isopleth + Meta-learner model """

from pathlib import Path

import pandas as pd


def read_predictions(path):
    frame = pd.read_csv(path)
    columns = [
        "peak_id", "candidate_smiles", "candidate_inchikey", "cosine",
        "ml_prob", "ml_rank",
    ]
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["peak_id"] = frame["peak_id"].astype(str)
    return frame[columns]


groups = [read_predictions(path) for path in snakemake.input.folds]
groups.append(read_predictions(snakemake.input.final))

peak_groups = [set(frame["peak_id"]) for frame in groups]
for i in range(len(peak_groups)):
    for j in range(i + 1, len(peak_groups)):
        if peak_groups[i].intersection(peak_groups[j]):
            raise ValueError("meta-learning prediction groups overlap")

predictions = pd.concat(groups, ignore_index=True)
keys = ["peak_id", "candidate_smiles"]
if predictions.duplicated(keys).any():
    raise ValueError("combined meta-learning predictions contain duplicates")
if predictions[["cosine", "ml_prob", "ml_rank"]].isna().any().any():
    raise ValueError("combined meta-learning predictions contain missing scores")

expected = pd.read_csv(
    snakemake.input.candidates,
    usecols=["peak_id", "candidate_smiles"])
expected["peak_id"] = expected["peak_id"].astype(str)
if expected.duplicated(keys).any():
    raise ValueError("input feature table contains duplicate candidates")
if len(predictions) != len(expected):
    raise ValueError(
        f"scored {len(predictions)} candidates; expected {len(expected)}")
if not predictions[keys].merge(
        expected, on=keys, how="outer", indicator=True)["_merge"].eq(
            "both").all():
    raise ValueError("meta-learning candidate keys do not match input features")

numeric_peak_ids = pd.to_numeric(predictions["peak_id"], errors="coerce")
if numeric_peak_ids.notna().all():
    predictions["peak_id"] = numeric_peak_ids.astype(int)
predictions = predictions.sort_values(["peak_id", "ml_rank"])
Path(snakemake.output.predictions).parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(snakemake.output.predictions, index=False)

split = pd.read_csv(snakemake.input.split)
standard_ids = set(split.loc[split["Source"] == "Standard", "Peak #"].astype(str))
cv_predictions = predictions.copy()
cv_predictions["peak_id"] = cv_predictions["peak_id"].astype(str)
cv_predictions = cv_predictions.loc[
    cv_predictions["peak_id"].isin(standard_ids),
    ["peak_id", "candidate_smiles", "ml_prob"]]
if cv_predictions["peak_id"].nunique() != 231:
    raise ValueError("meta-learning predictions do not cover 231 standards")
cv_predictions = cv_predictions.rename(columns={"ml_prob": "cosine"})
cv_predictions.to_csv(snakemake.output.cv_predictions, index=False)

print(
    f"wrote {len(predictions)} meta-learning predictions for "
    f"{predictions['peak_id'].nunique()} peaks")
