""" Code prepares the ten-folds for cross-validation and the all-data final training set"""

import gzip
from pathlib import Path
import pandas as pd


output_dir = Path(snakemake.output.prepared)
folds = [str(fold) for fold in snakemake.config.get("folds", range(1, 11))]

split = pd.read_csv(snakemake.input.split)
required_split = {"Test fold", "Peak #", "SMILES", "InChIKey", "Source"}
missing = required_split.difference(split.columns)
if missing:
    raise ValueError(f"split table is missing columns: {sorted(missing)}")

split = split.rename(columns={
    "Test fold": "test_fold",
    "Peak #": "peak_id",
    "SMILES": "canonical_smiles",
})
split["peak_id"] = split["peak_id"].astype(str)
split["test_fold"] = split["test_fold"].astype(int)
if split["peak_id"].duplicated().any():
    raise ValueError("split table contains duplicate peak IDs")
if split[["canonical_smiles", "InChIKey"]].isna().any().any():
    raise ValueError("split table contains missing training structures")
if set(split["test_fold"].astype(str)) != set(folds):
    raise ValueError("configured folds do not match the published split")

train_structures = pd.read_csv(snakemake.input.train_structures)
required_structures = {"peak_id", "canonical_smiles", "InChIKey"}
missing = required_structures.difference(train_structures.columns)
if missing:
    raise ValueError(
        f"annotated structure table is missing columns: {sorted(missing)}")
train_structures["peak_id"] = train_structures["peak_id"].astype(str)
if train_structures["peak_id"].duplicated().any():
    raise ValueError("annotated structure table contains duplicate peak IDs")
train_structures = train_structures.dropna(
    subset=["canonical_smiles", "InChIKey"]).copy()
if set(train_structures["peak_id"]) != set(split["peak_id"]):
    raise ValueError("valid annotated structures do not match the split table")

meta_peak_order = (
    pd.read_csv(snakemake.input.metalearning_train, usecols=["peak_id"])
    ["peak_id"].astype(str).drop_duplicates().tolist())
if set(meta_peak_order) != set(train_structures["peak_id"]):
    raise ValueError(
        "meta-learning training peaks do not match annotated structures")
cv_train_structures = (
    train_structures.set_index("peak_id").loc[meta_peak_order]
    .reset_index())

train_patterns = pd.read_csv(snakemake.input.train_patterns, index_col=0)
train_patterns.index = train_patterns.index.astype(str)
test_patterns = pd.read_csv(snakemake.input.test_patterns, index_col=0)
test_patterns.index = test_patterns.index.astype(str)

if not train_patterns.columns.equals(test_patterns.columns):
    raise ValueError("annotated and unseen pattern matrices have different columns")
if not train_patterns.index.is_unique or not test_patterns.index.is_unique:
    raise ValueError("pattern matrix peak IDs must be unique")

annotated_ids = set(split["peak_id"])
missing_patterns = annotated_ids.difference(train_patterns.index)
if missing_patterns:
    raise ValueError(
        f"{len(missing_patterns)} annotated peaks lack labelling patterns")
if set(train_patterns.index).intersection(test_patterns.index):
    raise ValueError("annotated and unseen pattern matrices overlap")

cv_assignments = pd.read_csv(snakemake.input.cv_assignments)
required_assignments = {"peak_id", "test_fold"}
missing = required_assignments.difference(cv_assignments.columns)
if missing:
    raise ValueError(
        f"augmentation fold table is missing columns: {sorted(missing)}")
cv_assignments["peak_id"] = cv_assignments["peak_id"].astype(str)
cv_assignments["test_fold"] = cv_assignments["test_fold"].astype(int)
if cv_assignments["peak_id"].duplicated().any():
    raise ValueError("augmentation fold table contains duplicate peak IDs")

split_fold_for_peak = dict(zip(split["peak_id"], split["test_fold"]))
for row in cv_assignments.itertuples(index=False):
    if (row.peak_id in split_fold_for_peak and
            split_fold_for_peak[row.peak_id] != row.test_fold):
        raise ValueError(f"conflicting fold assignments for peak {row.peak_id}")


def metadata(rows):
    return rows[["peak_id", "canonical_smiles", "InChIKey"]].copy()


def write_matrix(frame, path):
    frame = frame.copy()
    frame.index.name = "peak_id"
    frame.to_csv(path, compression="gzip")


output_dir.mkdir(parents=True, exist_ok=True)
fold_for_peak = dict(split_fold_for_peak)
fold_for_peak.update(dict(zip(
    cv_assignments["peak_id"], cv_assignments["test_fold"])))
if set(fold_for_peak).difference(train_patterns.index):
    raise ValueError("fold assignments contain peaks without annotated patterns")

for fold in folds:
    fold_dir = output_dir / "cv" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    test_rows = split.loc[split["test_fold"] == int(fold)].copy()
    train_rows = cv_train_structures.loc[
        cv_train_structures["peak_id"].map(split_fold_for_peak) !=
        int(fold)].copy()
    train_ids = train_rows["peak_id"].tolist()
    test_ids = [
        peak_id for peak_id, assigned_fold in fold_for_peak.items()
        if assigned_fold == int(fold)
    ]

    metadata(train_rows).to_csv(
        fold_dir / "train.csv.gz", index=False, compression="gzip")
    write_matrix(
        train_patterns.loc[train_ids], fold_dir / "train.mat.csv.gz")
    write_matrix(
        train_patterns.loc[test_ids], fold_dir / "test.mat.csv.gz")

final_dir = output_dir / "final"
final_dir.mkdir(parents=True, exist_ok=True)
all_annotated_ids = train_structures["peak_id"].tolist()
metadata(train_structures).to_csv(
    final_dir / "train.csv.gz", index=False, compression="gzip")
write_matrix(
    train_patterns.loc[all_annotated_ids],
    final_dir / "train.mat.csv.gz")
write_matrix(test_patterns, final_dir / "test.mat.csv.gz")

standard_truth = split.loc[split["Source"] == "Standard"].copy()
if len(standard_truth) != 231:
    raise ValueError(f"expected 231 standards, found {len(standard_truth)}")
metadata(standard_truth).to_csv(
    output_dir / "cv_standards.csv.gz", index=False, compression="gzip")

candidate_paths = {
    fold: output_dir / "cv" / f"fold_{fold}" / "test_candidates.csv.gz"
    for fold in folds
}
candidate_paths["final"] = final_dir / "test_candidates.csv.gz"
meta_candidate_paths = {
    fold: output_dir / "cv" / f"fold_{fold}" / "meta_candidates.csv.gz"
    for fold in folds
}
candidate_handles = {
    key: gzip.open(path, "wt", newline="")
    for key, path in candidate_paths.items()
}
candidate_headers = {key: True for key in candidate_handles}
meta_candidate_handles = {
    key: gzip.open(path, "wt", newline="")
    for key, path in meta_candidate_paths.items()
}
meta_candidate_headers = {key: True for key in meta_candidate_handles}

feature_paths = {}
for fold in folds:
    feature_paths[(fold, "train")] = (
        output_dir / "cv" / f"fold_{fold}" / "train_features.csv.gz")
    feature_paths[(fold, "test")] = (
        output_dir / "cv" / f"fold_{fold}" / "test_features.csv.gz")
feature_paths[("final", "train")] = final_dir / "train_features.csv.gz"
feature_paths[("final", "test")] = final_dir / "test_features.csv.gz"
feature_handles = {
    key: gzip.open(path, "wt", newline="")
    for key, path in feature_paths.items()
}
feature_headers = {key: True for key in feature_handles}

feature_columns = [
    "peak_id", "candidate_smiles", "candidate_inchikey",
    "structure_size", "structure_rank", "structure_confidence",
    "ppm_error", "delta_RT", "dot_product", "n_overlap",
]
candidate_columns = ["peak_id", "candidate_smiles"]
candidate_header = pd.read_csv(snakemake.input.candidates, nrows=0).columns
missing = set(feature_columns).difference(candidate_header)
if missing:
    raise ValueError(
        f"candidate feature table is missing columns: {sorted(missing)}")

candidate_peak_ids = set()
try:
    for chunk in pd.read_csv(
            snakemake.input.candidates,
            usecols=feature_columns,
            chunksize=100000):
        chunk["peak_id"] = chunk["peak_id"].astype(str)
        candidate_peak_ids.update(chunk["peak_id"].unique())
        assignments = chunk["peak_id"].map(fold_for_peak)

        for fold in folds:
            current = chunk.loc[assignments == int(fold)]
            current[candidate_columns].to_csv(
                candidate_handles[fold], index=False,
                header=candidate_headers[fold])
            candidate_headers[fold] = False

            current.to_csv(
                feature_handles[(fold, "test")], index=False,
                header=feature_headers[(fold, "test")])
            feature_headers[(fold, "test")] = False

        current = chunk.loc[chunk["peak_id"].isin(test_patterns.index)]
        current[candidate_columns].to_csv(
            candidate_handles["final"], index=False,
            header=candidate_headers["final"])
        candidate_headers["final"] = False
        current.to_csv(
            feature_handles[("final", "test")], index=False,
            header=feature_headers[("final", "test")])
        feature_headers[("final", "test")] = False

    meta_header = pd.read_csv(
        snakemake.input.metalearning_train, nrows=0).columns
    required_meta = set(feature_columns + ["match"])
    missing = required_meta.difference(meta_header)
    if missing:
        raise ValueError(
            "meta-learning training table is missing columns: "
            f"{sorted(missing)}")

    for chunk in pd.read_csv(
            snakemake.input.metalearning_train,
            usecols=feature_columns + ["match"], chunksize=100000,
            low_memory=False):
        chunk["peak_id"] = chunk["peak_id"].astype(str)
        assignments = chunk["peak_id"].map(fold_for_peak)
        if assignments.isna().any():
            examples = chunk.loc[
                assignments.isna(), "peak_id"].drop_duplicates().head(5)
            raise ValueError(
                "meta-learning training peaks lack fold assignments, e.g. "
                f"{examples.tolist()}")

        for fold in folds:
            current = chunk.loc[assignments != int(fold)]
            current.to_csv(
                feature_handles[(fold, "train")], index=False,
                header=feature_headers[(fold, "train")])
            feature_headers[(fold, "train")] = False

            current = chunk.loc[assignments == int(fold), candidate_columns]
            current.to_csv(
                meta_candidate_handles[fold], index=False,
                header=meta_candidate_headers[fold])
            meta_candidate_headers[fold] = False

        chunk.to_csv(
            feature_handles[("final", "train")], index=False,
            header=feature_headers[("final", "train")])
        feature_headers[("final", "train")] = False
finally:
    for handle in candidate_handles.values():
        handle.close()
    for handle in meta_candidate_handles.values():
        handle.close()
    for handle in feature_handles.values():
        handle.close()

known_pattern_ids = set(train_patterns.index).union(test_patterns.index)
unknown_candidates = candidate_peak_ids.difference(known_pattern_ids)
if unknown_candidates:
    examples = sorted(unknown_candidates)[:5]
    raise ValueError(
        f"candidate table contains peaks without patterns, e.g. {examples}")

print(
    f"prepared {len(folds)} folds from {len(split)} annotated peaks; "
    f"{len(test_patterns)} peaks are reserved for the all-data model")
