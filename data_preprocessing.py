## Code prepares generating the training sets for the contrastive learning model ##

from pathlib import Path
import pandas as pd

# Defining input and output files for Snakemake workflow
if "snakemake" in globals():
    pks_file = snakemake.input.peaks
    standards_file = snakemake.input.standards
    hmdb_file = snakemake.input.hmdb
    labelling_file = snakemake.input.labelling_matrix
    fold_file = snakemake.input.fold_assignments
    split_file = snakemake.output.split
    cv_file = snakemake.output.cv_assignments
    train_file = snakemake.output.train_structures
    train_mat_file = snakemake.output.train_patterns
    test_mat_file = snakemake.output.test_patterns
else:
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "data" / "preprocessing_inputs"
    output_dir = base_dir / "data" / "workflow_inputs"
    pks_file = input_dir / "mouse_tissue_metabolomics_dataset.csv"
    standards_file = input_dir / "matched_standards_to_mouse.csv"
    hmdb_file = input_dir / "HMDB5-metabolites-from-xml-canonical.csv.gz"
    labelling_file = input_dir / "labelling_matrix.csv.gz"
    fold_file = input_dir / "fold_assignments.csv.gz"
    split_file = output_dir / "split.csv"
    cv_file = output_dir / "cv_test_assignments.csv.gz"
    train_file = output_dir / "train.csv.gz"
    train_mat_file = output_dir / "train.mat.gz"
    test_mat_file = output_dir / "test.mat.gz"

# Parameters
rt_tol = 0.5            # retention time tolerance in minutes
ms1_ppm_err = 5         # MS1 mass error in parts per million (ppm)


# Helper functions: 
def calc_ppm_range(theor_mass, err_ppm=5):
    """ Calculates and returns the mass range within a given error rate (ppm) """
    return (
        theor_mass * (1 - err_ppm / 1e6),
        theor_mass * (1 + err_ppm / 1e6),
    )

################################################################################
## Step 1: Load all 231 one-to-one-matches used as the evaluation set
################################################################################
pks_table = pd.read_csv(pks_file)
match_df = pd.read_csv(standards_file)
match_df = match_df.loc[:, ~match_df.columns.str.startswith("Unnamed:")]

if len(match_df) != 231 or match_df["peak_id"].nunique() != 231:
    raise ValueError(
        "expected 231 one-to-one standard matches, "
        f"found {len(match_df)} rows and {match_df['peak_id'].nunique()} peaks"
    )

print(f"Loaded {len(match_df)} one-to-one standard matches")

################################################################################
## Step 2: Find all artifacts that match the 231 standards
################################################################################
pks_filtered = pks_table[pks_table["peak_id"].isin(match_df["peak_id"])]

# Keep peaks annotated as Metabolite
pks_standards_accessions = pks_filtered[pks_filtered["NetID_class"] == "Metabolite"].copy()

# Extract the first annotation token containing HMDB
pks_standards_accessions["hmdb_id"] = pks_standards_accessions["NetID_annotation"].str.extract(r"(\S*HMDB\S*)", expand=False)

# Remove peaks without an HMDB ID
pks_standards_accessions = pks_standards_accessions.dropna(subset=["hmdb_id"])

print(f"  {len(pks_standards_accessions)} peaks with HMDB accessions")

# Using NetIDs annotation, we can now filter for all artifacts dervied from those original peaks 
artifacts = {}

for row in pks_standards_accessions.itertuples(index=False):
    matches = pks_table[
        pks_table["NetID_annotation"].str.contains(
            row.hmdb_id, regex=False, na=False
        )
        & (pks_table["NetID_class"] != "Putative metabolite")
        & ~pks_table["peak_id"].isin(match_df["peak_id"])
    ]

    if len(matches) > 0:
        artifacts[row.peak_id] = matches["peak_id"].tolist()

n_standards = len(artifacts)
n_matches = sum(len(ids) for ids in artifacts.values())
n_artifacts = len({
    peak_id
    for artifact_ids in artifacts.values()
    for peak_id in artifact_ids
})

print(
    f"Found {n_matches} matches between "
    f"{n_standards} standards and {n_artifacts} unique artifact peaks"
)

# Add the standard information to each matched artifact
artifact_rows = []

for matched_peak, artifact_ids in artifacts.items():
    std_row = match_df[match_df["peak_id"] == matched_peak].iloc[0]
    peak_meta = pks_table[pks_table["peak_id"].isin(artifact_ids)][["peak_id", "mz", "RT", "NetID_annotation"]].copy()
    peak_meta = peak_meta.rename(columns={"mz": "medMz", "RT": "medRt"})
    peak_meta["matched_peak"] = matched_peak

    for column in ["compound_id", "name", "SMILES", "canonical_smiles",
                   "InChIKey", "Formula", "mz", "rt"]:
        peak_meta[column] = std_row[column]

    artifact_rows.append(peak_meta)

artifact_df = pd.concat(artifact_rows, ignore_index=True)

# Add standards without artifacts
no_artifact_df = match_df[~match_df["peak_id"].isin(artifact_df["peak_id"])]
no_artifact_df["matched_peak"] = no_artifact_df["peak_id"]
artifact_df = pd.concat([artifact_df, no_artifact_df], ignore_index=True)

# Keep artifacts within the retention-time range
artifact_df0 = artifact_df[(artifact_df["rt"] - artifact_df["medRt"]).abs() <= rt_tol]

print(f"After adding artifacts: {len(artifact_df0)} annotations for " f"{artifact_df0['peak_id'].nunique()} unique peaks")

# Remove duplicate annotations
group_columns = [column for column in artifact_df0.columns if column not in ["NetID_annotation", "matched_peak"]]

artifact_df1 = artifact_df0.groupby(group_columns, dropna=False, as_index=False).agg({
    "NetID_annotation": lambda x: "|".join(x.dropna().astype(str).unique()),
    "matched_peak": lambda x: "|".join(x.astype(str).unique()),
})

print(f"After removing duplicates: {len(artifact_df1)} annotations for " f"{artifact_df1['peak_id'].nunique()} unique peaks")

# Keep the original peak when it is both a standard and an artifact
n_matches = artifact_df1.groupby("peak_id")["matched_peak"].transform("nunique")
dup_peaks = artifact_df1.loc[n_matches > 1, "peak_id"].unique()

artifact_df2a = artifact_df1[~artifact_df1["peak_id"].isin(dup_peaks)]
artifact_df2b = artifact_df1[artifact_df1["peak_id"].isin(dup_peaks) & (artifact_df1["peak_id"].astype(str) == artifact_df1["matched_peak"])]
artifact_df2 = pd.concat([artifact_df2a, artifact_df2b]).sort_values(["matched_peak", "peak_id"])

print(f"After removing conflicts: {len(artifact_df2)} annotations for "
      f"{artifact_df2['peak_id'].nunique()} unique peaks")

# Label standards and artifacts, then update the matched data frame
artifact_df2["ann_source"] = "Artifact"
artifact_df2.loc[artifact_df2["peak_id"].isin(match_df["peak_id"]), "ann_source"] = "Standard"
artifact_names = artifact_df2["ann_source"] == "Artifact"
artifact_df2.loc[artifact_names, "name"] = artifact_df2.loc[artifact_names, "name"].str.replace("\t", "", regex=False) # handle white spaces
match_df = artifact_df2.reset_index(drop=True)

print(f"Final match data: {len(match_df)} peaks " f"({sum(match_df['ann_source'] == 'Standard')} standards and " f"{sum(match_df['ann_source'] == 'Artifact')} artifacts)")

################################################################################
## Step 3: Add NetID annotations 
################################################################################

# Keep non-artifact peaks with an HMDB accession
pks_NetID = pks_table[pks_table["NetID_class"] != "Putative metabolite"]
pks_NetID["hmdb_id"] = pks_NetID["NetID_annotation"].str.extract(r"(\S*HMDB\S*)", expand=False)
pks_NetID = pks_NetID.dropna(subset=["hmdb_id"])

# Map the HMDB accessions to structures
hmdb = pd.read_csv(hmdb_file, low_memory=False)
hmdb0 = hmdb.dropna(subset=["canonical_smiles"])
hmdb_map = hmdb0[["hmdb_id", "name", "canonical_smiles", "inchikey", "formula"]]
pks_NetID = pks_NetID.merge(hmdb_map, on="hmdb_id", how="left")

# Match the format of the standards and artifacts
pks_NetID = pks_NetID[["peak_id", "mz", "RT", "name", "canonical_smiles", "inchikey", "formula", "hmdb_id"]]
pks_NetID = pks_NetID.rename(columns={"mz": "medMz", "RT": "medRt", "inchikey": "InChIKey", "formula": "Formula"})

print(f"Found {len(pks_NetID)} peaks annotated with " f"{pks_NetID['InChIKey'].nunique()} unique InChIKeys by NetID")

# Prefer existing standard and artifact annotations over NetID
pks1 = pks_NetID[~pks_NetID["peak_id"].isin(match_df["peak_id"])]
pks1["ann_source"] = "NetID"

print(f"Of these, {len(pks1)} peaks annotated with " f"{pks1['InChIKey'].nunique()} unique InChIKeys were not already annotated")

# This is the total and final number of all annotated peaks (4,873 annotated peaks)
match_df = pd.concat([match_df, pks1], ignore_index=True)

print(f"After merging, {len(match_df)} peaks annotated with " f"{match_df['InChIKey'].nunique()} unique InChIKeys")

################################################################################
## Step 4: Generate training and testing folds  
################################################################################

# These fold assignments were originally generated in R using set.seed(0).
# Python uses a different random-number generator, so the original assignments
# are provided here to reproduce the same training and testing folds.
fold_assignments = pd.read_csv(fold_file)
match_df = match_df.merge(fold_assignments, on="peak_id")

# Match the annotation format used by the Snakemake workflow
train_df = match_df[["peak_id", "medMz", "medRt", "Formula",
                     "canonical_smiles", "InChIKey", "ann_source",
                     "compound_id", "name", "matched_peak",
                     "NetID_annotation", "hmdb_id"]]
train_df = train_df.rename(columns={"NetID_annotation": "annotation"})

# Split the labelling matrix into annotated and unannotated peaks
label_mat = pd.read_csv(labelling_file, index_col="peak_id")
train_mat = label_mat.loc[train_df["peak_id"]]
test_mat = label_mat.drop(train_df["peak_id"])
train_mat.index.name = None
test_mat.index.name = None

# The split table contains annotated peaks with identifiers
split_df = match_df.dropna(subset=["canonical_smiles", "InChIKey"])
split_df = split_df[["test_fold", "peak_id", "canonical_smiles", "InChIKey",
                     "name", "ann_source", "NetID_annotation", "hmdb_id"]]
split_df = split_df.rename(columns={
    "test_fold": "Test fold",
    "peak_id": "Peak #",
    "canonical_smiles": "SMILES",
    "name": "Name",
    "ann_source": "Source",
    "NetID_annotation": "NetID artifact annotation",
    "hmdb_id": "HMDB ID",
})
split_df = split_df.sort_values(["Test fold", "Peak #"])

# Fold assignments for artifact and NetID peaks
cv_assignments = match_df[match_df["ann_source"] != "Standard"]
cv_assignments = cv_assignments[["peak_id", "test_fold"]]

# Write output to be used in the Snakemake workflow
for output_file in [split_file, cv_file, train_file, train_mat_file, test_mat_file]:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
train_df.to_csv(train_file, index=False)
train_mat.to_csv(train_mat_file)
test_mat.to_csv(test_mat_file)
split_df.to_csv(split_file, index=False)
cv_assignments.to_csv(cv_file, index=False)

print(f"Wrote {len(train_df)} annotated and {len(test_mat)} unannotated peaks")
