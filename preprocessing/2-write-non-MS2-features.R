# Consolidate features derived from DeepMet, RT prediction, or MS1 (ppm error)
setwd("Isopleth")
library(dplyr)
library(readr)
library(magrittr)

# Code writes table with non MS2 features 
non_ms2_features = "data/preprocessing_inputs/non_ms2_feature_table.csv"

# load DeepMet candidates and features for each:
# - confidence
# - size
# - rank
matches = read.csv("data/preprocessing_inputs/annotation_file.csv")
features = matches %>% 
  dplyr::select(peak_id, SMILES, InChIKey, mz, Mass,
                structure_size, structure_rank, structure_confidence) %>% 
  dplyr::rename(candidate_smiles = SMILES, candidate_inchikey = InChIKey)
stopifnot(!any(is.na(features$structure_size)))
stopifnot(!any(is.na(features$structure_rank)))
stopifnot(!any(is.na(features$structure_confidence)))

# compute ppm error
features %<>%
  mutate(theor_mz = Mass - 1.007276, 
         ppm_error = abs(theor_mz - mz) / mz * 1e6) %>% 
  dplyr::select(-Mass, -mz, -theor_mz)

# load RT predictions
rts = read_csv("data/preprocessing_inputs/retention_time_predictions.csv.gz")
# load peak metadata to calculate exp vs pred delta
pks = read.csv("data/preprocessing_inputs/mouse_tissue_metabolomics_dataset.csv")
rts %<>% 
  left_join(pks, by = 'peak_id') %>% 
  mutate(delta_RT = abs(pred_RT - RT))
delta_rt = distinct(rts, peak_id, candidate_smiles, candidate_inchikey, delta_RT) 
features %<>% left_join(delta_rt, by = c('peak_id', 'candidate_smiles', 
                                         'candidate_inchikey'))
stopifnot(!any(is.na(features$delta_RT)))

# write to file
write.csv(features, non_ms2_features, row.names = FALSE)
