# Consolidate features derived from DeepMet, RT prediction, or MS1 (ppm error)
library(tidyverse)
library(magrittr)
library(data.table)
library(readxl)
library(data.table)

# dynamically parse arguments
parser = ArgumentParser(prog = 'write-non-MS2-features.R')
grid = read.delim("sh/grids/write-non-MS2-features.txt")
for (param_name in colnames(grid))
  parser$add_argument(paste0('--', param_name),
                      type = typeof(grid[[param_name]]))
args = parser$parse_args()
print(args)

# load DeepMet candidates and features for each:
# - confidence
# - size
# - rank
matches = fread(args$ann_file, data.table = FALSE)
features = matches %>% 
  dplyr::select(peak_id, SMILES, InChIKey, medMz, Mass,
                structure_size, structure_rank, structure_confidence) %>% 
  dplyr::rename(candidate_smiles = SMILES, candidate_inchikey = InChIKey)
stopifnot(!any(is.na(features$structure_size)))
stopifnot(!any(is.na(features$structure_rank)))
stopifnot(!any(is.na(features$structure_confidence)))

# compute ppm error
features %<>%
  mutate(theor_mz = Mass - 1.007276, 
         ppm_error = abs(theor_mz - medMz) / medMz * 1e6) %>% 
  dplyr::select(-Mass, -medMz, -theor_mz)

# load RT predictions
rts = fread(args$rt_pred_file, data.table = FALSE) 
# load peak metadata to calculate exp vs pred delta
pks = read_excel(args$pks_file)
rts %<>% 
  left_join(pks, by = 'peak_id') %>% 
  mutate(delta_RT = abs(avg_RT - medRt))
delta_rt = distinct(rts, peak_id, SMILES, InChIKey, delta_RT) %>% 
  dplyr::rename(candidate_smiles = SMILES, candidate_inchikey = InChIKey)
features %<>% left_join(delta_rt, by = c('peak_id', 'candidate_smiles', 
                                         'candidate_inchikey'))
stopifnot(!any(is.na(features$delta_RT)))

# write to file
write.csv(features, args$output_file, row.names = FALSE)
