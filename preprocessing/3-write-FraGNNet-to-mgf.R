# Read predicted FraGNNet MS2 file and filter to those that match a
# DeepMet candidate structure with MS/MS, and write to MGF. 

setwd("Isopleth") 

library(tidyverse)
library(magrittr)
library(readxl)
library(Spectra)
library(MsBackendMgf)
library(furrr)
library(data.table)

# load experimental MS2
exp_ms2_file = "data/preprocessing_inputs/experimental-MS2.mgf.gz"
exp_ms2 = Spectra(
  exp_ms2_file,
  source = MsBackendMgf(),
  BPPARAM = BiocParallel::SerialParam()
)

# load DeepMet candidates
feat_file = "data/preprocessing_inputs/non_ms2_feature_table.csv"
feat = fread(feat_file, data.table = FALSE)

# what peaks both have experimental MS2 and at least one candidate structure?
exp_ids = gsub("^ID=|_.*$", "", exp_ms2$TITLE) %>% as.integer() %>% unique()
# get candidates for those structures
feat0 = filter(feat, peak_id %in% exp_ids)

# load the complete set of predicted MS/MS spectra
message('loading predicted MS2 spectra...')

# has to be downloaded from Zenodo: 10.5281/zenodo.23568529
msms_pred_file = "data/preprocessing_inputs/deepmet_dmids_fragnnet_merged.txt"
spectra_lines = readLines(msms_pred_file)

# split into a list of lines, one per spectrum
message('splitting predicted MS2 spectra...')
split_at = function(x, pos) unname(split(x, cumsum(seq_along(x) %in% pos)))
spectra_split = spectra_lines %>% 
  split_at(which(grepl("^#ESI", .)))

# parse to metadata
message('parsing metadata from predicted MS2 spectra...')
plan(multisession, workers = 16)
options(future.globals.maxSize = 3 * 1024^3)
spectra_meta = future_map_dfr(spectra_split, ~ {
  lines = .x
  meta_end_idx = which(lines == "") %>% dplyr::first()
  meta_lines = lines[1:(meta_end_idx - 1)] %>% 
    tail(-2)
  meta_fields = sub(": .*$", "", meta_lines) %>% gsub("^#", "", .)
  meta_values = sub("^[^: ]+", "", meta_lines) %>% gsub("^: ", "", .)
  meta_df = data.frame(t(meta_values)) %>% set_colnames(meta_fields)
  meta_df$adduct = strsplit(lines[1], ' ') %>% unlist() %>% extract(2)
  meta_df$collision_energy = lines[2] %>% gsub("^#", "", .)
  return(meta_df)
}, .progress = TRUE) %>% 
  mutate(spectrum_idx = row_number()) %>% 
  dplyr::select(spectrum_idx, everything()) %>% 
  type_convert() %>% 
  dplyr::rename(SMILES = Smiles,
                InChiKey = Inchikey) %>% 
  dplyr::select(-DataSource, -DataSourceID, -CompoundDataSource)

# filter to candidate structures for peaks of interest
message('filtering predicted MS2 spectra...')
spectra_meta0 = filter(spectra_meta, SMILES %in% feat0$candidate_smiles | 
                         InChiKey %in% feat0$candidate_inchikey)
spectra_split0 = spectra_split[spectra_meta0$spectrum_idx]

# create Spectra objects
message('creating Spectra objects...')
pred_spectra = future_map(seq_along(spectra_split0), ~ {
  spectrum_meta = spectra_meta0[.x, ]
  fragment_lines = spectra_split0[[.x]] %>% split_at(which(. == '')) %>% 
    extract2(2) %>% tail(-1)
  fragment_mz = map_dbl(fragment_lines, ~ gsub(" .*$", "", .) %>% as.numeric())
  fragment_intens = map_dbl(fragment_lines, ~ gsub("^.* ", "", .) %>% as.numeric())
  spd = data.frame(msLevel = 2L, rtime = 0.0)
  spd$mz = list(fragment_mz)
  spd$intensity = list(fragment_intens)
  spd$precursorMz = spectrum_meta$PrecursorMz
  spd$SMILES = spectrum_meta$SMILES
  spd$InChiKey = spectrum_meta$InChiKey
  spd$adduct = spectrum_meta$adduct
  spectrum = Spectra(spd)
}, .progress = TRUE)

# combine them all
message("merging ", length(pred_spectra), " predicted spectra ...")
ms2_pred = do.call(c, pred_spectra)

# export to MGF
message('exporting to MGF...')
pred_ms2_file = "data/preprocessing_inputs/predicted-MS2.mgf" 
export(ms2_pred, MsBackendMgf(), file = pred_ms2_file)

