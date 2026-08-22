# Compute the similarity between MS/MS spectra predicted for candidate 
# structures and experimental MS/MS spectra from mouse tissues.
library(argparse)
library(tidyverse)
library(magrittr)
library(readxl)
library(Spectra)
library(MsBackendMgf)
library(furrr)
library(data.table)

# dynamically parse arguments
parser = ArgumentParser(prog = 'write-MS2-similarity-predicted-vs-experimental.R')
grid = read.delim("sh/grids/write-MS2-similarity-predicted-vs-experimental.txt")
for (param_name in colnames(grid))
  parser$add_argument(paste0('--', param_name),
                      type = typeof(grid[[param_name]]))
args = parser$parse_args()
print(args)

# function to calculate ppm range
calc_ppm_range = function(theor_mass, err_ppm = 5) {
  c(
    (-err_ppm / 1e6 * theor_mass) + theor_mass,
    (err_ppm / 1e6 * theor_mass) + theor_mass
  )
}

# load metadata for all 8277 peaks
pks = read_excel(args$pks_file)

# load experimental MS2
message('reading MS2 data...')
exp_ms2 = Spectra(args$exp_ms2_file, source = MsBackendMgf())
# load predicted MS2 spectra
pred_ms2 = Spectra(args$pred_ms2_file, source = MsBackendMgf(),
                   BPPARAM = SerialParam())

# load DeepMet candidates, with features derived from DeepMet/RT pred
feat = fread(args$feat_file, data.table = FALSE)

# extract metadata
message('extracting metadata...')
pred_meta = spectraData(pred_ms2) %>% as.data.frame()

# remove precursor peaks from both
exp_ms2_no_prec = exp_ms2 %>% 
  filterPrecursorPeaks(ppm = 50) %>% applyProcessing()
pred_ms2_no_prec = pred_ms2 %>% 
  filterPrecursorPeaks(ppm = 50) %>% applyProcessing()

# compute overlap and dot-product for each pair
message('scoring MS2 similarities...')
mz_tol = 5.1
plan(multisession, workers = 16)
options(future.globals.maxSize = 6000*1024^2)
ms2_score_df = future_pmap_dfr(feat, function(...) {
  current = tibble(...)
  
  # find predicted MS/MS spectrum, if it exists
  mz = filter(pks, peak_id == current$peak_id) %>% 
    pull(medMz)
  mz_range = calc_ppm_range(mz, mz_tol)
  spectrum_idx = which((pred_meta$InChiKey == current$InChIKey |
                          pred_meta$SMILES == current$SMILES) &
                         pred_meta$adduct == '[M-H]-')
  # if there is no spectrum for this predicted structure (e.g. b/c 
  # net charge > 0), return NA
  if (length(spectrum_idx) == 0) {
    message("no predicted MS/MS for peak ", current$peak_id, ', InChIKey: ', 
            current$InChIKey)  
    return(current)
  }
  pred_spectrum = pred_ms2[spectrum_idx]
  pred_spectrum_no_prec = pred_ms2_no_prec[spectrum_idx]
  
  # get experimental spectrum
  exp_idx = which(grepl(paste0('^ID=', current$peak_id, '_mz'), exp_ms2$TITLE))
  if (length(spectrum_idx) == 0) {
    message("no experimental MS/MS for peak ", current$peak_id)  
    return(current)
  }
  exp_spectrum = exp_ms2[exp_idx]
  exp_spectrum_no_prec = exp_ms2_no_prec[exp_idx]
  
  # calculate dot-product
  dp = compareSpectra(pred_spectrum, exp_spectrum, ppm = 50) %>% 
    # if more than one experimental and/or predicted spectrum, keep higher value
    # replace with zero if not defined
    max(0, na.rm = TRUE)
  
  # calculate number of matching peaks (ignoring precursor)
  n_overlap = compareSpectra(pred_spectrum_no_prec, exp_spectrum_no_prec, 
                             ppm = 50, type = "inner", 
                             FUN = function(x, y, ...) nrow(x)) %>% 
    # if more than one experimental and/or predicted spectrum, keep higher value
    max(na.rm = TRUE)
  
  return(current %>% 
           mutate(dot_product = dp, n_overlap = n_overlap))
}, .progress = TRUE)

# confirm n_overlap is integer
stopifnot(all(ms2_score_df$n_overlap %% 1 == 0, na.rm = TRUE))
# confirm dot-product is [0, 1]
stopifnot(all(between(ms2_score_df$dot_product, 0, 1), na.rm = TRUE))

# write output: all meta-learning features except Isopleth cosine
write.csv(ms2_score_df, args$output_file, row.names = FALSE)
