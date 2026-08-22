# Write DeepMet candidates for mouse tissue peaks.
library(argparse)
library(tidyverse)
library(magrittr)
library(data.table)
library(readxl)
library(furrr)

# dynamically parse arguments
parser = ArgumentParser(prog = 'write-DeepMet-candidates.R')
grid = read.delim("sh/grids/write-DeepMet-candidates.txt")
for (param_name in colnames(grid))
  parser$add_argument(paste0('--', param_name),
                      type = typeof(grid[[param_name]]))
args = parser$parse_args()
print(args)

# load metadata for all 8277 peaks
pks = read_excel(args$pks_file)

# read DeepMet data, min freq = 5
deepmet = fread(args$deepmet_file, data.table = FALSE)

# function to calculate ppm range
calc_ppm_range = function(theor_mass, err_ppm = 10) {
  c(
    (-err_ppm / 1e6 * theor_mass) + theor_mass,
    (err_ppm / 1e6 * theor_mass) + theor_mass
  )
}

# iterate over peaks
split = pks %>% 
  distinct(peak_id, medMz) %>% 
  split(.$peak_id)
plan(multisession, workers = 12)
options(future.globals.maxSize = 1000 * 1024^2)
matches = future_map_dfr(split, ~ {
  current = .x
  stopifnot(n_distinct(current$medMz) == 1 & 
              n_distinct(current$peak_id) == 1)
  neutral_mass = dplyr::first(current$medMz + 1.007276)
  range = calc_ppm_range(neutral_mass, 5.1) ## 0.1 ppm buffer for FraGNNet
  m = filter(deepmet, between(Mass, range[1], range[2])) %>% 
    mutate(rank = row_number(),
           confidence = Frequency / sum(Frequency)) %>% 
    arrange(desc(Frequency))
  if (nrow(m) == 0) return(data.frame())
  m0 = m %>% 
    dplyr::rename(structure_rank = rank,
                  structure_size = Frequency, 
                  structure_confidence = confidence)
  f = m %>% 
    mutate(Mass = as.character(Mass)) %>% 
    group_by(Formula, Mass) %>% 
    summarise(size = sum(Frequency)) %>% 
    ungroup() %>% 
    arrange(desc(size)) %>% 
    mutate(confidence = size / sum(size), rank = row_number()) 
  f0 = dplyr::select(f, -Mass) %>% 
    dplyr::rename(formula_rank = rank,
                  formula_size = size, 
                  formula_confidence = confidence)
  x = left_join(m0, f0, by = 'Formula')
  cbind(current, x)
}, .progress = TRUE)

write.csv(matches, args$ann_file, row.names = FALSE)
