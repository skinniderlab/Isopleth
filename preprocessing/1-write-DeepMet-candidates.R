# Extract DeepMet structure candidates for each mouse tissue peak
setwd("Isopleth")
library(dplyr)
library(future)
library(furrr)
library(CHNOSZ)
library(OrgMassSpecR)

# Load files (DeepMet library is available for download via Zenodo: https://doi.org/10.5281/zenodo.16813151)
pks = read.csv("data/preprocessing_inputs/mouse_tissue_metabolomics_dataset.csv") # metabolomics data
standards = read.csv("data/preprocessing_inputs/matched_standards_to_mouse.csv") # 231 matched standards
deepmet = read.csv("data/preprocessing_inputs/generated-molecules-min_freq=5.csv.gz") # deepmet library, min frequency = 5
ann_file = "data/preprocessing_inputs/annotation_file.csv" # output

# function to calculate ppm range
calc_ppm_range = function(theor_mass, err_ppm = 10) {
  c(
    (-err_ppm / 1e6 * theor_mass) + theor_mass,
    (err_ppm / 1e6 * theor_mass) + theor_mass
  )
}

# function to calculate neutral mass from a known formula
get_neutral_mass = function(formula) {
  fmla = CHNOSZ::makeup(formula)
  exact_mass = OrgMassSpecR::MonoisotopicMass(fmla)
  if (grepl("\\+$", formula)) {
    # deprotonate 
    exact_mass = exact_mass - 1.007276
  } else if (grepl("\\-", formula)) {
    stop("not implemented: negative charge")
  } else if (grepl("\\+", formula)) {
    stop("not implemented: multiple positive charges")
  } else {
    # neutral formula - do nothing
  }
  return(exact_mass)
}

standard_formulas = standards %>%
  distinct(peak_id, Formula) %>%
  dplyr::rename(standard_formula = Formula) %>%
  mutate(formula_mass = vapply(standard_formula, get_neutral_mass, numeric(1)))
stopifnot(nrow(standard_formulas) == 231,
          n_distinct(standard_formulas$peak_id) == 231,
          !any(is.na(standard_formulas$standard_formula)))

# iterate over peaks
split = pks %>%
  distinct(peak_id, mz) %>%
  left_join(standard_formulas, by = "peak_id") %>%
  mutate(is_standard = !is.na(standard_formula),
         lookup_mass = ifelse(is_standard, formula_mass, mz + 1.007276),
         lookup_ppm = ifelse(is_standard, 5, 5.05)) %>% # +0.05 err rate for FraGNNet 
  split(.$peak_id)
plan(multisession, workers = 12)
options(future.globals.maxSize = 1000 * 1024^2)
matches = future_map_dfr(split, ~ {
  current = .x
  stopifnot(n_distinct(current$mz) == 1 &
              n_distinct(current$peak_id) == 1 &
              n_distinct(current$lookup_mass) == 1 &
              n_distinct(current$lookup_ppm) == 1)
  neutral_mass = dplyr::first(current$lookup_mass)
  ppm_error = dplyr::first(current$lookup_ppm)
  range = calc_ppm_range(neutral_mass, ppm_error)
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
  current = dplyr::select(current, peak_id, mz)
  cbind(current, x)
}, .progress = TRUE)

write.csv(matches, ann_file, row.names = FALSE)
