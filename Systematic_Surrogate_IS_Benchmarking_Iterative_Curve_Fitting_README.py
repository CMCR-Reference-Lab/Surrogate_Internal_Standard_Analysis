#!/usr/bin/env python3
"""
# `Systematic_Surrogate_IS_Benchmarking_Iterative_Curve_Fitting.py`

## Overview

`Systematic_Surrogate_IS_Benchmarking_Iterative_Curve_Fitting.py` builds analyte/internal-standard (IS)
calibration curves from long-format LC-MRM data using a two-stage workflow.

The final calibration model in Stage 2 is:

- `Ratio = Analyte Area / IS Area`
- fit as `Ratio = m * Conc + b`

where `Conc` is analyte actual concentration.

Unlike the earlier single-stage description, this version uses:

1. Stage 1 analyte-only selection (`Area vs Conc`) to choose concentration levels.
2. Stage 2 IS-normalized fitting (`Ratio vs Conc`) restricted to Stage 1 levels.

## Command-Line Usage

```bash
python3 Systematic_Surrogate_IS_Benchmarking_Iterative_Curve_Fitting.py data_file.csv \
  --min_points 5 \
  --max_abs_bias 20 \
  --min_r2 0.99 \
  --weighting 1/x \
  --search_mode greedy \
  --output analyte_multi_IS_fits_iterative.csv
```

## Positional Argument

- `input_csv`: Long-format CSV containing at least calibrator rows

## Options

### Input/Output

- `--output` (default: `analyte_multi_IS_fits_iterative.csv`)
  - Output path is cleaned by `safe_output_path()` to remove trailing `/` or `\`.

### Column Name Overrides

- `--sample_type_col` (default: `Sample Type`)
- `--component_type_col` (default: `Component Type`)
- `--component_name_col` (default: `Component Name`)
- `--component_group_col` (default: `Component Group Name`)
- `--actual_conc_col` (default: `Actual Concentration`)
- `--area_col` (default: `Area`)
- `--sample_index_col` (default: `Sample Index`)
- `--sample_name_col` (default: `Sample Name`)

### Label Overrides

- `--cal_sample_type` (default: `Standard`)
- `--analyte_type_label` (default: `Analyte`)
- `--is_type_label` (default: `Internal Standards`)

### Fitting / Selection Controls

- `--min_points` (default: `5`)
- `--max_abs_bias` (default: `20.0`)
- `--min_r2` (default: `0.99`)
- `--weighting` (default: `1/x`; choices: `1/x`, `1/x2`, `none`)
- `--search_mode` (default: `greedy`; choices: `greedy`, `contiguous`, `exhaustive`)
- `--exhaustive_max_n` (default: `18`)
  - Exhaustive mode automatically falls back to contiguous when point count exceeds this.

## Processing Workflow

1. Read CSV and filter calibrator rows:
   - Keep rows where `Sample Type == --cal_sample_type`.
2. Split calibrator rows into analyte and IS tables by component-type labels.
3. Pivot to wide matrices by `(Sample Index, Sample Name)`:
   - analyte area
   - analyte concentration
   - IS area
4. Build analyte-group lookup from component name to group name.
5. Stage 1 (per analyte): select concentration levels from `Area vs Conc` using
   `select_calibrator_levels(...)` under chosen search strategy.
6. Stage 2 (per analyte x IS): compute ratio and fit with
   `fit_curve_best_subset(...)` using Stage 1 selected concentrations as
   allowed concentration levels (greedy in Stage 2).
7. Write one output row per analyte x IS pair.

## Search Modes

- `greedy`: iteratively drop the worst absolute-bias point
- `contiguous`: test all contiguous windows in sorted concentration order
- `exhaustive`: test all subsets (true optimum), with safety fallback

Selection preference among passing candidates prioritizes:

1. more points
2. larger concentration range (`uloq - lloq`)
3. lower max absolute bias
4. higher R2
5. lower normalized RMSE

## Output

### Output File

- Default: `analyte_multi_IS_fits_iterative.csv`
- Set via `--output`

### Output Columns

Each row corresponds to one analyte x IS pairing:

- `analyte_component`
- `analyte_group`
- `internal_standard`
- `n_points`
- `slope`
- `intercept`
- `r2`
- `rmse_ratio`
- `mean_abs_bias_pct`
- `max_abs_bias_pct`
- `passes_thresholds`
- `lloq_conc`
- `uloq_conc`
- `search_mode`
- `n_points_start`
- `n_points_stage1`

## Notes and Assumptions

- Only calibrators are used for curve building.
- QCs can exist in the input but are ignored during fit-building.
- Concentration filtering requires finite `x` and `y`, and `x > 0`.
- `pivot_table(..., aggfunc="first")` is used for duplicate-safe reshaping.
- `rmse_ratio` is RMSE divided by mean response and may be unstable when mean
  response is near zero.
- If a pair has too few usable points, it is still reported with failed/NaN metrics.
"""
