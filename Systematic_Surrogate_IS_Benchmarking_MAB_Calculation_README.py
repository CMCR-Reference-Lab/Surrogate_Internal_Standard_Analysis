#!/usr/bin/env python3
"""
# `Systematic_Surrogate_IS_Benchmarking_MAB_Calculation.py`

## Overview

`Systematic_Surrogate_IS_Benchmarking_MAB_Calculation.py` applies previously fitted analyte/internal-standard (IS)
calibration curves to long-format sample data (QCs, unknowns, etc.) and computes:

- Area ratio (`analyte_area / is_area`)
- Back-calculated concentration:
  `calc_conc = (ratio - intercept) / slope`
- Percent bias relative to nominal concentration:
  `bias_pct = (calc_conc - true_conc) / true_conc * 100`

The script supports both quantifier and qualifier analyte components and adds
LoQ-tracking outputs using:

- Curve LoQs from the curves CSV (`lloq_conc`, `uloq_conc` by default)
- Original LoQs from the samples CSV (`LLoQ`, `ULoQ` by default)

## Command-Line Usage

```bash
python3 Systematic_Surrogate_IS_Benchmarking_MAB_Calculation.py CURVES.csv SAMPLES.csv \
  --analyte_component_types Quantifiers Qualifiers \
  --only_passed \
  --output qc_quant_and_qual.csv
```

## Positional Arguments

- `curves_csv`: CSV with fitted curves (typically from a curve-building script)
- `samples_csv`: Long-format CSV with sample-level component rows

## Options

### Curves File Column Options

- `--curves_analyte_col` (default: `analyte_component`)
- `--curves_is_col` (default: `internal_standard`)
- `--curves_slope_col` (default: `slope`)
- `--curves_intercept_col` (default: `intercept`)
- `--curves_pass_col` (default: `passes_thresholds`)
- `--curves_lloq_col` (default: `lloq_conc`)
- `--curves_uloq_col` (default: `uloq_conc`)

### Samples File Column Options

- `--sample_index_col` (default: `Sample Index`)
- `--sample_name_col` (default: `Sample Name`)
- `--sample_type_col` (default: `Sample Type`)
- `--component_type_col` (default: `Component Type`)
- `--component_name_col` (default: `Component Name`)
- `--area_col` (default: `Area`)
- `--actual_conc_col` (default: `Actual Concentration`)
- `--original_lloq_col` (default: `LLoQ`)
- `--original_uloq_col` (default: `ULoQ`)

### Analysis Behavior Options

- `--analyte_component_types` (default: `Quantifiers`)
  - Component Type values treated as analytes
  - Can include both `Quantifiers` and `Qualifiers`
- `--sample_types_to_eval`
  - Optional subset of `Sample Type` values to evaluate
- `--only_passed`
  - If provided, uses only curves with `passes_thresholds == True`
- `--output` (default: `qc_curve_application_long.csv`)

## Processing Workflow

1. Load curves and sample files.
2. Optionally filter curves (`--only_passed`) and sample types.
3. Pivot sample areas to wide format for all components.
4. Build analyte-only wide tables for:
   - true concentration
   - original LLoQ
   - original ULoQ
5. For each analyte/IS curve:
   - Skip invalid slopes (missing/non-finite/near-zero).
   - Skip pairs missing analyte or IS area columns.
   - Compute ratio, `calc_conc`, and `bias_pct`.
   - Compute curve-LoQ usage flag:
     `used_within_curve_LLoQ_ULoQ`
   - Compute original-LoQ usage flag:
     `used_within_original_LLoQ_ULoQ`
6. Concatenate all per-curve results to long-format output CSV.

## Output Columns

Core output columns:

- `Sample Index`
- `Sample Name`
- `analyte_component`
- `internal_standard`
- `slope`
- `intercept`
- `ratio`
- `calc_conc`
- `true_conc`
- `bias_pct`

Curve-LoQ columns:

- `curve_LLoQ`
- `curve_ULoQ`
- `used_within_curve_LLoQ_ULoQ`

Original-LoQ columns:

- `original_LLoQ`
- `original_ULoQ`
- `used_within_original_LLoQ_ULoQ`

Optional pass/fail column (if present in curves input):

- `curve_passes_thresholds`

## Important Notes

- Sample data must be in long format.
- Each analyte/IS pair should be represented in the curves file.
- Missing analyte/IS areas for a pair are skipped for that pair.
- Bias is only meaningful when `true_conc` exists and is non-zero.
- If no analyte/IS combinations can be evaluated, the script prints a message
  and exits without writing output.
"""
