#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply analyte/IS calibration curves to long-format sample data "
            "and compute back-calculated concentrations + bias for each analyte component "
            "(can include both quantifiers and qualifiers)."
        )
    )
    parser.add_argument("curves_csv", help="CSV with fitted curves (from build_curves_from_long.py)")
    parser.add_argument("samples_csv", help="Long-format CSV with samples (QCs, unknowns, etc.)")

    # --- curves file column names ---
    parser.add_argument("--curves_analyte_col", default="analyte_component",
                        help="Column in curves CSV with analyte component names.")
    parser.add_argument("--curves_is_col", default="internal_standard",
                        help="Column in curves CSV with internal standard names.")
    parser.add_argument("--curves_slope_col", default="slope",
                        help="Column in curves CSV with slope.")
    parser.add_argument("--curves_intercept_col", default="intercept",
                        help="Column in curves CSV with intercept.")
    parser.add_argument("--curves_pass_col", default="passes_thresholds",
                        help="Pass/fail column in curves CSV (if present).")

    # Curve LoQ columns (from curves file)
    parser.add_argument("--curves_lloq_col", default="lloq_conc",
                        help="Column in curves CSV containing LLoQ for the fitted curve.")
    parser.add_argument("--curves_uloq_col", default="uloq_conc",
                        help="Column in curves CSV containing ULoQ for the fitted curve.")

    # --- sample file column names ---
    parser.add_argument("--sample_index_col", default="Sample Index",
                        help="Sample index column name.")
    parser.add_argument("--sample_name_col", default="Sample Name",
                        help="Sample name column name.")
    parser.add_argument("--sample_type_col", default="Sample Type",
                        help="Sample Type column name (e.g. Standard, QC, Unknown).")
    parser.add_argument("--component_type_col", default="Component Type",
                        help="Component Type column name (Quantifiers, Qualifiers, Internal Standards, etc.).")
    parser.add_argument("--component_name_col", default="Component Name",
                        help="Component Name column name (transition).")
    parser.add_argument("--area_col", default="Area",
                        help="Area/intensity column name.")
    parser.add_argument("--actual_conc_col", default="Actual Concentration",
                        help="True/nominal concentration column name.")

    # NEW: original LoQs columns (from samples file)
    parser.add_argument("--original_lloq_col", default="LLoQ",
                        help="Column in samples CSV containing the original analyte LLoQ.")
    parser.add_argument("--original_uloq_col", default="ULoQ",
                        help="Column in samples CSV containing the original analyte ULoQ.")

    # Which component types count as "analytes" (for true conc)
    parser.add_argument(
        "--analyte_component_types",
        nargs="+",
        default=["Quantifiers"],  # you can pass Quantifiers Qualifiers
        help=(
            "Component Type values that should be treated as analytes for true concentrations "
            "(e.g. 'Quantifiers', 'Qualifiers')."
        ),
    )

    # Optionally filter by sample type (e.g. only QCs)
    parser.add_argument(
        "--sample_types_to_eval",
        nargs="+",
        help=(
            "Optional list of Sample Type values to evaluate (e.g. 'QC' 'Unknown'). "
            "If omitted, all sample types are used."
        ),
    )

    parser.add_argument(
        "--only_passed",
        action="store_true",
        help="If set, only apply curves where passes_thresholds == True.",
    )

    parser.add_argument("--output", default="qc_curve_application_long.csv",
                        help="Output CSV filename.")

    args = parser.parse_args()

    # --- Load data ---
    curves = pd.read_csv(args.curves_csv)
    samples = pd.read_csv(args.samples_csv)

    # Filter curves if requested
    if args.only_passed and args.curves_pass_col in curves.columns:
        curves = curves[curves[args.curves_pass_col] == True].copy()

    # Optionally filter samples by Sample Type (e.g. only QCs)
    if args.sample_types_to_eval is not None:
        samples = samples[samples[args.sample_type_col].isin(args.sample_types_to_eval)].copy()

    # Build wide area table for ALL components (analytes + IS)
    idx_cols = [args.sample_index_col, args.sample_name_col]
    area_wide = samples.pivot_table(
        index=idx_cols,
        columns=args.component_name_col,
        values=args.area_col,
        aggfunc="first",
    )

    # Rows corresponding to analyte components (quantifiers/qualifiers)
    analyte_rows = samples[samples[args.component_type_col].isin(args.analyte_component_types)].copy()

    # Build wide "true concentration" table for analyte components
    conc_wide = analyte_rows.pivot_table(
        index=idx_cols,
        columns=args.component_name_col,
        values=args.actual_conc_col,
        aggfunc="first",
    )

    # NEW: Build wide original LoQ tables from the samples file (if present)
    if args.original_lloq_col in analyte_rows.columns:
        orig_lloq_wide = analyte_rows.pivot_table(
            index=idx_cols,
            columns=args.component_name_col,
            values=args.original_lloq_col,
            aggfunc="first",
        )
    else:
        orig_lloq_wide = pd.DataFrame(index=area_wide.index)

    if args.original_uloq_col in analyte_rows.columns:
        orig_uloq_wide = analyte_rows.pivot_table(
            index=idx_cols,
            columns=args.component_name_col,
            values=args.original_uloq_col,
            aggfunc="first",
        )
    else:
        orig_uloq_wide = pd.DataFrame(index=area_wide.index)

    results = []

    for _, row in curves.iterrows():
        analyte_name = row[args.curves_analyte_col]
        is_name = row[args.curves_is_col]
        m = row[args.curves_slope_col]
        b = row[args.curves_intercept_col]

        # Curve LoQs (from curves file; may be non-numeric)
        curve_lloq = pd.to_numeric(row.get(args.curves_lloq_col, np.nan), errors="coerce")
        curve_uloq = pd.to_numeric(row.get(args.curves_uloq_col, np.nan), errors="coerce")

        # Skip if slope is missing/non-finite or basically zero
        if not np.isfinite(m) or abs(m) < 1e-12:
            continue

        # Need analyte and IS areas present in the sample file
        if analyte_name not in area_wide.columns or is_name not in area_wide.columns:
            continue

        ana_area = pd.to_numeric(area_wide[analyte_name], errors="coerce")
        is_area = pd.to_numeric(area_wide[is_name], errors="coerce")

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ana_area / is_area

        # True concentration for this analyte (if present)
        if analyte_name in conc_wide.columns:
            true_conc = pd.to_numeric(
                conc_wide[analyte_name].reindex(area_wide.index),
                errors="coerce",
            )
        else:
            true_conc = pd.Series(np.nan, index=area_wide.index)

        with np.errstate(divide="ignore", invalid="ignore"):
            calc_conc = (ratio - b) / m

        with np.errstate(divide="ignore", invalid="ignore"):
            bias_pct = (calc_conc - true_conc) / true_conc * 100.0

        # Used flag for curve LoQs: within-range AND has a valid calculated concentration
        used_within_curve = pd.Series(False, index=area_wide.index)
        if np.isfinite(curve_lloq) and np.isfinite(curve_uloq):
            within_curve = (true_conc >= curve_lloq) & (true_conc <= curve_uloq)
            has_calc = np.isfinite(calc_conc)
            used_within_curve = (within_curve & has_calc).fillna(False)

        # NEW: original LoQs from samples file (per sample/analyte; often constant but not assumed)
        if analyte_name in orig_lloq_wide.columns:
            orig_lloq = pd.to_numeric(orig_lloq_wide[analyte_name].reindex(area_wide.index), errors="coerce")
        else:
            orig_lloq = pd.Series(np.nan, index=area_wide.index)

        if analyte_name in orig_uloq_wide.columns:
            orig_uloq = pd.to_numeric(orig_uloq_wide[analyte_name].reindex(area_wide.index), errors="coerce")
        else:
            orig_uloq = pd.Series(np.nan, index=area_wide.index)

        # Used flag for original LoQs: within-range AND has a valid calculated concentration
        used_within_original = pd.Series(False, index=area_wide.index)
        within_original = (true_conc >= orig_lloq) & (true_conc <= orig_uloq)
        has_calc = np.isfinite(calc_conc)
        used_within_original = (within_original & has_calc).fillna(False)

        idx = area_wide.index
        out = pd.DataFrame({
            args.sample_index_col: idx.get_level_values(args.sample_index_col),
            args.sample_name_col: idx.get_level_values(args.sample_name_col),
            "analyte_component": analyte_name,
            "internal_standard": is_name,
            "slope": m,
            "intercept": b,
            "ratio": ratio.values,
            "calc_conc": calc_conc.values,
            "true_conc": true_conc.values,
            "bias_pct": bias_pct.values,

            # Curve LoQs (from curves file)
            "curve_LLoQ": curve_lloq,
            "curve_ULoQ": curve_uloq,
            "used_within_curve_LLoQ_ULoQ": used_within_curve.values,

            # NEW: Original LoQs (from samples file)
            "original_LLoQ": orig_lloq.values,
            "original_ULoQ": orig_uloq.values,
            "used_within_original_LLoQ_ULoQ": used_within_original.values,
        })

        if args.curves_pass_col in curves.columns:
            out["curve_passes_thresholds"] = row[args.curves_pass_col]

        results.append(out)

    if not results:
        print("No analyte–IS combinations could be evaluated.")
        return

    final = pd.concat(results, ignore_index=True)
    final.to_csv(args.output, index=False)
    print(f"Saved QC application results to {args.output}")


if __name__ == "__main__":
    main()
