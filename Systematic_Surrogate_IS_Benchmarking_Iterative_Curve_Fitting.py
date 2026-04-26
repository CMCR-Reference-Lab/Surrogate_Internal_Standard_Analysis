#!/usr/bin/env python3
import argparse
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]

# To run: python3 script_name.py data_file.csv \
#  --min_points 6 \
#  --max_abs_bias 20 \
#  --min_r2 0.995 \
#  --weighting 1/x \
#  --output analyte_multi_IS_fits_iterative.csv

def safe_output_path(p: str) -> str:
    """
    Prevent the common Windows/Spyder mistake where a trailing backslash
    accidentally gets appended (e.g., 'file.csv\\').
    """
    if p is None:
        return p
    return p.strip().strip('"').strip("'").rstrip("\\/")  # remove trailing slash/backslash


def wls_fit(x: np.ndarray, y: np.ndarray, weighting: str):
    """
    Weighted least squares linear fit: y = m*x + b
    weighting in {"1/x", "1/x2", "none"}
    Returns (m, b, r2, rmse_ratio)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if weighting == "1/x":
        w = 1.0 / x
        w[~np.isfinite(w)] = 1.0
    elif weighting == "1/x2":
        w = 1.0 / (x ** 2)
        w[~np.isfinite(w)] = 1.0
    else:
        w = None

    if w is not None:
        m, b = np.polyfit(x, y, 1, w=w)
    else:
        m, b = np.polyfit(x, y, 1)

    y_pred = m * x + b
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    rmse = float(np.sqrt(ss_res / len(y))) if len(y) else np.nan
    mean_y = float(np.mean(y)) if len(y) else np.nan
    rmse_ratio = rmse / mean_y if (mean_y != 0 and np.isfinite(mean_y)) else np.nan

    return float(m), float(b), float(r2), float(rmse_ratio)


def compute_curve_metrics(
    x: np.ndarray,
    y: np.ndarray,
    min_points: int,
    max_abs_bias_thresh: float,
    min_r2_thresh: float,
    weighting: str,
):
    """
    Fit and score a fixed subset (single pass; no trimming):
      - Fit ratio vs conc: y = m*x + b
      - Back-calc x_hat = (y-b)/m
      - Compute bias %, R2, etc.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # Basic cleaning
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[keep]
    y = y[keep]

    if len(x) < min_points:
        return {
            "n_points": int(len(x)),
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "rmse_ratio": np.nan,
            "mean_abs_bias_pct": np.nan,
            "max_abs_bias_pct": np.nan,
            "passes_thresholds": False,
            "lloq_conc": np.nan,
            "uloq_conc": np.nan,
        }

    m, b, r2, rmse_ratio = wls_fit(x, y, weighting=weighting)

    if m == 0 or not np.isfinite(m):
        return {
            "n_points": int(len(x)),
            "slope": float(m),
            "intercept": float(b),
            "r2": np.nan,
            "rmse_ratio": float(rmse_ratio) if np.isfinite(rmse_ratio) else np.nan,
            "mean_abs_bias_pct": np.nan,
            "max_abs_bias_pct": np.nan,
            "passes_thresholds": False,
            "lloq_conc": float(np.nanmin(x)) if len(x) else np.nan,
            "uloq_conc": float(np.nanmax(x)) if len(x) else np.nan,
        }

    x_calc = (y - b) / m
    with np.errstate(divide="ignore", invalid="ignore"):
        bias_pct = (x_calc - x) / x * 100.0

    bias_pct = bias_pct[np.isfinite(bias_pct)]
    if len(bias_pct) < min_points:
        return {
            "n_points": int(len(x)),
            "slope": float(m),
            "intercept": float(b),
            "r2": float(r2),
            "rmse_ratio": float(rmse_ratio),
            "mean_abs_bias_pct": np.nan,
            "max_abs_bias_pct": np.nan,
            "passes_thresholds": False,
            "lloq_conc": float(np.nanmin(x)) if len(x) else np.nan,
            "uloq_conc": float(np.nanmax(x)) if len(x) else np.nan,
        }

    abs_bias = np.abs(bias_pct)
    mean_abs_bias = float(np.nanmean(abs_bias))
    max_abs_bias = float(np.nanmax(abs_bias))

    passes_bias = max_abs_bias <= float(max_abs_bias_thresh)
    passes_r2 = (float(r2) >= float(min_r2_thresh)) if np.isfinite(r2) else False

    return {
        "n_points": int(len(x)),
        "slope": float(m),
        "intercept": float(b),
        "r2": float(r2),
        "rmse_ratio": float(rmse_ratio),
        "mean_abs_bias_pct": mean_abs_bias,
        "max_abs_bias_pct": max_abs_bias,
        "passes_thresholds": bool(passes_bias and passes_r2),
        "lloq_conc": float(np.nanmin(x)) if len(x) else np.nan,
        "uloq_conc": float(np.nanmax(x)) if len(x) else np.nan,
    }


def _score_metrics_for_selection(metrics: dict):
    """
    Sort key for "best" passing curve.
    Primary goal: keep as many points as possible, then maximize range, then minimize max bias,
    then maximize R2, then minimize rmse_ratio.
    """
    n = metrics.get("n_points", 0) or 0
    lloq = metrics.get("lloq_conc", np.nan)
    uloq = metrics.get("uloq_conc", np.nan)
    rng = float(uloq - lloq) if (np.isfinite(lloq) and np.isfinite(uloq)) else -np.inf
    max_bias = metrics.get("max_abs_bias_pct", np.inf)
    r2 = metrics.get("r2", -np.inf)
    rmse_ratio = metrics.get("rmse_ratio", np.inf)

    # Higher n and range are better -> negate for ascending sort
    return (-int(n), -float(rng), float(max_bias), -float(r2), float(rmse_ratio))


def select_calibrator_levels(
    x,
    y,
    min_points=5,
    max_abs_bias_thresh=20.0,
    min_r2_thresh=0.995,
    weighting="1/x",
    search_mode="greedy",
    exhaustive_max_n=18,
    max_iterations=50,
):
    """
    Select optimal calibrator concentration levels (x values) based on y vs x fit.
    Returns a set of selected x values (concentrations) that passed the criteria.
    """
    # Clean input
    data = pd.DataFrame({"x": x, "y": y}).copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    data = data[data["x"] > 0].reset_index(drop=True)
    
    if len(data) < min_points:
        return set()
    
    n0 = len(data)
    
    # ----- Greedy -----
    if search_mode == "greedy":
        iteration = 0
        while iteration < max_iterations and len(data) >= min_points:
            iteration += 1
            
            metrics = compute_curve_metrics(
                data["x"].to_numpy(float),
                data["y"].to_numpy(float),
                min_points=min_points,
                max_abs_bias_thresh=max_abs_bias_thresh,
                min_r2_thresh=min_r2_thresh,
                weighting=weighting,
            )
            
            if metrics["passes_thresholds"]:
                return set(data["x"].to_numpy(float))
            
            # Drop worst bias point
            m = metrics.get("slope", np.nan)
            b = metrics.get("intercept", np.nan)
            if not np.isfinite(m) or m == 0:
                break
            
            x_vals = data["x"].to_numpy(float)
            y_vals = data["y"].to_numpy(float)
            x_calc = (y_vals - b) / m
            with np.errstate(divide="ignore", invalid="ignore"):
                bias_pct = (x_calc - x_vals) / x_vals * 100.0
            
            valid = np.isfinite(bias_pct)
            if valid.sum() < min_points:
                break
            
            worst_idx = int(np.argmax(np.abs(bias_pct[valid])))
            valid_idx = np.flatnonzero(valid)[worst_idx]
            data = data.drop(data.index[valid_idx]).reset_index(drop=True)
        
        # Return whatever is left
        return set(data["x"].to_numpy(float)) if len(data) >= min_points else set()
    
    # ----- Contiguous ranges -----
    if search_mode == "contiguous":
        data_sorted = data.sort_values("x", ascending=True).reset_index(drop=True)
        xs = data_sorted["x"].to_numpy(float)
        ys = data_sorted["y"].to_numpy(float)
        n = len(xs)
        
        best_passing = None
        best_passing_indices = None
        
        # Search by window size from n down to min_points
        for size in range(n, min_points - 1, -1):
            for start in range(0, n - size + 1):
                end = start + size
                metrics = compute_curve_metrics(
                    xs[start:end],
                    ys[start:end],
                    min_points=min_points,
                    max_abs_bias_thresh=max_abs_bias_thresh,
                    min_r2_thresh=min_r2_thresh,
                    weighting=weighting,
                )
                if metrics["passes_thresholds"]:
                    if best_passing is None or _score_metrics_for_selection(metrics) < _score_metrics_for_selection(best_passing):
                        best_passing = metrics
                        best_passing_indices = (start, end)
            if best_passing is not None:
                return set(xs[best_passing_indices[0]:best_passing_indices[1]])
        
        # No passing subset found
        return set()
    
    # ----- Exhaustive -----
    if search_mode == "exhaustive":
        xs = data["x"].to_numpy(float)
        ys = data["y"].to_numpy(float)
        n = len(xs)
        
        if n > int(exhaustive_max_n):
            # Fallback to contiguous
            return select_calibrator_levels(
                xs, ys, min_points, max_abs_bias_thresh, min_r2_thresh,
                weighting, "contiguous", exhaustive_max_n, max_iterations
            )
        
        from itertools import combinations
        
        best_passing = None
        best_passing_indices = None
        
        idx = np.arange(n)
        for size in range(n, min_points - 1, -1):
            for comb in combinations(idx, size):
                sel = np.fromiter(comb, dtype=int)
                metrics = compute_curve_metrics(
                    xs[sel],
                    ys[sel],
                    min_points=min_points,
                    max_abs_bias_thresh=max_abs_bias_thresh,
                    min_r2_thresh=min_r2_thresh,
                    weighting=weighting,
                )
                if metrics["passes_thresholds"]:
                    if best_passing is None or _score_metrics_for_selection(metrics) < _score_metrics_for_selection(best_passing):
                        best_passing = metrics
                        best_passing_indices = sel
            if best_passing is not None:
                return set(xs[best_passing_indices])
        
        return set()
    
    raise ValueError(f"Unknown search_mode: {search_mode}")


def fit_curve_best_subset(
    x,
    y,
    min_points=5,
    max_abs_bias_thresh=20.0,
    min_r2_thresh=0.995,
    weighting="1/x",
    search_mode="greedy",
    exhaustive_max_n=18,
    max_iterations=50,
    allowed_concentrations=None,
):
    """
    Fit curve with one of several subset selection strategies:

    - search_mode="greedy": current behavior (iteratively drop worst-bias point)
    - search_mode="contiguous": try all contiguous ranges in x-sorted order (O(n^2))
    - search_mode="exhaustive": try all subsets (true optimum), with early-stop by subset size.
        Safety: if n > exhaustive_max_n, falls back to "contiguous".

    If allowed_concentrations is provided (set of x values), only points with x in that set
    will be used for fitting (no further trimming).

    Returns metrics dict (same shape as compute_curve_metrics + strategy fields).
    """
    # Clean input once, keep arrays aligned
    data = pd.DataFrame({"x": x, "y": y}).copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    data = data[data["x"] > 0]
    
    # If allowed_concentrations provided, filter to only those levels
    if allowed_concentrations is not None and len(allowed_concentrations) > 0:
        # Convert to set for fast lookup, then use tolerance-based matching
        allowed_set = set(allowed_concentrations)
        x_vals = data["x"].to_numpy(float)
        
        # For each x value, check if it's close to any allowed concentration
        # (handles floating point precision issues)
        mask = np.array([
            any(np.isclose(x_val, allowed_x, rtol=1e-9, atol=1e-9) 
                for allowed_x in allowed_set)
            for x_val in x_vals
        ])
        
        data = data[mask].copy()
        if len(data) == 0:
            # No overlap - return failed metrics
            m = compute_curve_metrics(
                np.array([]),
                np.array([]),
                min_points=min_points,
                max_abs_bias_thresh=max_abs_bias_thresh,
                min_r2_thresh=min_r2_thresh,
                weighting=weighting,
            )
            m["search_mode"] = search_mode
            m["n_points_start"] = 0
            return m

    n0 = len(data)
    if n0 < min_points:
        m = compute_curve_metrics(
            data["x"].to_numpy(float),
            data["y"].to_numpy(float),
            min_points=min_points,
            max_abs_bias_thresh=max_abs_bias_thresh,
            min_r2_thresh=min_r2_thresh,
            weighting=weighting,
        )
        m["search_mode"] = search_mode
        m["n_points_start"] = int(n0)
        return m

    # ----- Greedy (existing) -----
    if search_mode == "greedy":
        iteration = 0
        last_metrics = None
        while iteration < max_iterations and len(data) >= min_points:
            iteration += 1

            metrics = compute_curve_metrics(
                data["x"].to_numpy(float),
                data["y"].to_numpy(float),
                min_points=min_points,
                max_abs_bias_thresh=max_abs_bias_thresh,
                min_r2_thresh=min_r2_thresh,
                weighting=weighting,
            )
            last_metrics = metrics
            if metrics["passes_thresholds"]:
                metrics["search_mode"] = "greedy"
                metrics["n_points_start"] = int(n0)
                return metrics

            # Identify worst bias point within current subset
            m = metrics.get("slope", np.nan)
            b = metrics.get("intercept", np.nan)
            if not np.isfinite(m) or m == 0:
                metrics["search_mode"] = "greedy"
                metrics["n_points_start"] = int(n0)
                return metrics

            x_vals = data["x"].to_numpy(float)
            y_vals = data["y"].to_numpy(float)
            x_calc = (y_vals - b) / m
            with np.errstate(divide="ignore", invalid="ignore"):
                bias_pct = (x_calc - x_vals) / x_vals * 100.0

            valid = np.isfinite(bias_pct)
            if valid.sum() < min_points:
                metrics["search_mode"] = "greedy"
                metrics["n_points_start"] = int(n0)
                return metrics

            worst_idx = int(np.argmax(np.abs(bias_pct[valid])))
            valid_idx = np.flatnonzero(valid)[worst_idx]
            data = data.drop(data.index[valid_idx])

        # Fell out (too few points / iterations). Return last metrics.
        if last_metrics is None:
            last_metrics = compute_curve_metrics(
                data["x"].to_numpy(float),
                data["y"].to_numpy(float),
                min_points=min_points,
                max_abs_bias_thresh=max_abs_bias_thresh,
                min_r2_thresh=min_r2_thresh,
                weighting=weighting,
            )
        last_metrics["search_mode"] = "greedy"
        last_metrics["n_points_start"] = int(n0)
        return last_metrics

    # ----- Contiguous ranges (x-sorted) -----
    if search_mode == "contiguous":
        data_sorted = data.sort_values("x", ascending=True).reset_index(drop=True)
        xs = data_sorted["x"].to_numpy(float)
        ys = data_sorted["y"].to_numpy(float)
        n = len(xs)

        best_passing = None
        best_failing = None

        # Prefer larger subsets: search by window size from n down to min_points
        for size in range(n, min_points - 1, -1):
            found_any_passing = False
            for start in range(0, n - size + 1):
                end = start + size
                metrics = compute_curve_metrics(
                    xs[start:end],
                    ys[start:end],
                    min_points=min_points,
                    max_abs_bias_thresh=max_abs_bias_thresh,
                    min_r2_thresh=min_r2_thresh,
                    weighting=weighting,
                )
                if metrics["passes_thresholds"]:
                    found_any_passing = True
                    if best_passing is None or _score_metrics_for_selection(metrics) < _score_metrics_for_selection(best_passing):
                        best_passing = metrics
                else:
                    if best_failing is None or _score_metrics_for_selection(metrics) < _score_metrics_for_selection(best_failing):
                        best_failing = metrics
            if found_any_passing and best_passing is not None:
                best_passing["search_mode"] = "contiguous"
                best_passing["n_points_start"] = int(n0)
                return best_passing

        # No passing subset; return "best" failing.
        out = best_failing if best_failing is not None else compute_curve_metrics(
            xs,
            ys,
            min_points=min_points,
            max_abs_bias_thresh=max_abs_bias_thresh,
            min_r2_thresh=min_r2_thresh,
            weighting=weighting,
        )
        out["search_mode"] = "contiguous"
        out["n_points_start"] = int(n0)
        return out

    # ----- Exhaustive (true optimum) -----
    if search_mode == "exhaustive":
        data_clean = data.reset_index(drop=True)
        xs = data_clean["x"].to_numpy(float)
        ys = data_clean["y"].to_numpy(float)
        n = len(xs)

        if n > int(exhaustive_max_n):
            # Safety fallback
            return fit_curve_best_subset(
                xs,
                ys,
                min_points=min_points,
                max_abs_bias_thresh=max_abs_bias_thresh,
                min_r2_thresh=min_r2_thresh,
                weighting=weighting,
                search_mode="contiguous",
                exhaustive_max_n=exhaustive_max_n,
                max_iterations=max_iterations,
            )

        from itertools import combinations

        best_passing = None
        best_failing = None

        # Early-stop by subset size: first size with any pass is optimal in n_points.
        idx = np.arange(n)
        for size in range(n, min_points - 1, -1):
            found_any_passing = False
            for comb in combinations(idx, size):
                sel = np.fromiter(comb, dtype=int)
                metrics = compute_curve_metrics(
                    xs[sel],
                    ys[sel],
                    min_points=min_points,
                    max_abs_bias_thresh=max_abs_bias_thresh,
                    min_r2_thresh=min_r2_thresh,
                    weighting=weighting,
                )
                if metrics["passes_thresholds"]:
                    found_any_passing = True
                    if best_passing is None or _score_metrics_for_selection(metrics) < _score_metrics_for_selection(best_passing):
                        best_passing = metrics
                else:
                    if best_failing is None or _score_metrics_for_selection(metrics) < _score_metrics_for_selection(best_failing):
                        best_failing = metrics
            if found_any_passing and best_passing is not None:
                best_passing["search_mode"] = "exhaustive"
                best_passing["n_points_start"] = int(n0)
                return best_passing

        out = best_failing if best_failing is not None else compute_curve_metrics(
            xs,
            ys,
            min_points=min_points,
            max_abs_bias_thresh=max_abs_bias_thresh,
            min_r2_thresh=min_r2_thresh,
            weighting=weighting,
        )
        out["search_mode"] = "exhaustive"
        out["n_points_start"] = int(n0)
        return out

    raise ValueError(f"Unknown search_mode: {search_mode}")


def fit_curve_iterative(
    x,
    y,
    min_points=6,
    max_abs_bias_thresh=20.0,
    min_r2_thresh=0.995,
    weighting="1/x",
    max_iterations=100,
):
    """
    Iterative weighted linear calibration for one analyte/IS combo.

    Model: Ratio = m * Conc + b
      x = concentration
      y = analyte_area / IS_area

    Steps:
      - Start with all points
      - Fit y = m x + b with weighting
      - Back-calc x from y, compute % bias
      - If any |bias| > max_abs_bias_thresh or R^2 < min_r2_thresh,
        drop the worst-bias point and refit
      - Stop when criteria are met or when remaining points < min_points
    """
    # Backward-compatible wrapper: use greedy selection.
    metrics = fit_curve_best_subset(
        x=x,
        y=y,
        min_points=min_points,
        max_abs_bias_thresh=max_abs_bias_thresh,
        min_r2_thresh=min_r2_thresh,
        weighting=weighting,
        search_mode="greedy",
        max_iterations=max_iterations,
    )
    # Historically this function returned only the metric fields; keep those.
    metrics.pop("search_mode", None)
    metrics.pop("n_points_start", None)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build analyte/IS calibration curves from long-format LC-MRM data "
            "(like test_long.csv). Uses a two-stage approach:\n"
            "  Stage 1: Select optimal calibrator levels per analyte using Area vs Conc (no IS normalization).\n"
            "  Stage 2: Fit IS-normalized curves (Ratio = analyte_area/IS_area vs Conc) using only the "
            "calibrator levels selected in Stage 1."
        )
    )
    parser.add_argument("input_csv", help="Long-format CSV (calibrators + QCs, like test_long.csv)")

    # Column names (you can override if needed)
    parser.add_argument("--sample_type_col", default="Sample Type",
                        help="Column name for sample type.")
    parser.add_argument("--component_type_col", default="Component Type",
                        help="Column name for component type.")
    parser.add_argument("--component_name_col", default="Component Name",
                        help="Column name for component name (transition).")
    parser.add_argument("--component_group_col", default="Component Group Name",
                        help="Column name for analyte group.")
    parser.add_argument("--actual_conc_col", default="Actual Concentration",
                        help="Column name for concentration.")
    parser.add_argument("--area_col", default="Area",
                        help="Column name for peak area.")
    parser.add_argument("--sample_index_col", default="Sample Index",
                        help="Column name for sample index.")
    parser.add_argument("--sample_name_col", default="Sample Name",
                        help="Column name for sample name.")

    # Labels for Calibrators vs QCs and analyte vs IS
    parser.add_argument("--cal_sample_type", default="Standard",
                        help="Sample Type value used for calibrators (default: 'Standard').")
    parser.add_argument("--analyte_type_label", default="Analyte",
                        help="Component Type value for analytes (default: 'Analyte').")
    parser.add_argument("--is_type_label", default="Internal Standards",
                        help="Component Type value for IS (default: 'Internal Standards').")

    # Fitting options
    parser.add_argument("--min_points", type=int, default=5,
                        help="Minimum number of calibrator points required (after trimming).")
    parser.add_argument("--max_abs_bias", type=float, default=20.0,
                        help="Maximum allowed absolute %% bias for any calibrator.")
    parser.add_argument("--min_r2", type=float, default=0.99,
                        help="Minimum acceptable R^2.")
    parser.add_argument("--weighting", choices=["1/x", "1/x2", "none"],
                        default="1/x", help="Weighting scheme.")
    parser.add_argument(
        "--search_mode",
        choices=["greedy", "contiguous", "exhaustive"],
        default="greedy",
        help=(
            "How to select the calibrator subset. "
            "'greedy' iteratively drops the worst-bias point (fast). "
            "'contiguous' tries all contiguous ranges in concentration order (often realistic). "
            "'exhaustive' tries all subsets (true optimum; can be slow)."
        ),
    )
    parser.add_argument(
        "--exhaustive_max_n",
        type=int,
        default=18,
        help="Safety cutoff for exhaustive search: if n points > this, automatically fall back to contiguous.",
    )

    parser.add_argument("--output", default="analyte_multi_IS_fits_iterative.csv",
                        help="Output CSV filename.")

    args = parser.parse_args()
    args.output = safe_output_path(args.output)

    df = pd.read_csv(args.input_csv)

    # --- Filter to calibrator samples only ---
    cal = df[df[args.sample_type_col] == args.cal_sample_type].copy()

    analytes = cal[cal[args.component_type_col] == args.analyte_type_label].copy()
    iss = cal[cal[args.component_type_col] == args.is_type_label].copy()

    # Wide tables: rows = (Sample Index, Sample Name), cols = Component Name
    idx_cols = [args.sample_index_col, args.sample_name_col]

    area_analyte = analytes.pivot_table(
        index=idx_cols,
        columns=args.component_name_col,
        values=args.area_col,
        aggfunc="first",
    )

    conc_analyte = analytes.pivot_table(
        index=idx_cols,
        columns=args.component_name_col,
        values=args.actual_conc_col,
        aggfunc="first",
    )

    area_is = iss.pivot_table(
        index=idx_cols,
        columns=args.component_name_col,
        values=args.area_col,
        aggfunc="first",
    )

    # Map analyte component -> analyte group
    analyte_group_map = (
        analytes[[args.component_name_col, args.component_group_col]]
        .drop_duplicates()
        .set_index(args.component_name_col)[args.component_group_col]
        .to_dict()
    )

    print(f"Calibrator samples detected: {len(area_analyte)}")
    print(f"Analyte components: {len(area_analyte.columns)}")
    print(f"Internal standards: {len(area_is.columns)}")
    print("\nStage 1: Selecting calibrator levels based on analyte area (no IS normalization)...")

    # ===== STAGE 1: Select calibrator levels per analyte using area-only fits =====
    analyte_selected_levels = {}
    
    for analyte_name in area_analyte.columns:
        if analyte_name not in conc_analyte.columns:
            continue
        
        x = pd.to_numeric(conc_analyte[analyte_name], errors="coerce").to_numpy()
        analyte_area = pd.to_numeric(area_analyte[analyte_name], errors="coerce").to_numpy()
        
        # Fit Area vs Conc (no IS normalization)
        # Clean data
        sub_area = (
            pd.DataFrame({"Conc": x, "Area": analyte_area})
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        
        if len(sub_area) >= args.min_points:
            # Select optimal calibrator levels based on area-only fit
            selected_conc = select_calibrator_levels(
                x=sub_area["Conc"].values,
                y=sub_area["Area"].values,
                min_points=args.min_points,
                max_abs_bias_thresh=args.max_abs_bias,
                min_r2_thresh=args.min_r2,
                weighting=args.weighting,
                search_mode=args.search_mode,
                exhaustive_max_n=args.exhaustive_max_n,
            )
            analyte_selected_levels[analyte_name] = selected_conc
            print(f"  {analyte_name}: Selected {len(selected_conc)} calibrator level(s) from {len(sub_area)} total")
        else:
            analyte_selected_levels[analyte_name] = set()
            print(f"  {analyte_name}: Insufficient points ({len(sub_area)}) for stage 1 selection")
    
    print("\nStage 2: Fitting IS-normalized curves using selected calibrator levels...")
    
    results = []

    # ===== STAGE 2: Fit IS-normalized curves using only selected calibrator levels =====
    for analyte_name in area_analyte.columns:
        if analyte_name not in conc_analyte.columns:
            continue

        x = pd.to_numeric(conc_analyte[analyte_name], errors="coerce").to_numpy()
        analyte_area = pd.to_numeric(area_analyte[analyte_name], errors="coerce").to_numpy()
        analyte_group = analyte_group_map.get(analyte_name, "")
        
        # Get selected levels for this analyte
        selected_levels = analyte_selected_levels.get(analyte_name, set())

        # Loop over internal standards
        for is_name in area_is.columns:
            is_area = pd.to_numeric(area_is[is_name], errors="coerce").to_numpy()

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = analyte_area / is_area

            sub = (
                pd.DataFrame({"Conc": x, "Ratio": ratio})
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if len(sub) < args.min_points:
                metrics = {
                    "n_points": len(sub),
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                    "rmse_ratio": np.nan,
                    "mean_abs_bias_pct": np.nan,
                    "max_abs_bias_pct": np.nan,
                    "lloq_conc": np.nan,
                    "uloq_conc": np.nan,
                    "passes_thresholds": False,
                    "n_points_stage1": len(selected_levels),
                }
            else:
                # Use selected levels from stage 1 (if any), otherwise use all points
                allowed_conc = list(selected_levels) if len(selected_levels) > 0 else None
                
                metrics = fit_curve_best_subset(
                    x=sub["Conc"].values,
                    y=sub["Ratio"].values,
                    min_points=args.min_points,
                    max_abs_bias_thresh=args.max_abs_bias,
                    min_r2_thresh=args.min_r2,
                    weighting=args.weighting,
                    search_mode="greedy",  # Stage 2 uses greedy on pre-selected levels
                    exhaustive_max_n=args.exhaustive_max_n,
                    allowed_concentrations=allowed_conc,
                )
                metrics["n_points_stage1"] = len(selected_levels) if selected_levels else 0

            row = {
                "analyte_component": analyte_name,
                "analyte_group": analyte_group,
                "internal_standard": is_name,
            }
            row.update(metrics)
            results.append(row)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved curves to {args.output}\n")


if __name__ == "__main__":
    main()
