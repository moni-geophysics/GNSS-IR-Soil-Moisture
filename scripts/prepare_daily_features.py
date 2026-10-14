"""
Daily Feature Dataset Preparation for Model A (Random Forest)
Target Station: P041 (Marshall, Colorado), Year 2016.

This script:
1. Loads the annual GNSS-IR arc feature table and daily COSMOS validation target table.
2. Filters eligible arcs using documented QC flags (qc_flagged == False).
3. Generates two parallel daily aggregated datasets:
   - Mean dataset: arithmetic mean for height and amplitude; circular mean for phase.
   - Median dataset: median for height and amplitude; circular median for phase.
4. Merges by calendar date with valid COSMOS soil moisture targets (315 dates).
5. Applies the exact chronological 80/20 train/test partition:
   - Train: 252 days (2016-01-04 to 2016-10-06)
   - Test: 63 days (2016-10-07 to 2016-12-31)
6. Exports daily feature datasets to data/processed/p041/ (with fallback support for scratch/).
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "p041"
SCRATCH_DIR = PROJECT_ROOT / "scratch"

ANNUAL_FEATURES_CSV = PROCESSED_DIR / "p041_2016_annual_gnss_ir_features.csv"
COSMOS_VALIDATION_CSV = PROCESSED_DIR / "p041_2016_daily_gnss_cosmos_validation.csv"

# Primary output paths under data/processed/p041/
OUT_MEAN_CSV = PROCESSED_DIR / "p041_2016_daily_mean_dataset.csv"
OUT_MEDIAN_CSV = PROCESSED_DIR / "p041_2016_daily_median_dataset.csv"
OUT_COVERAGE_CSV = PROCESSED_DIR / "p041_2016_model_a_aggregation_coverage.csv"


def compute_circular_mean(angles: np.ndarray) -> tuple[float, float]:
    """
    Compute circular mean and resultant vector length R for wrapped angles in [-pi, pi].
    """
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    if len(angles) == 0:
        return np.nan, np.nan

    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    mean_angle = float(np.arctan2(s, c))
    r_length = float(np.sqrt(s**2 + c**2))
    return mean_angle, r_length


def compute_circular_median(angles: np.ndarray, grid_points: int = 720) -> float:
    """
    Compute exact circular median minimizing the mean circular arc distance on [-pi, pi].

    Definition (Mardia & Jupp 2000, Fisher 1993):
    theta* = argmin_{phi in [-pi, pi]} sum_{i=1}^N |Arg(exp(j*(angles_i - phi)))|
    """
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    if len(angles) == 0:
        return np.nan
    if len(angles) == 1:
        return float(angles[0])

    candidates = np.linspace(-np.pi, np.pi, grid_points, endpoint=False)
    diffs = np.abs(np.angle(np.exp(1j * (angles[:, None] - candidates[None, :]))))
    total_distance = np.sum(diffs, axis=0)
    best_idx = int(np.argmin(total_distance))
    return float(candidates[best_idx])


def prepare_datasets(output_dir: Path = None):
    print("=" * 80)
    print("PREPARING DAILY FEATURE DATASETS FOR RANDOM FOREST (MODEL A)")
    print("=" * 80)

    if output_dir is None:
        output_dir = PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    out_mean = output_dir / "p041_2016_daily_mean_dataset.csv"
    out_median = output_dir / "p041_2016_daily_median_dataset.csv"
    out_coverage = output_dir / "p041_2016_model_a_aggregation_coverage.csv"

    # 1. Load Annual Feature Table
    print(f"\n[STEP 1: LOADING ANNUAL FEATURE TABLE]")
    print(f"Loading: {ANNUAL_FEATURES_CSV}")
    if not ANNUAL_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Annual feature table not found at {ANNUAL_FEATURES_CSV}. "
            "Run full-year arc processing first."
        )

    df_annual = pd.read_csv(ANNUAL_FEATURES_CSV)
    n_rows_annual = len(df_annual)
    print(f"Total arc rows: {n_rows_annual}")

    clean_mask = df_annual["qc_flagged"] == False
    n_clean_arcs = int(clean_mask.sum())
    print(f"Retained Clean Arcs: {n_clean_arcs} ({n_clean_arcs / n_rows_annual * 100.0:.2f}%)")

    df_clean = df_annual[clean_mask].copy()

    # 2. Load COSMOS Daily Target Table
    print(f"\n[STEP 2: LOADING COSMOS DAILY VALIDATION TARGETS]")
    print(f"Loading: {COSMOS_VALIDATION_CSV}")
    if not COSMOS_VALIDATION_CSV.exists():
        raise FileNotFoundError(
            f"COSMOS validation table not found at {COSMOS_VALIDATION_CSV}. "
            "Run daily validation dataset preparation first."
        )

    df_val = pd.read_csv(COSMOS_VALIDATION_CSV)
    valid_cosmos_mask = df_val["cosmos_sm_median"].notnull()
    n_valid_cosmos = int(valid_cosmos_mask.sum())
    print(f"Days with valid COSMOS target: {n_valid_cosmos}")

    # 3. Perform Daily Aggregations
    print(f"\n[STEP 3: COMPUTING DAILY MEAN AND MEDIAN AGGREGATIONS]")
    daily_results = []
    unique_doys = sorted(df_annual["doy"].unique())

    for doy in unique_doys:
        sub = df_clean[df_clean["doy"] == doy]
        n_clean = len(sub)
        n_raw = len(df_annual[df_annual["doy"] == doy])

        if n_clean > 0:
            h_vals = sub["ls_reflector_height_m"].to_numpy()
            amp_vals = sub["ls_amplitude"].to_numpy()
            phase_vals = sub["ls_phase"].to_numpy()

            rh_mean = float(np.mean(h_vals))
            amp_mean = float(np.mean(amp_vals))
            circ_mean, r_len = compute_circular_mean(phase_vals)

            rh_median = float(np.median(h_vals))
            amp_median = float(np.median(amp_vals))
            circ_median = compute_circular_median(phase_vals)

            circ_diff = float(np.abs(np.angle(np.exp(1j * (circ_mean - circ_median)))))

            daily_results.append({
                "doy": doy,
                "raw_arc_count": n_raw,
                "clean_arc_count": n_clean,
                "qc_flagged_count": n_raw - n_clean,
                "qc_pass_rate": float(n_clean / n_raw) if n_raw > 0 else 0.0,
                # Mean features
                "rh_mean": rh_mean,
                "amp_mean": amp_mean,
                "phase_circ_mean": circ_mean,
                "phase_cos_mean": float(np.cos(circ_mean)),
                "phase_sin_mean": float(np.sin(circ_mean)),
                # Median features
                "rh_median": rh_median,
                "amp_median": amp_median,
                "phase_circ_median": circ_median,
                "phase_cos_median": float(np.cos(circ_median)),
                "phase_sin_median": float(np.sin(circ_median)),
                # Diagnostics
                "phase_resultant_length": r_len,
                "phase_circ_diff_mean_vs_median": circ_diff,
                "rh_diff_mean_minus_median": rh_mean - rh_median,
                "amp_diff_mean_minus_median": amp_mean - amp_median,
            })
        else:
            daily_results.append({
                "doy": doy,
                "raw_arc_count": n_raw,
                "clean_arc_count": 0,
                "qc_flagged_count": n_raw,
                "qc_pass_rate": 0.0,
                "rh_mean": np.nan,
                "amp_mean": np.nan,
                "phase_circ_mean": np.nan,
                "phase_cos_mean": np.nan,
                "phase_sin_mean": np.nan,
                "rh_median": np.nan,
                "amp_median": np.nan,
                "phase_circ_median": np.nan,
                "phase_cos_median": np.nan,
                "phase_sin_median": np.nan,
                "phase_resultant_length": np.nan,
                "phase_circ_diff_mean_vs_median": np.nan,
                "rh_diff_mean_minus_median": np.nan,
                "amp_diff_mean_minus_median": np.nan,
            })

    df_agg = pd.DataFrame(daily_results)

    # 4. Merge with Calendar Dates and COSMOS Targets
    df_val_sub = df_val[[
        "date", "year", "doy", "cosmos_sm_median", "cosmos_sm_mean",
        "cosmos_sm_std", "cosmos_sm_good_count", "cosmos_sm_total_count",
    ]].copy()

    df_coverage = df_val_sub.merge(df_agg, on="doy", how="left")
    df_coverage["gnss_data_status"] = np.where(
        df_coverage["clean_arc_count"].isnull() | (df_coverage["clean_arc_count"] == 0),
        "missing", "available"
    )
    df_coverage["cosmos_status"] = np.where(df_coverage["cosmos_sm_median"].isnull(), "missing", "valid")
    df_coverage["usable_for_model_a"] = (df_coverage["gnss_data_status"] == "available") & (df_coverage["cosmos_status"] == "valid")

    # Chronological Split on Valid COSMOS Dates (315 dates)
    valid_mask = df_coverage["cosmos_status"] == "valid"
    valid_dates_df = df_coverage[valid_mask].sort_values("date").reset_index(drop=True)
    n_valid_total = len(valid_dates_df)  # 315
    n_train = int(n_valid_total * 0.8)   # 252
    train_dates = set(valid_dates_df.iloc[:n_train]["date"])
    test_dates = set(valid_dates_df.iloc[n_train:]["date"])

    def assign_split(row):
        if row["date"] in train_dates:
            return "train"
        elif row["date"] in test_dates:
            return "test"
        else:
            return "unlabeled"

    df_coverage["split"] = df_coverage.apply(assign_split, axis=1)

    print("\nChronological Holdout Partition Summary:")
    print(f"  Total valid target dates: {n_valid_total}")
    print(f"  Training partition:       {len(train_dates)} dates ({valid_dates_df.iloc[0]['date']} to {valid_dates_df.iloc[n_train-1]['date']})")
    print(f"  Testing partition:        {len(test_dates)} dates ({valid_dates_df.iloc[n_train]['date']} to {valid_dates_df.iloc[-1]['date']})")

    labeled_df = df_coverage[df_coverage["usable_for_model_a"]].sort_values("date").reset_index(drop=True)

    # Dataset A: Mean Dataset
    mean_cols = [
        "date", "year", "doy", "split", "cosmos_sm_median", "cosmos_sm_mean",
        "rh_mean", "amp_mean", "phase_circ_mean", "phase_cos_mean", "phase_sin_mean",
        "phase_resultant_length", "clean_arc_count",
    ]
    df_mean_dataset = labeled_df[mean_cols].rename(columns={
        "cosmos_sm_median": "target_cosmos_sm_median",
        "cosmos_sm_mean": "target_cosmos_sm_mean",
        "phase_cos_mean": "phase_cos",
        "phase_sin_mean": "phase_sin",
    })

    # Dataset B: Median Dataset
    median_cols = [
        "date", "year", "doy", "split", "cosmos_sm_median", "cosmos_sm_mean",
        "rh_median", "amp_median", "phase_circ_median", "phase_cos_median", "phase_sin_median",
        "phase_resultant_length", "clean_arc_count",
    ]
    df_median_dataset = labeled_df[median_cols].rename(columns={
        "cosmos_sm_median": "target_cosmos_sm_median",
        "cosmos_sm_mean": "target_cosmos_sm_mean",
        "phase_cos_median": "phase_cos",
        "phase_sin_median": "phase_sin",
    })

    # Export CSVs
    df_mean_dataset.to_csv(out_mean, index=False)
    df_median_dataset.to_csv(out_median, index=False)
    df_coverage.to_csv(out_coverage, index=False)

    print("\n[SUCCESS] Exported Feature Datasets:")
    print(f"  1. Mean dataset:     {out_mean} ({len(df_mean_dataset)} rows)")
    print(f"  2. Median dataset:   {out_median} ({len(df_median_dataset)} rows)")
    print(f"  3. Coverage dataset: {out_coverage} ({len(df_coverage)} rows)")


if __name__ == "__main__":
    prepare_datasets()
