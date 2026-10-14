"""
Independent Metric Recomputation and Alignment for Final Benchmark Audit
Station: P041 (Marshall, Colorado), Year 2016
Held-Out Test Period: 63 dates (2016-10-07 to 2016-12-31)

Recomputes:
- RMSE: sqrt(mean((y_pred - y_true)^2))
- MAE: mean(|y_pred - y_true|)
- Bias: mean(y_pred - y_true)
- ubRMSE: sqrt(mean((y_pred - y_true - Bias)^2))
- Pearson r: corr(y_true, y_pred)
- Sequential Day-to-Day Delta r (N=62 sequential pairs)
- Strict Calendar-Consecutive Delta r (dt=1 day, N=57 pairs)
- Backward 30-Day Anomaly Correlation (r_anom, strictly leakage-safe)
- Diagnostic sensitivity without Dec 17 (N=62)

Outputs:
- data/processed/p041/p041_2016_final_benchmark_metrics.csv
- data/processed/p041/p041_2016_final_benchmark_predictions_aligned.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "p041"
SCRATCH_DIR = PROJECT_ROOT / "scratch"


def resolve_input_path(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


PREDICTIONS_CSV = resolve_input_path(
    PROCESSED_DIR / "p041_2016_rf_final_test_predictions.csv",
    SCRATCH_DIR / "model_a_final_test_predictions.csv",
)
VALIDATION_CSV = PROCESSED_DIR / "p041_2016_daily_gnss_cosmos_validation.csv"

OUT_METRICS_CSV = PROCESSED_DIR / "p041_2016_final_benchmark_metrics.csv"
OUT_ALIGNED_CSV = PROCESSED_DIR / "p041_2016_final_benchmark_predictions_aligned.csv"


def prepare_aligned_dataset() -> pd.DataFrame:
    df_pred = pd.read_csv(PREDICTIONS_CSV)
    df_pred["date_dt"] = pd.to_datetime(df_pred["date"])

    # Load validation table to compute backward rolling baseline across full 2016
    val = pd.read_csv(VALIDATION_CSV)
    val["date_dt"] = pd.to_datetime(val["date"])
    full_cal = pd.DataFrame({"date_dt": pd.date_range("2016-01-01", "2016-12-31")})
    merged = pd.merge(full_cal, val[["date_dt", "cosmos_sm_median"]], on="date_dt", how="left")

    # Strictly backward-looking: shift(1) ensures date t never sees date t
    merged["past_30d_baseline"] = (
        merged["cosmos_sm_median"].shift(1).rolling(window=30, min_periods=5).mean()
    )

    # Merge backward baseline into test predictions
    aligned = pd.merge(
        df_pred, merged[["date_dt", "past_30d_baseline"]], on="date_dt", how="left"
    )
    aligned = aligned.sort_values("date_dt").reset_index(drop=True)
    return aligned


def compute_comprehensive_metrics(
    valid: pd.DataFrame, pred_col: str, target_col: str = "actual_cosmos_sm"
) -> dict:
    n = len(valid)
    y_true = valid[target_col].values.astype(float)
    y_pred = valid[pred_col].values.astype(float)

    # 1. Total and unbiased error metrics
    error = y_pred - y_true
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    ubrmse = float(np.sqrt(np.mean((error - bias) ** 2)))

    # 2. Pearson correlation
    std_t = float(np.std(y_true))
    std_p = float(np.std(y_pred))
    if std_t > 1e-9 and std_p > 1e-9 and n > 2:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = np.nan

    # 3. Sequential difference correlation (all sequential test rows, N=n-1)
    if n > 2 and std_p > 1e-9:
        d_seq_t = np.diff(y_true)
        d_seq_p = np.diff(y_pred)
        if np.std(d_seq_t) > 1e-9 and np.std(d_seq_p) > 1e-9:
            delta_r_seq = float(np.corrcoef(d_seq_t, d_seq_p)[0, 1])
        else:
            delta_r_seq = np.nan
    else:
        delta_r_seq = np.nan

    # 4. Strict calendar-consecutive difference correlation (dt == 1 day)
    valid_dt = valid.copy()
    valid_dt["day_diff"] = valid_dt["date_dt"].diff().dt.days
    valid_dt["dy_true"] = valid_dt[target_col].diff()
    valid_dt["dy_pred"] = valid_dt[pred_col].diff()

    consec = valid_dt[valid_dt["day_diff"] == 1].dropna(subset=["dy_true", "dy_pred"])
    n_consec = len(consec)
    if n_consec > 2 and np.std(consec["dy_pred"]) > 1e-9:
        delta_r_consec = float(np.corrcoef(consec["dy_true"], consec["dy_pred"])[0, 1])
    else:
        delta_r_consec = np.nan

    # 5. Leakage-safe backward anomaly correlation
    anom_t = valid[target_col] - valid["past_30d_baseline"]
    anom_p = valid[pred_col] - valid["past_30d_baseline"]
    anom_df = pd.DataFrame({"anom_t": anom_t, "anom_p": anom_p}).dropna()
    if len(anom_df) > 2 and anom_df["anom_p"].std() > 1e-9:
        r_anom = float(np.corrcoef(anom_df["anom_t"], anom_df["anom_p"])[0, 1])
    else:
        r_anom = np.nan

    return {
        "n_dates": n,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "ubrmse": ubrmse,
        "pearson_r": r,
        "delta_r_seq": delta_r_seq,
        "n_seq_pairs": n - 1 if n > 1 else 0,
        "delta_r_consec": delta_r_consec,
        "n_consec_pairs": n_consec,
        "r_anom_30d": r_anom,
    }


def run_benchmark_audit():
    aligned_df = prepare_aligned_dataset()
    OUT_ALIGNED_CSV.parent.mkdir(parents=True, exist_ok=True)
    aligned_df.to_csv(OUT_ALIGNED_CSV, index=False)
    print(f"[DONE] Saved aligned predictions table to: {OUT_ALIGNED_CSV}")

    models = [
        (
            "RF Median (Locked Primary)",
            "rf_median_pred",
            "Machine Learning",
            "Random Forest (Median daily aggregation, 4 features)",
        ),
        (
            "RF Mean (Sensitivity)",
            "rf_mean_pred",
            "Machine Learning",
            "Random Forest (Mean daily aggregation, 4 features)",
        ),
        (
            "Precipitation Baseline",
            "precipitation_pred",
            "Empirical Baseline",
            "Multiple Linear Regression on 1d, 3d, 7d precipitation",
        ),
        (
            "Climatology Baseline",
            "climatology_pred",
            "Empirical Baseline",
            "Constant training-period mean soil moisture",
        ),
        (
            "Physical GNSS-IR Retrieval",
            None,
            "Physics-Based Retrieval",
            "Single-layer Fresnel amplitude-ratio inversion (Jing et al.)",
        ),
    ]

    records = []

    for name, col, paradigm, desc in models:
        if col is None:
            records.append({
                "model": name,
                "paradigm": paradigm,
                "evaluation_subset": "Official Test Set (All Dates)",
                "n_dates": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "bias": np.nan,
                "ubrmse": np.nan,
                "pearson_r": np.nan,
                "delta_r_seq": np.nan,
                "delta_r_consec": np.nan,
                "n_consec": 0,
                "r_anom_30d": np.nan,
                "status": "PENDING (Full-year retrieval uncalibrated/unavailable)",
                "caveats": (
                    "DOY 001 feasibility test showed phasor instability; "
                    "requires dielectric mixing model and choke-ring pattern calibration."
                ),
            })
            continue

        # Official Test Set (N=63)
        m_full = compute_comprehensive_metrics(aligned_df, col)
        records.append({
            "model": name,
            "paradigm": paradigm,
            "evaluation_subset": "Official Test Set (All Dates)",
            "n_dates": m_full["n_dates"],
            "rmse": m_full["rmse"],
            "mae": m_full["mae"],
            "bias": m_full["bias"],
            "ubrmse": m_full["ubrmse"],
            "pearson_r": m_full["pearson_r"],
            "delta_r_seq": m_full["delta_r_seq"],
            "delta_r_consec": m_full["delta_r_consec"],
            "n_consec": m_full["n_consec_pairs"],
            "r_anom_30d": m_full["r_anom_30d"],
            "status": "FROZEN BENCHMARK",
            "caveats": "Includes Dec 17 anomaly; evaluated on 63 untouched chronological dates.",
        })

        # Diagnostic Sensitivity Set (N=62, excluding Dec 17)
        aligned_no17 = aligned_df[aligned_df["doy"] != 352].copy().reset_index(drop=True)
        m_sens = compute_comprehensive_metrics(aligned_no17, col)
        records.append({
            "model": name,
            "paradigm": paradigm,
            "evaluation_subset": "Diagnostic Sensitivity (Excl. Dec 17)",
            "n_dates": m_sens["n_dates"],
            "rmse": m_sens["rmse"],
            "mae": m_sens["mae"],
            "bias": m_sens["bias"],
            "ubrmse": m_sens["ubrmse"],
            "pearson_r": m_sens["pearson_r"],
            "delta_r_seq": m_sens["delta_r_seq"],
            "delta_r_consec": m_sens["delta_r_consec"],
            "n_consec": m_sens["n_consec_pairs"],
            "r_anom_30d": m_sens["r_anom_30d"],
            "status": "DIAGNOSTIC CHECK ONLY",
            "caveats": "Diagnostic exclusion to isolate impact of Dec 17 target/observable mismatch.",
        })

    metrics_df = pd.DataFrame(records)
    OUT_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT_METRICS_CSV, index=False)
    print(f"[DONE] Saved benchmark metrics table to: {OUT_METRICS_CSV}")
    print(
        "\n"
        + metrics_df[
            [
                "model",
                "evaluation_subset",
                "rmse",
                "mae",
                "bias",
                "ubrmse",
                "pearson_r",
                "delta_r_seq",
                "r_anom_30d",
                "status",
            ]
        ].to_string()
    )


if __name__ == "__main__":
    run_benchmark_audit()
