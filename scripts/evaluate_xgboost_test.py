"""
Final Evaluation of XGBoost on the Untouched Held-Out Test Period
Station: P041 (Marshall, Colorado), Year 2016
Held-Out Test Period: 63 dates (2016-10-07 to 2016-12-31)

This script executes:
1. Fits the selected optimal XGBoost model (XGB_Grid_092: lr=0.05, max_depth=2, n_est=150, mcw=1, sub=1.0, col=1.0)
   strictly on the 252 training dates (2016-01-04 to 2016-10-06).
2. Fits the default untuned XGBoost model (XGB_Default: lr=0.3, max_depth=6, n_est=100) strictly on the 252 training dates.
3. Evaluates predictions on the 63 held-out test dates.
4. Calculates independent hydrological validation metrics:
   - Sample count (N)
   - RMSE, MAE, Mean Bias (prediction minus observation), ubRMSE (unbiased RMSE)
   - Pearson correlation (r)
   - Sequential Delta r and Calendar-consecutive Delta r
   - Causal backward 30-day trailing anomaly correlation (r_anom_30d)
   - Diagnostic sensitivity excluding Dec 17 (N=62)
5. Exports:
   - data/processed/p041/p041_2016_xgboost_predictions.csv
   - data/processed/p041/p041_2016_xgboost_final_test_metrics.csv
   (also maintains backward-compatible outputs in scratch/).
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "p041"
SCRATCH_DIR = PROJECT_ROOT / "scratch"


def resolve_input_path(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


MEDIAN_DATASET_CSV = resolve_input_path(
    PROCESSED_DIR / "p041_2016_daily_median_dataset.csv",
    SCRATCH_DIR / "model_a_daily_median_dataset.csv",
)
VALIDATION_CSV = PROCESSED_DIR / "p041_2016_daily_gnss_cosmos_validation.csv"

OUT_XGB_PRED_CSV = PROCESSED_DIR / "p041_2016_xgboost_predictions.csv"
OUT_XGB_METRICS_CSV = PROCESSED_DIR / "p041_2016_xgboost_final_test_metrics.csv"
SCRATCH_XGB_PRED_CSV = SCRATCH_DIR / "xgboost_final_test_predictions.csv"
SCRATCH_XGB_METRICS_CSV = SCRATCH_DIR / "xgboost_final_test_metrics.csv"

TARGET_COL = "target_cosmos_sm_median"
FEATURES_MEDIAN = ["rh_median", "amp_median", "phase_cos", "phase_sin"]

# Selected Tuned Configuration from 5-fold expanding temporal CV
BEST_XGB_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 2,
    "n_estimators": 150,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": 42,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "tree_method": "hist",
}

# Untuned Default Configuration
DEFAULT_XGB_PARAMS = {
    "learning_rate": 0.3,
    "max_depth": 6,
    "n_estimators": 100,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": 42,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "tree_method": "hist",
}


def compute_comprehensive_metrics(valid: pd.DataFrame, pred_col: str, target_col: str = "actual_cosmos_sm") -> dict:
    n = len(valid)
    y_true = valid[target_col].values.astype(float)
    y_pred = valid[pred_col].values.astype(float)

    error = y_pred - y_true
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    ubrmse = float(np.sqrt(np.mean((error - bias) ** 2)))

    std_t = float(np.std(y_true))
    std_p = float(np.std(y_pred))
    if std_t > 1e-9 and std_p > 1e-9 and n > 2:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = np.nan

    # Sequential difference correlation (all sequential test rows, N=n-1)
    if n > 2 and std_p > 1e-9:
        d_seq_t = np.diff(y_true)
        d_seq_p = np.diff(y_pred)
        if np.std(d_seq_t) > 1e-9 and np.std(d_seq_p) > 1e-9:
            delta_r_seq = float(np.corrcoef(d_seq_t, d_seq_p)[0, 1])
        else:
            delta_r_seq = np.nan
    else:
        delta_r_seq = np.nan

    # Strict calendar-consecutive difference correlation (dt == 1 day)
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

    # Leakage-safe backward anomaly correlation
    anom_t = valid[target_col] - valid["past_30d_baseline"]
    anom_p = valid[pred_col] - valid["past_30d_baseline"]
    if np.std(anom_t) > 1e-9 and np.std(anom_p) > 1e-9:
        r_anom_30d = float(np.corrcoef(anom_t, anom_p)[0, 1])
    else:
        r_anom_30d = np.nan

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
        "r_anom_30d": r_anom_30d,
    }


def compute_causal_trailing_baseline(val_df: pd.DataFrame, window_days: int = 30) -> pd.Series:
    full = val_df[["date", "cosmos_sm_median"]].copy().sort_values("date").reset_index(drop=True)
    full["date_dt"] = pd.to_datetime(full["date"])
    full = full.set_index("date_dt")

    cal = full.asfreq("D")
    cal["past_baseline"] = cal["cosmos_sm_median"].shift(1).rolling(window=window_days, min_periods=5).mean()

    res = full.reset_index().merge(cal[["past_baseline"]].reset_index(), on="date_dt", how="left")
    return res.set_index("date")["past_baseline"]


def main():
    print("=" * 80)
    print("RUNNING XGBOOST HELD-OUT TEST EVALUATION: P041 2016")
    print("=" * 80)

    # 1. Load Dataset
    df = pd.read_csv(MEDIAN_DATASET_CSV)
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt").reset_index(drop=True)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(f"Training Dates: {len(train_df)} ({train_df['date'].iloc[0]} to {train_df['date'].iloc[-1]})")
    print(f"Testing Dates:  {len(test_df)} ({test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]})")

    X_train = train_df[FEATURES_MEDIAN].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()

    X_test = test_df[FEATURES_MEDIAN].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    # 2. Fit Models strictly on Training Set
    print("\nFitting Tuned XGBoost (Grid_092: lr=0.05, max_depth=2, n_est=150)...")
    xgb_tuned = xgb.XGBRegressor(**BEST_XGB_PARAMS)
    xgb_tuned.fit(X_train, y_train)

    print("Fitting Default XGBoost (lr=0.30, max_depth=6, n_est=100)...")
    xgb_default = xgb.XGBRegressor(**DEFAULT_XGB_PARAMS)
    xgb_default.fit(X_train, y_train)

    # 3. Generate Predictions on Held-Out Test Set
    preds_tuned = xgb_tuned.predict(X_test)
    preds_default = xgb_default.predict(X_test)

    # 4. Trailing 30-day baseline for causal anomaly correlation
    val_full = pd.read_csv(VALIDATION_CSV)
    past_series = compute_causal_trailing_baseline(val_full, window_days=30)

    # 5. Build Dedicated Single-Model Prediction File
    test_df["actual_cosmos_sm"] = y_test
    test_df["xgb_tuned_prediction"] = preds_tuned
    test_df["xgb_default_prediction"] = preds_default
    test_df["xgb_tuned_pred"] = preds_tuned
    test_df["xgb_default_pred"] = preds_default
    test_df["prediction_error"] = preds_tuned - y_test
    test_df["past_30d_baseline"] = test_df["date"].map(past_series)

    dedicated_cols = [
        "date", "year", "doy", "actual_cosmos_sm",
        "xgb_tuned_prediction", "prediction_error",
        "xgb_default_prediction",
        "rh_median", "amp_median", "phase_cos", "phase_sin"
    ]
    dedicated_df = test_df[dedicated_cols].copy()
    OUT_XGB_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    dedicated_df.to_csv(OUT_XGB_PRED_CSV, index=False)
    print(f"Saved dedicated XGBoost predictions to: {OUT_XGB_PRED_CSV} ({len(dedicated_df)} rows)")

    # 6. Compute Metrics
    test_df["pred_tuned"] = preds_tuned
    test_df["pred_default"] = preds_default

    metrics_tuned_full = compute_comprehensive_metrics(test_df, "pred_tuned")
    metrics_default_full = compute_comprehensive_metrics(test_df, "pred_default")

    test_df_no17 = test_df[test_df["doy"] != 352].copy().reset_index(drop=True)
    metrics_tuned_no17 = compute_comprehensive_metrics(test_df_no17, "pred_tuned")
    metrics_default_no17 = compute_comprehensive_metrics(test_df_no17, "pred_default")

    xgb_metrics_records = [
        {
            "model": "XGBoost Tuned (Optimal Grid_092)",
            "paradigm": "Machine Learning (Gradient Boosting)",
            "benchmark_tier": "Primary Benchmark",
            "evaluation_subset": "Official Test Set (All Dates)",
            "n_dates": metrics_tuned_full["n_dates"],
            "rmse": metrics_tuned_full["rmse"],
            "mae": metrics_tuned_full["mae"],
            "bias": metrics_tuned_full["bias"],
            "ubrmse": metrics_tuned_full["ubrmse"],
            "pearson_r": metrics_tuned_full["pearson_r"],
            "delta_r_seq": metrics_tuned_full["delta_r_seq"],
            "delta_r_consec": metrics_tuned_full["delta_r_consec"],
            "n_consec": metrics_tuned_full["n_consec_pairs"],
            "r_anom_30d": metrics_tuned_full["r_anom_30d"],
            "description": "Tuned depth=2, lr=0.05, n_est=150 (Primary Configuration)",
        },
        {
            "model": "XGBoost Default (Untuned)",
            "paradigm": "Machine Learning (Gradient Boosting)",
            "benchmark_tier": "Sensitivity Analysis (Model Variant)",
            "evaluation_subset": "Official Test Set (All Dates)",
            "n_dates": metrics_default_full["n_dates"],
            "rmse": metrics_default_full["rmse"],
            "mae": metrics_default_full["mae"],
            "bias": metrics_default_full["bias"],
            "ubrmse": metrics_default_full["ubrmse"],
            "pearson_r": metrics_default_full["pearson_r"],
            "delta_r_seq": metrics_default_full["delta_r_seq"],
            "delta_r_consec": metrics_default_full["delta_r_consec"],
            "n_consec": metrics_default_full["n_consec_pairs"],
            "r_anom_30d": metrics_default_full["r_anom_30d"],
            "description": "Default depth=6, lr=0.30, n_est=100 (Diagnostic Baseline)",
        },
        {
            "model": "XGBoost Tuned (Optimal Grid_092)",
            "paradigm": "Machine Learning (Gradient Boosting)",
            "benchmark_tier": "Sensitivity Analysis (Excl. Dec 17)",
            "evaluation_subset": "Diagnostic Sensitivity (Excl. Dec 17)",
            "n_dates": metrics_tuned_no17["n_dates"],
            "rmse": metrics_tuned_no17["rmse"],
            "mae": metrics_tuned_no17["mae"],
            "bias": metrics_tuned_no17["bias"],
            "ubrmse": metrics_tuned_no17["ubrmse"],
            "pearson_r": metrics_tuned_no17["pearson_r"],
            "delta_r_seq": metrics_tuned_no17["delta_r_seq"],
            "delta_r_consec": metrics_tuned_no17["delta_r_consec"],
            "n_consec": metrics_tuned_no17["n_consec_pairs"],
            "r_anom_30d": metrics_tuned_no17["r_anom_30d"],
            "description": "Tuned depth=2, lr=0.05, n_est=150 (Dec 17 excluded)",
        },
        {
            "model": "XGBoost Default (Untuned)",
            "paradigm": "Machine Learning (Gradient Boosting)",
            "benchmark_tier": "Sensitivity Analysis (Excl. Dec 17)",
            "evaluation_subset": "Diagnostic Sensitivity (Excl. Dec 17)",
            "n_dates": metrics_default_no17["n_dates"],
            "rmse": metrics_default_no17["rmse"],
            "mae": metrics_default_no17["mae"],
            "bias": metrics_default_no17["bias"],
            "ubrmse": metrics_default_no17["ubrmse"],
            "pearson_r": metrics_default_no17["pearson_r"],
            "delta_r_seq": metrics_default_no17["delta_r_seq"],
            "delta_r_consec": metrics_default_no17["delta_r_consec"],
            "n_consec": metrics_default_no17["n_consec_pairs"],
            "r_anom_30d": metrics_default_no17["r_anom_30d"],
            "description": "Default depth=6, lr=0.30, n_est=100 (Dec 17 excluded)",
        },
    ]

    metrics_df = pd.DataFrame(xgb_metrics_records)
    metrics_df.to_csv(OUT_XGB_METRICS_CSV, index=False)
    print(f"Saved XGBoost test metrics to: {OUT_XGB_METRICS_CSV}")

    # Backward compatibility for scratch/
    if SCRATCH_DIR.exists():
        metrics_df.to_csv(SCRATCH_XGB_METRICS_CSV, index=False)
        dedicated_df.to_csv(SCRATCH_XGB_PRED_CSV, index=False)

    print("\n[Official Held-Out Test Set (N=63)]")
    print(f"  Tuned XGBoost RMSE:   {metrics_tuned_full['rmse']:.6f} m^3/m^3 (ubRMSE={metrics_tuned_full['ubrmse']:.6f}, Bias={metrics_tuned_full['bias']:+.6f}, r={metrics_tuned_full['pearson_r']:.4f})")
    print(f"  Default XGBoost RMSE: {metrics_default_full['rmse']:.6f} m^3/m^3 (ubRMSE={metrics_default_full['ubrmse']:.6f}, Bias={metrics_default_full['bias']:+.6f}, r={metrics_default_full['pearson_r']:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
