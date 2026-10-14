"""
Final Evaluation of Model A (Random Forest) on the Untouched Held-Out Test Period
Target Station: P041 (Marshall, Colorado), Year 2016.

This script executes:
1. Model Locks (before evaluation):
   - Primary Model: Model A-Median (RF, Median aggregation, min_samples_leaf=3, max_features=1.0, max_depth=None, n_estimators=300, random_state=42)
   - Secondary Sensitivity Model: Model A-Mean (RF, Mean aggregation, min_samples_leaf=3, max_features=1.0, max_depth=None, n_estimators=300, random_state=42)
2. Fits both models strictly on the 252 training dates (2016-01-04 to 2016-10-06).
3. Generates predictions on the 63 held-out test dates (2016-10-07 to 2016-12-31).
4. Compares with Climatology and Precipitation baselines on the identical test dates.
5. Calculates independent hydrological validation metrics:
   - Sample count (N)
   - RMSE
   - Mean Bias (prediction minus observation)
   - ubRMSE (unbiased RMSE)
   - Pearson correlation (r)
   - Day-to-day rate of change correlation: r(Delta y, Delta y_hat)
   - Leakage-safe causal trailing 31-day anomaly correlation (secondary)
6. Quantifies the test-period distribution shift and MSE error decomposition.
7. Exports:
   - data/processed/p041/p041_2016_rf_final_test_predictions.csv
   - data/processed/p041/p041_2016_rf_final_test_metrics.csv
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "p041"
SCRATCH_DIR = PROJECT_ROOT / "scratch"


def resolve_input_path(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


MEAN_DATASET_CSV = resolve_input_path(
    PROCESSED_DIR / "p041_2016_daily_mean_dataset.csv",
    SCRATCH_DIR / "model_a_daily_mean_dataset.csv",
)
MEDIAN_DATASET_CSV = resolve_input_path(
    PROCESSED_DIR / "p041_2016_daily_median_dataset.csv",
    SCRATCH_DIR / "model_a_daily_median_dataset.csv",
)
CLIM_PRED_CSV = PROCESSED_DIR / "p041_2016_climatology_predictions.csv"
PRECIP_PRED_CSV = PROCESSED_DIR / "p041_2016_precipitation_predictions.csv"

OUT_PRED_CSV = PROCESSED_DIR / "p041_2016_rf_final_test_predictions.csv"
OUT_METRICS_CSV = PROCESSED_DIR / "p041_2016_rf_final_test_metrics.csv"

TARGET_COL = "target_cosmos_sm_median"
FEATURES_MEDIAN = ["rh_median", "amp_median", "phase_cos", "phase_sin"]
FEATURES_MEAN = ["rh_mean", "amp_mean", "phase_cos", "phase_sin"]

# Locked hyperparameter specification
RF_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 3,
    "max_features": 1.0,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
}


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute standard hydrological validation metrics.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(actual)

    error = predicted - actual
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))
    ubrmse = float(np.sqrt(np.mean((error - bias) ** 2)))

    std_act = float(np.std(actual))
    std_pred = float(np.std(predicted))
    if std_act > 1e-9 and std_pred > 1e-9 and n > 2:
        r = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        r = np.nan

    # Day-to-day rate of change correlation (first differences)
    if n > 3:
        d_act = np.diff(actual)
        d_pred = np.diff(predicted)
        if np.std(d_act) > 1e-9 and np.std(d_pred) > 1e-9:
            delta_r = float(np.corrcoef(d_act, d_pred)[0, 1])
        else:
            delta_r = np.nan
    else:
        delta_r = np.nan

    return {
        "n_samples": n,
        "rmse": rmse,
        "bias": bias,
        "ubrmse": ubrmse,
        "pearson_r": r,
        "delta_r": delta_r,
    }


def compute_causal_trailing_anomaly_r(
    full_actual: pd.Series,
    full_predicted: pd.Series,
    test_idx: pd.Index,
    window_size: int = 31,
) -> float:
    """
    Leakage-safe anomaly correlation:
    Computes a causal, backward-looking rolling mean (center=False) across the continuous historical timeline,
    subtracts it to derive daily anomalies, and evaluates Pearson r over the held-out test dates.
    Zero future observations are ever accessed.
    """
    roll_act = full_actual.rolling(window=window_size, min_periods=7, center=False).mean()
    roll_pred = full_predicted.rolling(window=window_size, min_periods=7, center=False).mean()
    anom_act = (full_actual - roll_act).loc[test_idx].dropna()
    anom_pred = (full_predicted - roll_pred).loc[test_idx].dropna()
    valid_idx = anom_act.index.intersection(anom_pred.index)
    if len(valid_idx) > 10 and anom_act.loc[valid_idx].std() > 1e-9 and anom_pred.loc[valid_idx].std() > 1e-9:
        return float(np.corrcoef(anom_act.loc[valid_idx], anom_pred.loc[valid_idx])[0, 1])
    return np.nan


def main():
    print("=" * 80)
    print("FINAL EVALUATION: LOCKED MODEL A RANDOM FOREST VS. BASELINES")
    print("Station: P041 | Year: 2016 | Held-out Period: 2016-10-07 to 2016-12-31")
    print("=" * 80)

    # 1. Load Data
    df_mean = pd.read_csv(MEAN_DATASET_CSV)
    df_med = pd.read_csv(MEDIAN_DATASET_CSV)
    df_clim = pd.read_csv(CLIM_PRED_CSV)
    df_precip = pd.read_csv(PRECIP_PRED_CSV)

    train_mean = df_mean[df_mean["split"] == "train"].copy().reset_index(drop=True)
    test_mean = df_mean[df_mean["split"] == "test"].copy().reset_index(drop=True)

    train_med = df_med[df_med["split"] == "train"].copy().reset_index(drop=True)
    test_med = df_med[df_med["split"] == "test"].copy().reset_index(drop=True)

    # Verification of split
    assert len(train_mean) == 252 and len(test_mean) == 63
    assert len(train_med) == 252 and len(test_med) == 63
    assert (test_mean["date"] == test_med["date"]).all()
    assert (test_mean["date"] == df_clim["date"]).all()
    assert (test_mean["date"] == df_precip["date"]).all()

    print(f"\n[PARTITION VERIFICATION]")
    print(f"  Training Dates: {len(train_mean)} dates ({train_mean['date'].iloc[0]} to {train_mean['date'].iloc[-1]})")
    print(f"  Testing Dates:  {len(test_mean)} dates ({test_mean['date'].iloc[0]} to {test_mean['date'].iloc[-1]})")

    y_train = train_mean[TARGET_COL].to_numpy()
    y_test = test_mean[TARGET_COL].to_numpy()

    # 2. Fit Primary Model: Model A-Median
    print("\n[FITTING PRIMARY MODEL: MODEL A-MEDIAN]")
    print(f"  Features: {FEATURES_MEDIAN}")
    print(f"  Hyperparameters: {RF_PARAMS}")
    X_train_med = train_med[FEATURES_MEDIAN].to_numpy()
    X_test_med = test_med[FEATURES_MEDIAN].to_numpy()

    rf_median = RandomForestRegressor(**RF_PARAMS)
    rf_median.fit(X_train_med, y_train)

    train_pred_med = rf_median.predict(X_train_med)
    test_pred_med = rf_median.predict(X_test_med)

    train_metrics_med = compute_metrics(y_train, train_pred_med)
    test_metrics_med = compute_metrics(y_test, test_pred_med)

    # 3. Fit Secondary Sensitivity Model: Model A-Mean
    print("\n[FITTING SECONDARY MODEL: MODEL A-MEAN]")
    print(f"  Features: {FEATURES_MEAN}")
    print(f"  Hyperparameters: {RF_PARAMS}")
    X_train_mean = train_mean[FEATURES_MEAN].to_numpy()
    X_test_mean = test_mean[FEATURES_MEAN].to_numpy()

    rf_mean = RandomForestRegressor(**RF_PARAMS)
    rf_mean.fit(X_train_mean, y_train)

    train_pred_mean = rf_mean.predict(X_train_mean)
    test_pred_mean = rf_mean.predict(X_test_mean)

    train_metrics_mean = compute_metrics(y_train, train_pred_mean)
    test_metrics_mean = compute_metrics(y_test, test_pred_mean)

    # 4. Load Baseline Predictions
    print("\n[ALIGNING BASELINES]")
    clim_pred = df_clim["climatology_prediction"].to_numpy()
    precip_pred = df_precip["precip_prediction"].to_numpy()

    test_metrics_clim = compute_metrics(y_test, clim_pred)
    test_metrics_precip = compute_metrics(y_test, precip_pred)

    # 5. Causal Trailing Anomaly Correlation
    full_y = pd.Series(df_mean[TARGET_COL].to_numpy(), index=df_mean.index)
    test_idx = df_mean[df_mean["split"] == "test"].index

    # Full predictions series for anomaly calculation
    full_pred_med = pd.Series(index=df_mean.index, dtype=float)
    full_pred_med.loc[df_mean["split"] == "train"] = train_pred_med
    full_pred_med.loc[df_mean["split"] == "test"] = test_pred_med

    full_pred_mean = pd.Series(index=df_mean.index, dtype=float)
    full_pred_mean.loc[df_mean["split"] == "train"] = train_pred_mean
    full_pred_mean.loc[df_mean["split"] == "test"] = test_pred_mean

    full_pred_precip = pd.Series(index=df_mean.index, dtype=float)
    full_pred_precip.loc[df_mean["split"] == "test"] = precip_pred

    full_pred_clim = pd.Series(index=df_mean.index, dtype=float)
    full_pred_clim.loc[:] = float(np.mean(y_train))

    anom_r_med = compute_causal_trailing_anomaly_r(full_y, full_pred_med, test_idx)
    anom_r_mean = compute_causal_trailing_anomaly_r(full_y, full_pred_mean, test_idx)
    anom_r_precip = compute_causal_trailing_anomaly_r(full_y, full_pred_precip, test_idx)
    anom_r_clim = compute_causal_trailing_anomaly_r(full_y, full_pred_clim, test_idx)

    # 6. Build and Export Test Predictions Table
    test_pred_df = pd.DataFrame({
        "date": test_mean["date"],
        "year": test_mean["year"],
        "doy": test_mean["doy"],
        "actual_cosmos_sm": y_test,
        "climatology_pred": clim_pred,
        "precipitation_pred": precip_pred,
        "rf_median_pred": test_pred_med,
        "rf_mean_pred": test_pred_mean,
        "rf_median_error": test_pred_med - y_test,
        "rf_mean_error": test_pred_mean - y_test,
        "precip_error": precip_pred - y_test,
        "climatology_error": clim_pred - y_test,
    })
    OUT_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    test_pred_df.to_csv(OUT_PRED_CSV, index=False)
    print(f"\nSaved final test predictions to: {OUT_PRED_CSV} ({len(test_pred_df)} rows)")

    # 7. Build and Export Metrics Table
    metrics_records = [
        {
            "model_role": "Baseline 1",
            "model_name": "Climatology Baseline",
            "aggregation": "constant",
            "period": "test",
            "n_samples": test_metrics_clim["n_samples"],
            "rmse": test_metrics_clim["rmse"],
            "bias": test_metrics_clim["bias"],
            "ubrmse": test_metrics_clim["ubrmse"],
            "pearson_r": test_metrics_clim["pearson_r"],
            "delta_r": test_metrics_clim["delta_r"],
            "causal_trailing_anomaly_r": anom_r_clim,
        },
        {
            "model_role": "Baseline 2",
            "model_name": "Precipitation Baseline (OLS)",
            "aggregation": "daily_accumulated",
            "period": "test",
            "n_samples": test_metrics_precip["n_samples"],
            "rmse": test_metrics_precip["rmse"],
            "bias": test_metrics_precip["bias"],
            "ubrmse": test_metrics_precip["ubrmse"],
            "pearson_r": test_metrics_precip["pearson_r"],
            "delta_r": test_metrics_precip["delta_r"],
            "causal_trailing_anomaly_r": anom_r_precip,
        },
        {
            "model_role": "Primary RF",
            "model_name": "Model A-Median (Random Forest)",
            "aggregation": "median",
            "period": "test",
            "n_samples": test_metrics_med["n_samples"],
            "rmse": test_metrics_med["rmse"],
            "bias": test_metrics_med["bias"],
            "ubrmse": test_metrics_med["ubrmse"],
            "pearson_r": test_metrics_med["pearson_r"],
            "delta_r": test_metrics_med["delta_r"],
            "causal_trailing_anomaly_r": anom_r_med,
        },
        {
            "model_role": "Primary RF (Diagnostic Train)",
            "model_name": "Model A-Median (Random Forest)",
            "aggregation": "median",
            "period": "train",
            "n_samples": train_metrics_med["n_samples"],
            "rmse": train_metrics_med["rmse"],
            "bias": train_metrics_med["bias"],
            "ubrmse": train_metrics_med["ubrmse"],
            "pearson_r": train_metrics_med["pearson_r"],
            "delta_r": train_metrics_med["delta_r"],
            "causal_trailing_anomaly_r": np.nan,
        },
        {
            "model_role": "Secondary Sensitivity",
            "model_name": "Model A-Mean (Random Forest)",
            "aggregation": "mean",
            "period": "test",
            "n_samples": test_metrics_mean["n_samples"],
            "rmse": test_metrics_mean["rmse"],
            "bias": test_metrics_mean["bias"],
            "ubrmse": test_metrics_mean["ubrmse"],
            "pearson_r": test_metrics_mean["pearson_r"],
            "delta_r": test_metrics_mean["delta_r"],
            "causal_trailing_anomaly_r": anom_r_mean,
        },
        {
            "model_role": "Secondary Sensitivity (Diagnostic Train)",
            "model_name": "Model A-Mean (Random Forest)",
            "aggregation": "mean",
            "period": "train",
            "n_samples": train_metrics_mean["n_samples"],
            "rmse": train_metrics_mean["rmse"],
            "bias": train_metrics_mean["bias"],
            "ubrmse": train_metrics_mean["ubrmse"],
            "pearson_r": train_metrics_mean["pearson_r"],
            "delta_r": train_metrics_mean["delta_r"],
            "causal_trailing_anomaly_r": np.nan,
        },
    ]

    df_metrics = pd.DataFrame(metrics_records)
    OUT_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(OUT_METRICS_CSV, index=False)
    print(f"Saved final test metrics to: {OUT_METRICS_CSV}")

    # 8. Distribution Shift & MSE Decomposition
    print("\n" + "=" * 80)
    print("DISTRIBUTION SHIFT & ERROR DECOMPOSITION ANALYSIS")
    print("=" * 80)
    tr_mean, tr_std, tr_min, tr_max = np.mean(y_train), np.std(y_train, ddof=1), np.min(y_train), np.max(y_train)
    te_mean, te_std, te_min, te_max = np.mean(y_test), np.std(y_test, ddof=1), np.min(y_test), np.max(y_test)
    print(f"Train Ground Truth (N=252): Mean={tr_mean:.6f}, Std={tr_std:.6f}, Range=[{tr_min:.4f}, {tr_max:.4f}]")
    print(f"Test  Ground Truth (N=63):  Mean={te_mean:.6f}, Std={te_std:.6f}, Range=[{te_min:.4f}, {te_max:.4f}]")
    print(f"Shift (Test Mean - Train Mean): {te_mean - tr_mean:+.6f} m^3/m^3 ({(te_mean - tr_mean) / tr_mean * 100.0:+.2f}%)")

    print("\nMSE Decomposition on Held-Out Test Set: MSE = Bias^2 + ubRMSE^2")
    for rec in metrics_records:
        if rec["period"] == "test":
            m_rmse = rec["rmse"]
            m_bias = rec["bias"]
            m_ub = rec["ubrmse"]
            mse = m_rmse ** 2
            bias_sq = m_bias ** 2
            ub_sq = m_ub ** 2
            bias_pct = (bias_sq / mse) * 100.0
            ub_pct = (ub_sq / mse) * 100.0
            r_str = f"{rec['pearson_r']:.4f}" if np.isfinite(rec["pearson_r"]) else "   N/A  "
            dr_str = f"{rec['delta_r']:+.4f}" if np.isfinite(rec["delta_r"]) else "   N/A  "
            print(f"  {rec['model_name']:<32s} | RMSE={m_rmse:.5f} | Bias={m_bias:+.5f} ({bias_pct:4.1f}% of MSE) | ubRMSE={m_ub:.5f} ({ub_pct:4.1f}% of MSE) | r={r_str} | Delta r={dr_str}")

    print("\n" + "=" * 80)
    print("FINAL HELD-OUT TEST SUMMARY TABLE (N = 63 DATES)")
    print("=" * 80)
    print(f"{'Model Name':<32s} | {'Role':<22s} | {'RMSE':<8s} | {'Bias':<8s} | {'ubRMSE':<8s} | {'Pearson r':<9s} | {'Delta r':<8s}")
    print("-" * 105)
    for rec in metrics_records:
        if rec["period"] == "test":
            r_str = f"{rec['pearson_r']:.4f}" if np.isfinite(rec["pearson_r"]) else "  N/A   "
            dr_str = f"{rec['delta_r']:+.4f}" if np.isfinite(rec["delta_r"]) else "  N/A   "
            print(f"{rec['model_name']:<32s} | {rec['model_role']:<22s} | {rec['rmse']:.6f} | {rec['bias']:+.6f} | {rec['ubrmse']:.6f} | {r_str:<9s} | {dr_str:<8s}")
    print("=" * 80)


if __name__ == "__main__":
    main()
