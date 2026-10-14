"""
Precipitation-Based Soil-Moisture Baseline for P041 (Marshall, Colorado) in 2016.

Methodological Safeguards & Implementation Details:
1. Primary Source: In-situ Vaisala WXT520 precipitation observations at MARSHALL (collocated with P041, <1 m separation).
2. Documented Cross-fill: MARSHALL2 (210 m south of P041) is used strictly as a verified backup
   for 5 missing days in late June (2016-06-23 to 2016-06-27).
3. Missing Precipitation Handling: On 2016-08-22 (DOY 235), precipitation is missing in both MARSHALL and MARSHALL2.
   It is preserved as NaN (not silently zero-filled). Training rows affected by NaN precipitation or incomplete
   antecedent rolling windows are excluded from fitting (242 training rows used out of 252).
4. Zero Test Leakage: The 63 designated test dates (2016-10-07 to 2016-12-31) remain 100% complete and identical
   to the climatology baseline. All model fitting, coefficient estimation, and feature statistics are computed
   strictly on training data.
5. Strictly Backward-Looking Rolling Windows:
   - Daily precipitation: P_daily(t) (reported daily accumulated total up to 12:00 UTC).
   - 3-day antecedent precipitation: sum of precipitation on days [t-2, t-1, t].
   - 7-day antecedent precipitation: sum of precipitation on days [t-6, ..., t].
   No future rainfall is ever accessed.
6. Evaluation: Multiple linear regression (OLS) predicting `cosmos_sm_median` on the exact 63 test dates,
   computing RMSE, Bias, and ubRMSE.
"""

import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_raw_ismn_precipitation(zip_path: Path) -> pd.DataFrame:
    """
    Extract and align daily precipitation from MARSHALL and MARSHALL2 archives in P041_MARSHALL2_ISMN_SM_P_2016_2017.zip.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date string 'YYYY-MM-DD' with columns:
        - p_marshall: Raw MARSHALL precipitation (mm)
        - p_marshall2: Raw MARSHALL2 precipitation (mm)
        - precip_mm: Final cross-filled precipitation (mm)
        - source: Data source indicator ('MARSHALL', 'MARSHALL2_crossfill', 'missing')
    """
    def parse_stm(stream):
        header = stream.readline().decode("utf-8").strip()
        rows = [line.decode("utf-8").strip().split() for line in stream if line.strip()]
        df = pd.DataFrame(rows, columns=["date", "time", "precip", "flag1", "flag2"])
        df["date"] = df["date"].str.replace("/", "-")
        df["precip"] = df["precip"].astype(float)
        # Filter strictly for 2016
        df = df[(df["date"] >= "2016-01-01") & (df["date"] <= "2016-12-31")].copy()
        return df.set_index("date")

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("PBO_H2O/MARSHALL/PBO-H2O_PBO-H2O_MARSHALL_p_-2.000000_-2.000000_Vaisala-WXT520_1_1_20160101_20171231.stm") as f1:
            df_m1 = parse_stm(f1)
        with z.open("PBO_H2O/MARSHALL2/PBO-H2O_PBO-H2O_MARSHALL2_p_-2.000000_-2.000000_Vaisala-WXT520_1_1_20160101_20171231.stm") as f2:
            df_m2 = parse_stm(f2)

    cal = pd.DataFrame(index=pd.date_range("2016-01-01", "2016-12-31").strftime("%Y-%m-%d"))
    cal.index.name = "date"
    cal["p_marshall"] = df_m1["precip"]
    cal["p_marshall2"] = df_m2["precip"]

    # Cross-fill MARSHALL with MARSHALL2 where MARSHALL is missing
    cal["precip_mm"] = cal["p_marshall"].combine_first(cal["p_marshall2"])

    cal["source"] = "MARSHALL"
    cal.loc[cal["p_marshall"].isnull() & cal["p_marshall2"].notnull(), "source"] = "MARSHALL2_crossfill"
    cal.loc[cal["precip_mm"].isnull(), "source"] = "missing"

    return cal.reset_index()


def build_precipitation_features(df_precip: pd.DataFrame) -> pd.DataFrame:
    """
    Compute strictly backward-looking daily, 3-day, and 7-day accumulated precipitation.

    Windows require complete data across the entire window (min_periods = window size).
    Any window overlapping a missing date evaluates to NaN.
    """
    df = df_precip.copy().sort_values("date").reset_index(drop=True)
    df["p_daily"] = df["precip_mm"]
    # min_periods equal to window size ensures no silent zero-filling of missing days
    df["p_3d"] = df["precip_mm"].rolling(window=3, min_periods=3).sum()
    df["p_7d"] = df["precip_mm"].rolling(window=7, min_periods=7).sum()
    return df


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE, Bias (prediction - observation), and ubRMSE.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = predicted - actual
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))
    ubrmse = float(np.sqrt(np.maximum(0.0, rmse ** 2 - bias ** 2)))

    return {
        "rmse": rmse,
        "bias": bias,
        "ubrmse": ubrmse,
    }


def fit_ols_regression(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Fit Ordinary Least Squares multiple linear regression: y = beta_0 + X * beta.
    Uses numpy.linalg.lstsq on training data only.

    Returns
    -------
    Tuple[float, np.ndarray]
        (intercept, coefficients)
    """
    X_design = np.column_stack([np.ones(len(X_train)), X_train])
    weights, _, _, _ = np.linalg.lstsq(X_design, y_train, rcond=None)
    intercept = float(weights[0])
    coefficients = np.asarray(weights[1:], dtype=float)
    return intercept, coefficients


def predict_ols_regression(X_test: np.ndarray, intercept: float, coefficients: np.ndarray) -> np.ndarray:
    """
    Generate predictions using fitted OLS weights.
    """
    return intercept + X_test @ coefficients


def run_precipitation_baseline(
    precip_zip_path: Path,
    cosmos_daily_path: Path,
    output_predictions_path: Path,
    target_col: str = "cosmos_sm_median",
    feature_names = ("p_daily", "p_3d", "p_7d"),
) -> Dict[str, object]:
    """
    Execute full precipitation baseline workflow.
    """
    # 1. Load and cross-fill precipitation
    df_p = load_raw_ismn_precipitation(precip_zip_path)
    df_features = build_precipitation_features(df_p)

    # 2. Load COSMOS daily dataset
    df_cosmos = pd.read_csv(cosmos_daily_path)

    # Merge on calendar date
    merged = df_cosmos[["date", "year", "doy", target_col]].merge(df_features, on="date", how="left")

    # 3. Filter valid COSMOS observations (315 dates)
    valid_cosmos = merged[merged[target_col].notnull()].copy().sort_values("date").reset_index(drop=True)

    # 4. Standard 80/20 chronological partition matching climatology baseline
    n_total = len(valid_cosmos) # 315
    n_train = int(n_total * 0.8) # 252
    train_full = valid_cosmos.iloc[:n_train].copy().reset_index(drop=True)
    test_df = valid_cosmos.iloc[n_train:].copy().reset_index(drop=True)

    # Check test-set completeness
    for feat in feature_names:
        if test_df[feat].isnull().any():
            raise ValueError(f"Feature '{feat}' has missing values in the test set.")

    # In train set: explicitly exclude rows where precipitation features are incomplete (e.g. early Jan, late Aug)
    complete_train_mask = train_full[list(feature_names)].notnull().all(axis=1)
    train_clean = train_full[complete_train_mask].copy().reset_index(drop=True)
    excluded_train_count = len(train_full) - len(train_clean)

    # 5. Fit OLS regression strictly on clean training data
    X_train = train_clean[list(feature_names)].to_numpy()
    y_train = train_clean[target_col].to_numpy()
    intercept, coefficients = fit_ols_regression(X_train, y_train)

    # 6. Predict on test set
    X_test = test_df[list(feature_names)].to_numpy()
    y_test = test_df[target_col].to_numpy()
    test_preds = predict_ols_regression(X_test, intercept, coefficients)

    test_df["precip_prediction"] = test_preds
    test_df["prediction_error"] = test_df["precip_prediction"] - test_df[target_col]

    # 7. Evaluate test metrics
    metrics = compute_metrics(actual=y_test, predicted=test_preds)

    # 8. Save test predictions CSV
    export_df = test_df[[
        "date",
        "year",
        "doy",
        target_col,
        "precip_prediction",
        "prediction_error",
        "p_daily",
        "p_3d",
        "p_7d",
    ]].rename(columns={target_col: "actual_cosmos_sm"})

    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_predictions_path, index=False)

    summary = {
        "target_col": target_col,
        "train_total_dates": len(train_full),
        "train_used_dates": len(train_clean),
        "train_excluded_dates": excluded_train_count,
        "test_total_dates": len(test_df),
        "train_start_date": str(train_full["date"].iloc[0]),
        "train_end_date": str(train_full["date"].iloc[-1]),
        "test_start_date": str(test_df["date"].iloc[0]),
        "test_end_date": str(test_df["date"].iloc[-1]),
        "intercept": intercept,
        "coefficients": dict(zip(feature_names, coefficients)),
        "rmse": metrics["rmse"],
        "bias": metrics["bias"],
        "ubrmse": metrics["ubrmse"],
        "output_path": str(output_predictions_path),
    }

    return summary


def main():
    project_root = Path(__file__).resolve().parent.parent
    precip_zip_path = Path("E:/MTP Data/P041_MARSHALL2_ISMN_SM_P_2016_2017.zip")
    cosmos_daily_path = project_root / "data" / "processed" / "p041" / "p041_2016_daily_gnss_cosmos_validation.csv"
    output_predictions_path = project_root / "data" / "processed" / "p041" / "p041_2016_precipitation_predictions.csv"

    print("=" * 80)
    print("RUNNING PRECIPITATION-BASED SOIL MOISTURE BASELINE: P041 2016")
    print("=" * 80)

    summary = run_precipitation_baseline(
        precip_zip_path=precip_zip_path,
        cosmos_daily_path=cosmos_daily_path,
        output_predictions_path=output_predictions_path,
        target_col="cosmos_sm_median",
    )

    print("\n[Dataset Partition & Coverage]")
    print(f"  Training period: {summary['train_start_date']} to {summary['train_end_date']}")
    print(f"  Total training dates: {summary['train_total_dates']}")
    print(f"  Clean training dates used for fitting: {summary['train_used_dates']}")
    print(f"  Incomplete training dates excluded:    {summary['train_excluded_dates']}")
    print(f"  Test period: {summary['test_start_date']} to {summary['test_end_date']}")
    print(f"  Test dates:  {summary['test_total_dates']} (strictly preserved)")

    print("\n[Fitted OLS Model Parameters (Fitted on Training Data Only)]")
    print(f"  Intercept: {summary['intercept']:.6f}")
    for k, v in summary['coefficients'].items():
        print(f"  Coefficient {k}: {v:.6f}")

    print("\n[Held-Out Test Set Metrics (N = 63 Days)]")
    print(f"  Test RMSE:             {summary['rmse']:.6f} m^3/m^3")
    print(f"  Test Bias (Pred - Obs):{summary['bias']:+.6f} m^3/m^3")
    print(f"  Test ubRMSE:           {summary['ubrmse']:.6f} m^3/m^3")
    print(f"  Saved predictions to:  {summary['output_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
