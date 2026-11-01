"""
Primary Three-Station Leave-One-Station-Out (LOSO) Cross-Validation and Feature Ablation.
Evaluates spatial generalization of GNSS-IR soil moisture estimation across:
  Fold 1: Train P041 + P043 -> Test P037 (Fremont Pass)
  Fold 2: Train P041 + P037 -> Test P043 (Newcastle)
  Fold 3: Train P037 + P043 -> Test P041 (Marshall)

Methods Evaluated:
  1. Mean Climatology Baseline (Weighted & Unweighted)
  2. Precipitation OLS Baseline (Weighted & Unweighted)
  3. Random Forest (RF): Model A, Model B, Model C (Weighted & Unweighted)
  4. XGBoost: Model A, Model B, Model C (Weighted & Unweighted)

Feature Ablation Sets:
  - Model A: ['rh', 'amp']
  - Model B: ['rh', 'amp', 'phase_cos', 'phase_sin']
  - Model C: ['rh', 'amp', 'phase_cos', 'phase_sin', 'phase_resultant_length', 'clean_arc_count', 'qc_pass_rate']

Safeguards:
  - Target: PBO_H2O surface soil moisture (0-5 cm, Quality Flag 'G').
  - Equal station weighting: w(s, i) = 1 / (2 * N_s).
  - Preprocessing fit strictly on training set only (zero leakage).
  - No target information from held-out station used for tuning or selection.
  - Moving block bootstrap (block length = 7 days, B = 1000) for uncertainty accounting for temporal autocorrelation.
  - Generates comprehensive sample-availability audit for conditional spatio-temporal split (July 1).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HARMONIZED_CSV = PROCESSED_DIR / "loso_harmonized_dataset_median.csv"

OUT_PRED_CSV = PROCESSED_DIR / "loso_primary_spatial_predictions.csv"
OUT_METRICS_CSV = PROCESSED_DIR / "loso_primary_spatial_metrics.csv"
OUT_ABLATION_CSV = PROCESSED_DIR / "loso_feature_ablation_comparison.csv"
OUT_WEIGHTING_CSV = PROCESSED_DIR / "loso_weighting_sensitivity_comparison.csv"
OUT_SPATIOTEMPORAL_CSV = PROCESSED_DIR / "loso_spatiotemporal_split_audit.csv"

# Hyperparameter locks (matching P041 benchmarks)
RF_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 3,
    "max_features": 1.0,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
}

XGB_PARAMS = {
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

FEATURE_SETS = {
    "Model_A": ["rh", "amp"],
    "Model_B": ["rh", "amp", "phase_cos", "phase_sin"],
    "Model_C": ["rh", "amp", "phase_cos", "phase_sin", "phase_resultant_length", "clean_arc_count", "qc_pass_rate"],
}


def block_bootstrap_ci(
    actual: np.ndarray,
    predicted: np.ndarray,
    block_length: int = 7,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute 95% confidence intervals using Moving Block Bootstrap (MBB)
    to account for temporal autocorrelation.
    """
    rng = np.random.RandomState(seed)
    n = len(actual)
    if n < block_length * 2:
        return {}

    n_blocks = int(np.ceil(n / block_length))

    rmse_boot = []
    mae_boot = []
    bias_boot = []
    ubrmse_boot = []
    r_boot = []

    for _ in range(n_bootstraps):
        start_indices = rng.randint(0, n - block_length + 1, size=n_blocks)
        sample_indices = np.concatenate([np.arange(idx, idx + block_length) for idx in start_indices])[:n]

        act_b = actual[sample_indices]
        pred_b = predicted[sample_indices]

        err = pred_b - act_b
        rmse_val = np.sqrt(np.mean(err**2))
        mae_val = np.mean(np.abs(err))
        bias_val = np.mean(err)
        ubrmse_val = np.sqrt(np.mean((err - bias_val)**2))

        std_act = np.std(act_b)
        std_pred = np.std(pred_b)
        if std_act > 1e-9 and std_pred > 1e-9:
            r_val = np.corrcoef(act_b, pred_b)[0, 1]
        else:
            r_val = np.nan

        rmse_boot.append(rmse_val)
        mae_boot.append(mae_val)
        bias_boot.append(bias_val)
        ubrmse_boot.append(ubrmse_val)
        r_boot.append(r_val)

    def get_ci(vals: List[float]) -> Tuple[float, float]:
        v = np.array(vals)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return np.nan, np.nan
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    return {
        "rmse_ci": get_ci(rmse_boot),
        "mae_ci": get_ci(mae_boot),
        "bias_ci": get_ci(bias_boot),
        "ubrmse_ci": get_ci(ubrmse_boot),
        "r_ci": get_ci(r_boot),
    }


def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    block_length: int = 7,
    n_bootstraps: int = 1000,
) -> Dict[str, Any]:
    """Compute hydrological metrics and block-bootstrap 95% confidence intervals."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(actual)

    error = predicted - actual
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    ubrmse = float(np.sqrt(np.mean((error - bias)**2)))

    std_act = float(np.std(actual))
    std_pred = float(np.std(predicted))
    if std_act > 1e-9 and std_pred > 1e-9 and n > 2:
        r = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        r = np.nan

    ci_dict = block_bootstrap_ci(
        actual, predicted, block_length=block_length, n_bootstraps=n_bootstraps
    )

    return {
        "n": int(n),
        "rmse": rmse,
        "rmse_ci_lower": ci_dict.get("rmse_ci", (np.nan, np.nan))[0],
        "rmse_ci_upper": ci_dict.get("rmse_ci", (np.nan, np.nan))[1],
        "mae": mae,
        "mae_ci_lower": ci_dict.get("mae_ci", (np.nan, np.nan))[0],
        "mae_ci_upper": ci_dict.get("mae_ci", (np.nan, np.nan))[1],
        "bias": bias,
        "bias_ci_lower": ci_dict.get("bias_ci", (np.nan, np.nan))[0],
        "bias_ci_upper": ci_dict.get("bias_ci", (np.nan, np.nan))[1],
        "ubrmse": ubrmse,
        "ubrmse_ci_lower": ci_dict.get("ubrmse_ci", (np.nan, np.nan))[0],
        "ubrmse_ci_upper": ci_dict.get("ubrmse_ci", (np.nan, np.nan))[1],
        "pearson_r": r,
        "r_ci_lower": ci_dict.get("r_ci", (np.nan, np.nan))[0],
        "r_ci_upper": ci_dict.get("r_ci", (np.nan, np.nan))[1],
    }


def execute_primary_loso():
    print("=" * 85)
    print("EXECUTING PRIMARY SPATIAL LOSO EXPERIMENT ACROSS THREE STATIONS")
    print("=" * 85)

    df_full = pd.read_csv(HARMONIZED_CSV)
    df_full["date"] = pd.to_datetime(df_full["date"])
    df = df_full[df_full["usable_for_loso"]].sort_values(["station", "date"]).reset_index(drop=True)

    print(f"Loaded usable dataset: {len(df)} total station-day observations across 3 stations.")
    for st in ["p041", "p037", "p043"]:
        cnt = len(df[df["station"] == st])
        print(f"  - Station {st.upper()}: {cnt} usable observations ({cnt/len(df)*100:.1f}%)")

    # Define Folds
    folds = [
        {"fold_id": "Fold_1", "test_station": "p037", "train_stations": ["p041", "p043"]},
        {"fold_id": "Fold_2", "test_station": "p043", "train_stations": ["p041", "p037"]},
        {"fold_id": "Fold_3", "test_station": "p041", "train_stations": ["p037", "p043"]},
    ]

    all_pred_dfs = []
    all_metrics = []

    for fold in folds:
        fold_id = fold["fold_id"]
        test_st = fold["test_station"]
        train_sts = fold["train_stations"]

        print(f"\n" + "-" * 75)
        print(f"RUNNING {fold_id.upper()}: Test on {test_st.upper()} | Train on {[s.upper() for s in train_sts]}")
        print("-" * 75)

        train_df = df[df["station"].isin(train_sts)].copy().reset_index(drop=True)
        test_df = df[df["station"] == test_st].copy().reset_index(drop=True)

        n_train = len(train_df)
        n_test = len(test_df)
        print(f"  Training set: {n_train} observations | Test set: {n_test} observations")

        # 1. Compute Equal Station Sample Weights
        # w_raw(s, i) = 1 / (2 * N_s), so each station contributes exactly 50% of total loss.
        # Scaled by n_train (sum = n_train, mean = 1.0) so that gradient/hessian-based split criteria
        # (e.g. XGBoost min_child_weight=1) operate at the standard sample count scale while preserving
        # the exact 50.0% / 50.0% loss contribution per station.
        st1, st2 = train_sts[0], train_sts[1]
        n_st1 = len(train_df[train_df["station"] == st1])
        n_st2 = len(train_df[train_df["station"] == st2])
        w_st1 = (n_train / 2.0) / n_st1
        w_st2 = (n_train / 2.0) / n_st2

        train_df["sample_weight_equal"] = np.where(
            train_df["station"] == st1, w_st1, w_st2
        )
        train_df["sample_weight_unweighted"] = 1.0

        print(f"  Sample weights: w({st1.upper()}) = {w_st1:.6f} (N={n_st1}, sum={w_st1*n_st1:.1f}), w({st2.upper()}) = {w_st2:.6f} (N={n_st2}, sum={w_st2*n_st2:.1f})")

        y_train = train_df["pbo_sm"].to_numpy()
        y_test = test_df["pbo_sm"].to_numpy()

        # Initialize prediction record for this test station
        pred_record = test_df[["station", "name", "date", "year", "doy", "pbo_sm", "precip_mm"]].copy()
        pred_record["fold"] = fold_id

        # -------------------------------------------------------------
        # 1. Climatology Baseline
        # -------------------------------------------------------------
        for weight_type, w_col in [("weighted", "sample_weight_equal"), ("unweighted", "sample_weight_unweighted")]:
            weights = train_df[w_col].to_numpy()
            clim_pred_val = float(np.sum(weights * y_train) / np.sum(weights))
            pred_col = f"pred_climatology_{weight_type}"
            pred_record[pred_col] = clim_pred_val

            m = compute_metrics(y_test, np.full(n_test, clim_pred_val))
            m.update({
                "fold": fold_id,
                "test_station": test_st.upper(),
                "model_family": "Climatology",
                "feature_set": "Baseline",
                "weighting": weight_type,
            })
            all_metrics.append(m)

        # -------------------------------------------------------------
        # 2. Precipitation OLS Baseline
        # -------------------------------------------------------------
        X_p_train = train_df[["precip_mm"]].fillna(0.0).to_numpy()
        X_p_test = test_df[["precip_mm"]].fillna(0.0).to_numpy()

        for weight_type, w_col in [("weighted", "sample_weight_equal"), ("unweighted", "sample_weight_unweighted")]:
            weights = train_df[w_col].to_numpy()
            ols = LinearRegression()
            ols.fit(X_p_train, y_train, sample_weight=weights)
            p_preds = ols.predict(X_p_test)
            pred_col = f"pred_precip_ols_{weight_type}"
            pred_record[pred_col] = p_preds

            m = compute_metrics(y_test, p_preds)
            m.update({
                "fold": fold_id,
                "test_station": test_st.upper(),
                "model_family": "Precipitation_OLS",
                "feature_set": "Baseline",
                "weighting": weight_type,
            })
            all_metrics.append(m)

        # -------------------------------------------------------------
        # 3. Machine Learning: Random Forest & XGBoost across Ablation Sets
        # -------------------------------------------------------------
        for model_code, f_cols in FEATURE_SETS.items():
            X_tr = train_df[f_cols].to_numpy()
            X_te = test_df[f_cols].to_numpy()

            # Verify no NaNs in features
            assert not np.isnan(X_tr).any(), f"NaN found in X_tr for {model_code}"
            assert not np.isnan(X_te).any(), f"NaN found in X_te for {model_code}"

            for weight_type, w_col in [("weighted", "sample_weight_equal"), ("unweighted", "sample_weight_unweighted")]:
                weights = train_df[w_col].to_numpy()

                # a) Random Forest
                rf = RandomForestRegressor(**RF_PARAMS)
                rf.fit(X_tr, y_train, sample_weight=weights)
                rf_preds = rf.predict(X_te)
                pred_col_rf = f"pred_rf_{model_code}_{weight_type}"
                pred_record[pred_col_rf] = rf_preds

                m_rf = compute_metrics(y_test, rf_preds)
                m_rf.update({
                    "fold": fold_id,
                    "test_station": test_st.upper(),
                    "model_family": "Random_Forest",
                    "feature_set": model_code,
                    "weighting": weight_type,
                })
                all_metrics.append(m_rf)

                # b) XGBoost
                xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
                xgb_model.fit(X_tr, y_train, sample_weight=weights)
                xgb_preds = xgb_model.predict(X_te)
                pred_col_xgb = f"pred_xgboost_{model_code}_{weight_type}"
                pred_record[pred_col_xgb] = xgb_preds

                m_xgb = compute_metrics(y_test, xgb_preds)
                m_xgb.update({
                    "fold": fold_id,
                    "test_station": test_st.upper(),
                    "model_family": "XGBoost",
                    "feature_set": model_code,
                    "weighting": weight_type,
                })
                all_metrics.append(m_xgb)

        all_pred_dfs.append(pred_record)

    # Consolidate Predictions Table across all 3 Folds
    full_preds_df = pd.concat(all_pred_dfs, ignore_index=True)
    full_preds_df.to_csv(OUT_PRED_CSV, index=False)
    print(f"\n[SAVED] Consolidated LOSO Predictions Table: {OUT_PRED_CSV.relative_to(PROJECT_ROOT)} ({len(full_preds_df)} rows)")

    # -----------------------------------------------------------------
    # Compute Pooled Metrics Across all 771 Samples
    # -----------------------------------------------------------------
    y_all_actual = full_preds_df["pbo_sm"].to_numpy()
    pred_columns = [c for c in full_preds_df.columns if c.startswith("pred_")]

    for p_col in pred_columns:
        # Parse model_family, feature_set, weighting from col name
        # Examples: pred_climatology_weighted, pred_rf_Model_A_weighted
        parts = p_col.split("_")
        if "climatology" in p_col:
            fam = "Climatology"
            f_set = "Baseline"
            w_type = parts[-1]
        elif "precip" in p_col:
            fam = "Precipitation_OLS"
            f_set = "Baseline"
            w_type = parts[-1]
        elif "rf" in p_col:
            fam = "Random_Forest"
            f_set = f"{parts[2]}_{parts[3]}"
            w_type = parts[-1]
        elif "xgboost" in p_col:
            fam = "XGBoost"
            f_set = f"{parts[2]}_{parts[3]}"
            w_type = parts[-1]
        else:
            fam = "Other"
            f_set = "Other"
            w_type = "Other"

        y_all_pred = full_preds_df[p_col].to_numpy()
        m_pooled = compute_metrics(y_all_actual, y_all_pred)
        m_pooled.update({
            "fold": "Pooled_Overall",
            "test_station": "ALL_3_STATIONS",
            "model_family": fam,
            "feature_set": f_set,
            "weighting": w_type,
        })
        all_metrics.append(m_pooled)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT_METRICS_CSV, index=False)
    print(f"[SAVED] Consolidated LOSO Metrics Table: {OUT_METRICS_CSV.relative_to(PROJECT_ROOT)} ({len(metrics_df)} records)")

    # -----------------------------------------------------------------
    # Generate Feature Ablation Comparison Table (Weighted Primary)
    # -----------------------------------------------------------------
    ablation_df = metrics_df[
        (metrics_df["weighting"] == "weighted")
        & (metrics_df["model_family"].isin(["Random_Forest", "XGBoost"]))
    ][["fold", "test_station", "model_family", "feature_set", "n", "rmse", "mae", "bias", "ubrmse", "pearson_r"]].sort_values(["model_family", "fold", "feature_set"])
    ablation_df.to_csv(OUT_ABLATION_CSV, index=False)
    print(f"[SAVED] Feature Ablation Comparison: {OUT_ABLATION_CSV.relative_to(PROJECT_ROOT)}")

    # -----------------------------------------------------------------
    # Generate Station-Weighting Sensitivity Comparison
    # -----------------------------------------------------------------
    weighting_df = metrics_df[
        (metrics_df["feature_set"] == "Model_A")
        | (metrics_df["model_family"].isin(["Climatology", "Precipitation_OLS"]))
    ][["fold", "test_station", "model_family", "feature_set", "weighting", "rmse", "mae", "bias", "ubrmse", "pearson_r"]].sort_values(["model_family", "fold", "weighting"])
    weighting_df.to_csv(OUT_WEIGHTING_CSV, index=False)
    print(f"[SAVED] Weighting Sensitivity Comparison: {OUT_WEIGHTING_CSV.relative_to(PROJECT_ROOT)}")

    return full_preds_df, metrics_df


def audit_spatiotemporal_split():
    print("\n" + "=" * 85)
    print("CONDITIONAL SPATIO-TEMPORAL SAMPLE-AVAILABILITY AUDIT (JULY 1 SPLIT)")
    print("=" * 85)

    df_full = pd.read_csv(HARMONIZED_CSV)
    df_full["date"] = pd.to_datetime(df_full["date"])
    df = df_full[df_full["usable_for_loso"]].sort_values(["station", "date"]).reset_index(drop=True)

    split_date = pd.to_datetime("2016-07-01")

    folds = [
        {"fold": "Fold 1", "test_st": "p037", "train_sts": ["p041", "p043"]},
        {"fold": "Fold 2", "test_st": "p043", "train_sts": ["p041", "p037"]},
        {"fold": "Fold 3", "test_st": "p041", "train_sts": ["p037", "p043"]},
    ]

    records = []

    for f in folds:
        fold_name = f["fold"]
        test_st = f["test_st"]
        st1, st2 = f["train_sts"][0], f["train_sts"][1]

        # Training: early period (Jan 1 - Jun 30) for training stations
        tr_df = df[
            (df["station"].isin(f["train_sts"]))
            & (df["date"] < split_date)
        ]
        tr_st1 = tr_df[tr_df["station"] == st1]
        tr_st2 = tr_df[tr_df["station"] == st2]

        # Testing: late period (Jul 1 - Dec 31) for held-out station
        te_df = df[
            (df["station"] == test_st)
            & (df["date"] >= split_date)
        ]

        # SM range and distribution
        tr_sm = tr_df["pbo_sm"]
        te_sm = te_df["pbo_sm"]

        # Longest gap in test period
        full_te_dates = pd.date_range("2016-07-01", "2016-12-31", freq="D")
        te_usable_dates = te_df["date"].tolist()
        is_u = full_te_dates.isin(te_usable_dates)
        curr_gap, gaps = [], []
        for d, u in zip(full_te_dates, is_u):
            if not u:
                curr_gap.append(d)
            else:
                if curr_gap:
                    gaps.append(curr_gap)
                    curr_gap = []
        if curr_gap:
            gaps.append(curr_gap)
        longest_gap_days = len(max(gaps, key=len)) if gaps else 0

        # Monthly sample counts in test period
        monthly_counts = te_df["date"].dt.month.value_counts().to_dict()

        rec = {
            "Fold": fold_name,
            "Held_Out_Test_Station": test_st.upper(),
            "Train_Stations": f"{st1.upper()} + {st2.upper()}",
            "Train_Total_N": len(tr_df),
            f"Train_{st1.upper()}_N": len(tr_st1),
            f"Train_{st2.upper()}_N": len(tr_st2),
            "Test_N": len(te_df),
            "Test_Period_Total_Days": len(full_te_dates),
            "Test_Retention_Pct": len(te_df) / len(full_te_dates) * 100.0,
            "Test_Longest_Gap_Days": longest_gap_days,
            "Train_SM_Mean": float(tr_sm.mean()),
            "Train_SM_Median": float(tr_sm.median()),
            "Train_SM_Min": float(tr_sm.min()),
            "Train_SM_Max": float(tr_sm.max()),
            "Train_SM_Std": float(tr_sm.std()),
            "Test_SM_Mean": float(te_sm.mean()),
            "Test_SM_Median": float(te_sm.median()),
            "Test_SM_Min": float(te_sm.min()),
            "Test_SM_Max": float(te_sm.max()),
            "Test_SM_Std": float(te_sm.std()),
            "Test_Jul_N": monthly_counts.get(7, 0),
            "Test_Aug_N": monthly_counts.get(8, 0),
            "Test_Sep_N": monthly_counts.get(9, 0),
            "Test_Oct_N": monthly_counts.get(10, 0),
            "Test_Nov_N": monthly_counts.get(11, 0),
            "Test_Dec_N": monthly_counts.get(12, 0),
        }
        records.append(rec)

    spatio_df = pd.DataFrame(records)
    spatio_df.to_csv(OUT_SPATIOTEMPORAL_CSV, index=False)
    print(f"[SAVED] Spatio-Temporal Split Audit: {OUT_SPATIOTEMPORAL_CSV.relative_to(PROJECT_ROOT)}")

    print("\nSPATIO-TEMPORAL SAMPLE AUDIT (SPLIT AT 2016-07-01):")
    for r in records:
        print(f"\n[{r['Fold']}] Test Station: {r['Held_Out_Test_Station']} (Late: Jul 1 - Dec 31)")
        print(f"  Training Set ({r['Train_Stations']}, Early: Jan 1 - Jun 30): N = {r['Train_Total_N']}")
        print(f"  Test Set: N = {r['Test_N']} days ({r['Test_Retention_Pct']:.1f}% coverage) | Longest Gap: {r['Test_Longest_Gap_Days']} days")
        print(f"  Monthly Test Counts: Jul={r['Test_Jul_N']}, Aug={r['Test_Aug_N']}, Sep={r['Test_Sep_N']}, Oct={r['Test_Oct_N']}, Nov={r['Test_Nov_N']}, Dec={r['Test_Dec_N']}")
        print(f"  Train Soil Moisture: Mean={r['Train_SM_Mean']:.4f}, Range=[{r['Train_SM_Min']:.4f}, {r['Train_SM_Max']:.4f}], Std={r['Train_SM_Std']:.4f}")
        print(f"  Test Soil Moisture:  Mean={r['Test_SM_Mean']:.4f}, Range=[{r['Test_SM_Min']:.4f}, {r['Test_SM_Max']:.4f}], Std={r['Test_SM_Std']:.4f}")

    return spatio_df


if __name__ == "__main__":
    execute_primary_loso()
    audit_spatiotemporal_split()
