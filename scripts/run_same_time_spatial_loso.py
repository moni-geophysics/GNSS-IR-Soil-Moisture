"""
Same-Time Spatial Leave-One-Station-Out (LOSO) Cross-Validation and Evaluation.

Objective:
  Evaluate spatial generalization of GNSS-IR soil moisture estimation across stations
  when training and testing observations come from the exact same dates.

Common-Date Dataset:
  - Synchronized across all 3 stations (P041, P037, P043).
  - Strictly requires usable GNSS features and valid PBO_H2O target (flag 'G') at all 3 sites.
  - Exactly 216 common paired dates (2016-02-18 to 2016-11-22).
  - 648 total station-day observations (216 per station).

Folds:
  - Fold 1: Train on P041 + P043 (N=432) -> Test on P037 (N=216)
  - Fold 2: Train on P041 + P037 (N=432) -> Test on P043 (N=216)
  - Fold 3: Train on P037 + P043 (N=432) -> Test on P041 (N=216)
  - Pooled: N_test = 648 paired observations across all 3 stations.

Methods Evaluated:
  1. Mean Climatology Baseline (Weighted & Unweighted)
  2. Precipitation OLS Baseline (Weighted & Unweighted)
  3. Random Forest (RF): Model A (Primary), Model B, Model C (Secondary Ablations)
  4. XGBoost: Model A (Primary), Model B, Model C (Secondary Ablations)

Safeguards & Gap-Respecting Bootstrap:
  - Equal station weighting: w(s, i) = (N_train / 2) / N_{s, train}.
  - Moving Block Bootstrap (MBB) respecting actual chronological calendar gaps (L = 7 calendar days, B = 1000).
  - Comparison with primary all-available-date spatial LOSO and the common-date spatial subset control.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HARMONIZED_CSV = PROCESSED_DIR / "loso_harmonized_dataset_median.csv"
SPATIAL_PREDS_CSV = PROCESSED_DIR / "loso_primary_spatial_predictions.csv"
SPATIAL_METRICS_CSV = PROCESSED_DIR / "loso_primary_spatial_metrics.csv"

OUT_PRED_CSV = PROCESSED_DIR / "loso_same_time_predictions.csv"
OUT_METRICS_CSV = PROCESSED_DIR / "loso_same_time_metrics.csv"
OUT_ABLATION_CSV = PROCESSED_DIR / "loso_same_time_feature_ablation.csv"
OUT_WEIGHTING_CSV = PROCESSED_DIR / "loso_same_time_weighting_sensitivity.csv"
OUT_COMPARISON_CSV = PROCESSED_DIR / "loso_same_time_vs_all_dates_comparison.csv"

# Hyperparameter locks (matching primary LOSO and P041 benchmarks)
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


def make_calendar_gap_blocks(dates: pd.Series | pd.DatetimeIndex, block_days: int = 7) -> List[np.ndarray]:
    """
    Generate candidate block index slices based on calendar days,
    respecting actual non-consecutive chronological gaps.
    """
    dates_dt = pd.to_datetime(dates).to_numpy()
    n = len(dates_dt)
    blocks = []
    one_day_ns = np.timedelta64(1, 'D')
    block_span_ns = np.timedelta64(block_days, 'D')

    for i in range(n):
        j = i
        blk = []
        start_t = dates_dt[i]
        while j < n and (dates_dt[j] - start_t) < block_span_ns:
            blk.append(j)
            j += 1
        blocks.append(np.array(blk, dtype=int))
    return blocks


def calendar_block_bootstrap_ci(
    dates: pd.Series | pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    block_days: int = 7,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute 95% Confidence Intervals using Moving Block Bootstrap
    respecting calendar time gaps to prevent artificial joining across gaps.
    """
    rng = np.random.RandomState(seed)
    n = len(actual)
    if n < block_days * 2:
        return {}

    candidate_blocks = make_calendar_gap_blocks(dates, block_days=block_days)
    n_cands = len(candidate_blocks)

    rmse_boot, mae_boot, bias_boot, ubrmse_boot, r_boot = [], [], [], [], []

    for _ in range(n_bootstraps):
        sampled_indices = []
        while len(sampled_indices) < n:
            rand_idx = rng.randint(0, n_cands)
            sampled_indices.extend(candidate_blocks[rand_idx])
        sample_indices = np.array(sampled_indices[:n], dtype=int)

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
    dates: pd.Series | pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    block_days: int = 7,
    n_bootstraps: int = 1000,
) -> Dict[str, Any]:
    """Compute hydrological metrics and gap-respecting bootstrap 95% confidence intervals."""
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

    ci_dict = calendar_block_bootstrap_ci(
        dates, actual, predicted, block_days=block_days, n_bootstraps=n_bootstraps
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


def execute_same_time_spatial_loso():
    print("=" * 85)
    print("EXECUTING SAME-TIME SPATIAL LOSO EXPERIMENT ACROSS THREE STATIONS")
    print("=" * 85)

    df_full = pd.read_csv(HARMONIZED_CSV)
    df_full["date"] = pd.to_datetime(df_full["date"])
    usable = df_full[df_full["usable_for_loso"]].sort_values(["station", "date"]).reset_index(drop=True)

    # 1. Identify Strictly Common Dates across all 3 stations
    p041_dates = set(usable[usable["station"] == "p041"]["date"])
    p037_dates = set(usable[usable["station"] == "p037"]["date"])
    p043_dates = set(usable[usable["station"] == "p043"]["date"])

    common_dates = sorted(list(p041_dates.intersection(p037_dates).intersection(p043_dates)))
    n_common = len(common_dates)
    print(f"Verified common dates with usable GNSS features and PBO_H2O target across all 3 stations: {n_common}")
    print(f"  Range: {common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}")

    # Filter dataset strictly for common dates
    df = usable[usable["date"].isin(common_dates)].sort_values(["station", "date"]).reset_index(drop=True)
    print(f"Common-date dataset: {len(df)} total paired station-days (216 per station).")

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
        print(f"RUNNING {fold_id.upper()}: Test on {test_st.upper()} | Train on {[s.upper() for s in train_sts]} (Common Dates N={n_common})")
        print("-" * 75)

        train_df = df[df["station"].isin(train_sts)].copy().reset_index(drop=True)
        test_df = df[df["station"] == test_st].copy().reset_index(drop=True)

        n_train = len(train_df)
        n_test = len(test_df)
        assert n_test == n_common, f"Expected {n_common} test samples, got {n_test}"
        assert n_train == 2 * n_common, f"Expected {2*n_common} train samples, got {n_train}"

        # Equal station loss weights
        st1, st2 = train_sts[0], train_sts[1]
        n_st1 = len(train_df[train_df["station"] == st1])
        n_st2 = len(train_df[train_df["station"] == st2])
        w_st1 = (n_train / 2.0) / n_st1
        w_st2 = (n_train / 2.0) / n_st2

        train_df["sample_weight_equal"] = np.where(train_df["station"] == st1, w_st1, w_st2)
        train_df["sample_weight_unweighted"] = 1.0

        print(f"  Training set: {n_train} obs (ST1={n_st1}, ST2={n_st2}) | Test set: {n_test} obs")
        print(f"  Sample weights: w({st1.upper()})={w_st1:.4f}, w({st2.upper()})={w_st2:.4f}")

        y_train = train_df["pbo_sm"].to_numpy()
        y_test = test_df["pbo_sm"].to_numpy()
        test_dates = test_df["date"]

        pred_record = test_df[["station", "name", "date", "year", "doy", "pbo_sm", "precip_mm"]].copy()
        pred_record["fold"] = fold_id
        pred_record["n_common_dates"] = n_common

        # -------------------------------------------------------------
        # 1. Climatology Baseline
        # -------------------------------------------------------------
        for weight_type, w_col in [("weighted", "sample_weight_equal"), ("unweighted", "sample_weight_unweighted")]:
            weights = train_df[w_col].to_numpy()
            clim_pred_val = float(np.sum(weights * y_train) / np.sum(weights))
            pred_col = f"pred_climatology_{weight_type}"
            pred_record[pred_col] = clim_pred_val

            m = compute_metrics(test_dates, y_test, np.full(n_test, clim_pred_val))
            m.update({
                "fold": fold_id,
                "test_station": test_st.upper(),
                "model_family": "Climatology",
                "feature_set": "Baseline",
                "weighting": weight_type,
                "n_common_dates": n_common,
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

            m = compute_metrics(test_dates, y_test, p_preds)
            m.update({
                "fold": fold_id,
                "test_station": test_st.upper(),
                "model_family": "Precipitation_OLS",
                "feature_set": "Baseline",
                "weighting": weight_type,
                "n_common_dates": n_common,
            })
            all_metrics.append(m)

        # -------------------------------------------------------------
        # 3. Machine Learning: Random Forest & XGBoost across Ablations
        # -------------------------------------------------------------
        for model_code, f_cols in FEATURE_SETS.items():
            X_tr = train_df[f_cols].to_numpy()
            X_te = test_df[f_cols].to_numpy()

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

                m_rf = compute_metrics(test_dates, y_test, rf_preds)
                m_rf.update({
                    "fold": fold_id,
                    "test_station": test_st.upper(),
                    "model_family": "Random_Forest",
                    "feature_set": model_code,
                    "weighting": weight_type,
                    "n_common_dates": n_common,
                })
                all_metrics.append(m_rf)

                # b) XGBoost
                xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
                xgb_model.fit(X_tr, y_train, sample_weight=weights)
                xgb_preds = xgb_model.predict(X_te)
                pred_col_xgb = f"pred_xgboost_{model_code}_{weight_type}"
                pred_record[pred_col_xgb] = xgb_preds

                m_xgb = compute_metrics(test_dates, y_test, xgb_preds)
                m_xgb.update({
                    "fold": fold_id,
                    "test_station": test_st.upper(),
                    "model_family": "XGBoost",
                    "feature_set": model_code,
                    "weighting": weight_type,
                    "n_common_dates": n_common,
                })
                all_metrics.append(m_xgb)

        all_pred_dfs.append(pred_record)

    # Consolidate Predictions across all 3 Folds
    full_preds_df = pd.concat(all_pred_dfs, ignore_index=True)
    full_preds_df.to_csv(OUT_PRED_CSV, index=False)
    print(f"\n[SAVED] Same-Time LOSO Predictions Table: {OUT_PRED_CSV.relative_to(PROJECT_ROOT)} ({len(full_preds_df)} rows)")

    # -----------------------------------------------------------------
    # Compute Pooled Metrics Across all 648 Common Observations
    # -----------------------------------------------------------------
    y_all_actual = full_preds_df["pbo_sm"].to_numpy()
    all_dates = full_preds_df["date"]
    pred_columns = [c for c in full_preds_df.columns if c.startswith("pred_")]

    for p_col in pred_columns:
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
        m_pooled = compute_metrics(all_dates, y_all_actual, y_all_pred)
        m_pooled.update({
            "fold": "Pooled_Overall",
            "test_station": "ALL_3_STATIONS",
            "model_family": fam,
            "feature_set": f_set,
            "weighting": w_type,
            "n_common_dates": n_common,
        })
        all_metrics.append(m_pooled)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT_METRICS_CSV, index=False)
    print(f"[SAVED] Same-Time LOSO Metrics Table: {OUT_METRICS_CSV.relative_to(PROJECT_ROOT)} ({len(metrics_df)} records)")

    # -----------------------------------------------------------------
    # Generate Feature Ablation Comparison Table (Weighted Primary)
    # -----------------------------------------------------------------
    ablation_df = metrics_df[
        (metrics_df["weighting"] == "weighted")
        & (metrics_df["model_family"].isin(["Random_Forest", "XGBoost"]))
    ][["fold", "test_station", "model_family", "feature_set", "n", "rmse", "mae", "bias", "ubrmse", "pearson_r"]].sort_values(["model_family", "fold", "feature_set"])
    ablation_df.to_csv(OUT_ABLATION_CSV, index=False)
    print(f"[SAVED] Same-Time Feature Ablation: {OUT_ABLATION_CSV.relative_to(PROJECT_ROOT)}")

    # -----------------------------------------------------------------
    # Generate Station-Weighting Sensitivity Comparison
    # -----------------------------------------------------------------
    weighting_df = metrics_df[
        (metrics_df["feature_set"] == "Model_A")
        | (metrics_df["model_family"].isin(["Climatology", "Precipitation_OLS"]))
    ][["fold", "test_station", "model_family", "feature_set", "weighting", "n", "rmse", "mae", "bias", "ubrmse", "pearson_r"]].sort_values(["model_family", "fold", "weighting"])
    weighting_df.to_csv(OUT_WEIGHTING_CSV, index=False)
    print(f"[SAVED] Same-Time Weighting Sensitivity: {OUT_WEIGHTING_CSV.relative_to(PROJECT_ROOT)}")

    # -----------------------------------------------------------------
    # Cross-Experiment Comparison: Same-Time vs All-Available-Dates Spatial LOSO
    # -----------------------------------------------------------------
    if SPATIAL_METRICS_CSV.exists() and SPATIAL_PREDS_CSV.exists():
        spat_metrics = pd.read_csv(SPATIAL_METRICS_CSV)
        spat_preds = pd.read_csv(SPATIAL_PREDS_CSV)
        spat_preds["date"] = pd.to_datetime(spat_preds["date"])

        comp_records = []
        for fold in ["Fold_1", "Fold_2", "Fold_3", "Pooled_Overall"]:
            for fam in ["Climatology", "Precipitation_OLS", "Random_Forest", "XGBoost"]:
                fset = "Baseline" if "Baseline" in fam or fam in ["Climatology", "Precipitation_OLS"] else "Model_A"
                
                # 1. Full Spatial LOSO metric (All available dates)
                m_spat = spat_metrics[
                    (spat_metrics["fold"] == fold)
                    & (spat_metrics["model_family"] == fam)
                    & (spat_metrics["feature_set"] == fset)
                    & (spat_metrics["weighting"] == "weighted")
                ]

                # 2. Same-Time Spatial LOSO metric (Strictly common dates)
                m_st = metrics_df[
                    (metrics_df["fold"] == fold)
                    & (metrics_df["model_family"] == fam)
                    & (metrics_df["feature_set"] == fset)
                    & (metrics_df["weighting"] == "weighted")
                ]

                # 3. All-Available Spatial LOSO model evaluated on the Common Dates subset (Internal Control)
                if fold != "Pooled_Overall":
                    st_code = {"Fold_1": "p037", "Fold_2": "p043", "Fold_3": "p041"}[fold]
                    spat_sub = spat_preds[(spat_preds["station"] == st_code) & (spat_preds["date"].isin(common_dates))].sort_values("date")
                else:
                    spat_sub = spat_preds[spat_preds["date"].isin(common_dates)].sort_values(["station", "date"])

                p_col_map = {
                    "Climatology": "pred_climatology_weighted",
                    "Precipitation_OLS": "pred_precip_ols_weighted",
                    "Random_Forest": "pred_rf_Model_A_weighted",
                    "XGBoost": "pred_xgboost_Model_A_weighted",
                }
                m_spat_common_subset = compute_metrics(spat_sub["date"], spat_sub["pbo_sm"].to_numpy(), spat_sub[p_col_map[fam]].to_numpy())

                if len(m_spat) > 0 and len(m_st) > 0:
                    rec = {
                        "fold": fold,
                        "model_family": fam,
                        "feature_set": fset,
                        "all_dates_spatial_rmse": m_spat["rmse"].values[0],
                        "all_dates_spatial_ubrmse": m_spat["ubrmse"].values[0],
                        "all_dates_spatial_r": m_spat["pearson_r"].values[0],
                        "all_dates_spatial_n": m_spat["n"].values[0],
                        "all_dates_model_on_common_subset_rmse": m_spat_common_subset["rmse"],
                        "all_dates_model_on_common_subset_ubrmse": m_spat_common_subset["ubrmse"],
                        "all_dates_model_on_common_subset_r": m_spat_common_subset["pearson_r"],
                        "all_dates_model_on_common_subset_n": m_spat_common_subset["n"],
                        "same_time_spatial_rmse": m_st["rmse"].values[0],
                        "same_time_spatial_ubrmse": m_st["ubrmse"].values[0],
                        "same_time_spatial_r": m_st["pearson_r"].values[0],
                        "same_time_spatial_n": m_st["n"].values[0],
                        "delta_rmse_same_time_vs_all_dates": m_st["rmse"].values[0] - m_spat["rmse"].values[0],
                        "delta_rmse_same_time_vs_common_subset_control": m_st["rmse"].values[0] - m_spat_common_subset["rmse"],
                    }
                    comp_records.append(rec)

        comp_df = pd.DataFrame(comp_records)
        comp_df.to_csv(OUT_COMPARISON_CSV, index=False)
        print(f"[SAVED] Same-Time vs All-Dates Comparison Table: {OUT_COMPARISON_CSV.relative_to(PROJECT_ROOT)}")

    return full_preds_df, metrics_df


if __name__ == "__main__":
    execute_same_time_spatial_loso()
